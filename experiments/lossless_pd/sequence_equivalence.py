"""Audit batched speculative verification against sequential target execution.

Both paths start from bit-identical prompt KV and consume the same teacher-forced
candidate tokens.  This isolates numerical shape effects from proposal quality
and from P-side replay drift.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from experiments.lossless_pd.pilot import load_requests, load_target, write_json
from experiments.lossless_pd.shape_invariant_attention import (
    register_shape_invariant_attention,
    row_invariant_verifier_ops,
)
from experiments.lossless_pd.verifier_probe import cache_with_buffers


def fresh_cache(target, host):
    buffers = [(key.to("cuda"), value.to("cuda")) for key, value in host]
    return cache_with_buffers(target, buffers)


def greedy_commit(proposals, target_argmax):
    """Standard greedy speculative output: accepted prefix plus one target token."""

    for index, proposed in enumerate(proposals):
        if proposed != target_argmax[index]:
            return proposals[:index] + [target_argmax[index]], index
    return proposals + [target_argmax[len(proposals)]], len(proposals)


def rowwise_target(target, block, cache):
    """Layer-major oracle with the endpoint's qlen=1 arithmetic shape.

    This is intentionally unfused and slow. It tests whether a future kernel
    can parallelize across rows without changing each row's reduction shape.
    """

    prompt_tokens = cache.layers[0].keys.shape[-2]
    hidden = target.model.embed_tokens(block)
    positions = torch.arange(prompt_tokens, prompt_tokens + block.shape[1], device="cuda")
    rotary = target.model.rotary_emb(hidden, positions[None])
    for layer in target.model.layers:
        next_hidden = []
        for index in range(block.shape[1]):
            token_position = positions[index:index + 1]
            token_rotary = tuple(value[:, index:index + 1] for value in rotary)
            token_hidden = layer(
                hidden[:, index:index + 1], attention_mask=None,
                position_ids=token_position[None], past_key_values=cache,
                use_cache=True, cache_position=token_position,
                position_embeddings=token_rotary,
            )
            next_hidden.append(token_hidden)
        hidden = torch.cat(next_hidden, dim=1)
    logits = []
    for index in range(block.shape[1]):
        normalized = target.model.norm(hidden[:, index:index + 1])
        logits.append(target.lm_head(normalized).float())
    return torch.cat(logits, dim=1), cache


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--proposal-tokens", type=int, default=15)
    parser.add_argument(
        "--attention", choices=["eager", "sdpa", "shape_invariant"], default="eager"
    )
    parser.add_argument("--verification-shape", choices=["batched", "rowwise"],
                        default="batched")
    args = parser.parse_args()
    if args.requests <= 0 or args.proposal_tokens < 0:
        parser.error("invalid request or proposal count")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    target = load_target(args.target)
    data = json.loads(Path(args.proposals).read_text())
    proposals = [row for row in data["rows"] if row["order"] == "priority"
                 and row["fraction"] == .1 and row["memory_mode"] == "exact"]
    proposals = proposals[:args.requests]
    rows = []
    for completed, proposal in enumerate(proposals):
        request = load_requests(proposal["source"], proposal["source_index"] + 1,
                                proposal["prompt_tokens"])[-1]
        ids = torch.tensor([request["tokens"]], device="cuda")
        target.config._attn_implementation = "sdpa"
        prompt = target.model(ids, use_cache=True)
        seed = int(target.lm_head(prompt.last_hidden_state[:, -1]).argmax(-1).item())
        if seed != proposal["seed"]:
            raise RuntimeError("proposal P seed differs")
        host = [(layer.keys.cpu(), layer.values.cpu()) for layer in prompt.past_key_values.layers]
        block = torch.tensor(
            [[seed] + proposal["proposal"][:args.proposal_tokens]], device="cuda"
        )
        target.config._attn_implementation = (
            register_shape_invariant_attention()
            if args.attention == "shape_invariant" else args.attention
        )

        block_cache = fresh_cache(target, host)
        torch.cuda.synchronize()
        block_started = time.perf_counter()
        exact_linears = (
            row_invariant_verifier_ops(target)
            if args.attention == "shape_invariant" else nullcontext()
        )
        with exact_linears:
            if args.verification_shape == "batched":
                block_output = target.model(block, past_key_values=block_cache, use_cache=True)
                block_logits = target.lm_head(block_output.last_hidden_state).float()
            else:
                block_logits, block_cache = rowwise_target(target, block, block_cache)
                block_output = None
        torch.cuda.synchronize()
        block_ms = (time.perf_counter() - block_started) * 1000
        block_tail = [(layer.keys[..., -block.shape[1]:, :].clone(),
                       layer.values[..., -block.shape[1]:, :].clone())
                      for layer in block_cache.layers]

        sequence_cache = fresh_cache(target, host)
        sequence_logits = []
        sequence_tail = [[] for _ in target.model.layers]
        torch.cuda.synchronize()
        sequence_started = time.perf_counter()
        for position in range(block.shape[1]):
            step = target.model(block[:, position:position + 1],
                                past_key_values=sequence_cache, use_cache=True)
            sequence_logits.append(target.lm_head(step.last_hidden_state).float())
            for layer_id, layer in enumerate(sequence_cache.layers):
                sequence_tail[layer_id].append((layer.keys[..., -1:, :].clone(),
                                                layer.values[..., -1:, :].clone()))
        torch.cuda.synchronize()
        sequence_ms = (time.perf_counter() - sequence_started) * 1000
        sequential_logits = torch.cat(sequence_logits, dim=1)
        sequential_tail = [
            (torch.cat([pair[0] for pair in layer], dim=-2),
             torch.cat([pair[1] for pair in layer], dim=-2))
            for layer in sequence_tail
        ]
        tail_pairs = [(a, b) for left, right in zip(block_tail, sequential_tail)
                      for a, b in zip(left, right)]
        argmax_left = block_logits.argmax(-1)
        argmax_right = sequential_logits.argmax(-1)
        first_argmax_difference = next(
            (i for i in range(block.shape[1])
             if int(argmax_left[0, i]) != int(argmax_right[0, i])), None
        )
        proposed = block[0, 1:].tolist()
        batched_argmax = argmax_left[0].tolist()
        sequential_argmax = argmax_right[0].tolist()
        batched_commit, batched_accepted = greedy_commit(proposed, batched_argmax)
        sequential_commit, sequential_accepted = greedy_commit(proposed, sequential_argmax)
        row = {
            "record_id": proposal["record_id"],
            "context": ids.shape[1],
            "input_tokens": block.shape[1],
            "verification_shape": args.verification_shape,
            "verification_ms": block_ms,
            "sequential_ms": sequence_ms,
            "logits_bitwise_equal": torch.equal(block_logits, sequential_logits),
            "generated_kv_bitwise_equal": all(torch.equal(a, b) for a, b in tail_pairs),
            "argmax_equal": torch.equal(argmax_left, argmax_right),
            "first_argmax_difference": first_argmax_difference,
            "proposal": proposed,
            "batched_argmax": batched_argmax,
            "sequential_argmax": sequential_argmax,
            "batched_accepted": batched_accepted,
            "sequential_accepted": sequential_accepted,
            "batched_commit": batched_commit,
            "sequential_commit": sequential_commit,
            "committed_output_equal": batched_commit == sequential_commit,
            "max_logit_abs_error": float((block_logits - sequential_logits).abs().max()),
            "max_generated_kv_abs_error": max(float((a - b).abs().max())
                                               for a, b in tail_pairs),
        }
        rows.append(row)
        write_json(output / "progress.json", {"rows": rows})
        print(json.dumps({"event": "sequence_equivalence", "completed": completed + 1,
                          "total": len(proposals), "argmax_equal": row["argmax_equal"]}),
              flush=True)
        del prompt, host, block_cache, block_output, block_logits, block_tail
        del sequence_cache, sequence_logits, sequence_tail, sequential_logits, sequential_tail
    summary = {
        "requests": len(rows),
        "proposal_tokens": args.proposal_tokens,
        "verifier_input_tokens": args.proposal_tokens + 1,
        "attention": args.attention,
        "verification_shape": args.verification_shape,
        "logits_bitwise_equal": all(row["logits_bitwise_equal"] for row in rows),
        "generated_kv_bitwise_equal": all(row["generated_kv_bitwise_equal"] for row in rows),
        "argmax_equal": all(row["argmax_equal"] for row in rows),
        "argmax_mismatch_requests": sum(not row["argmax_equal"] for row in rows),
        "committed_output_equal": all(row["committed_output_equal"] for row in rows),
        "committed_output_mismatch_requests": sum(
            not row["committed_output_equal"] for row in rows
        ),
        "max_logit_abs_error": max(row["max_logit_abs_error"] for row in rows),
        "max_generated_kv_abs_error": max(row["max_generated_kv_abs_error"] for row in rows),
        "mean_verification_ms": sum(row["verification_ms"] for row in rows) / len(rows),
        "mean_sequential_ms": sum(row["sequential_ms"] for row in rows) / len(rows),
        "contract": (
            "same prompt KV and teacher-forced candidate tokens; batched versus sequential "
            "target arithmetic; this numerical audit supplements rather than replaces the "
            "speculative-decoding distribution proof"
        ),
    }
    write_json(output / "results.json", {"summary": summary, "rows": rows})
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
