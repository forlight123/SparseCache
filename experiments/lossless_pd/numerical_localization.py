"""Locate the first BF16 arithmetic drift in block verification.

This diagnostic compares one batched Target pass with token-by-token Target
execution from bit-identical prompt KV.  It records module outputs, not merely
the final logits, so a verifier-kernel design can target the first operation
whose arithmetic shape changes the endpoint.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from experiments.lossless_pd.pilot import load_requests, load_target, write_json
from experiments.lossless_pd.sequence_equivalence import fresh_cache, greedy_commit
from experiments.lossless_pd.shape_invariant_attention import (
    register_shape_invariant_attention,
    row_invariant_verifier_ops,
)


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def traced_modules(target):
    """Return (name, module, capture-input) probes in execution order."""

    modules = [("embed_tokens", target.model.embed_tokens, False)]
    for layer_id, layer in enumerate(target.model.layers):
        prefix = f"layer_{layer_id:02d}"
        modules.extend([
            (f"{prefix}.input_layernorm", layer.input_layernorm, False),
            (f"{prefix}.q_proj", layer.self_attn.q_proj, False),
            (f"{prefix}.k_proj", layer.self_attn.k_proj, False),
            (f"{prefix}.v_proj", layer.self_attn.v_proj, False),
            # The o_proj input is the attention reduction result.  Capturing
            # both sides distinguishes attention drift from GEMM-shape drift.
            (f"{prefix}.attention_result", layer.self_attn.o_proj, True),
            (f"{prefix}.o_proj", layer.self_attn.o_proj, False),
            (f"{prefix}.post_attention_layernorm", layer.post_attention_layernorm, False),
            (f"{prefix}.gate_proj", layer.mlp.gate_proj, False),
            (f"{prefix}.up_proj", layer.mlp.up_proj, False),
            (f"{prefix}.down_proj", layer.mlp.down_proj, False),
            (f"{prefix}.output", layer, False),
        ])
    modules.extend([
        ("final_norm", target.model.norm, False),
        ("lm_head", target.lm_head, False),
    ])
    return modules


@contextmanager
def capture_outputs(modules):
    captured = {name: [] for name, _, _ in modules}
    handles = []

    def hook(name):
        def save(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                raise TypeError(f"{name} did not return a tensor-containing output")
            captured[name].append(tensor.detach().cpu().clone())
        return save

    try:
        for name, module, capture_input in modules:
            if capture_input:
                handles.append(module.register_forward_pre_hook(
                    lambda _module, inputs, name=name: hook(name)(
                        _module, inputs, inputs
                    )
                ))
            else:
                handles.append(module.register_forward_hook(hook(name)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def concatenate_calls(name, calls, expected_calls):
    if len(calls) != expected_calls:
        raise RuntimeError(f"{name}: expected {expected_calls} calls, saw {len(calls)}")
    return calls[0] if expected_calls == 1 else torch.cat(calls, dim=1)


def compare_trace(modules, batched, sequential, tokens):
    rows = []
    first_drift = None
    for name, _, _ in modules:
        left = concatenate_calls(name, batched[name], 1)
        right = concatenate_calls(name, sequential[name], tokens)
        if left.shape != right.shape:
            raise RuntimeError(f"{name}: shape mismatch {left.shape} versus {right.shape}")
        difference = (left.float() - right.float()).abs()
        per_token = difference.reshape(1, tokens, -1).amax(dim=-1)[0]
        unequal = per_token != 0
        first_token = int(unequal.nonzero()[0]) if bool(unequal.any()) else None
        row = {
            "module": name,
            "shape": list(left.shape),
            "bitwise_equal": torch.equal(left, right),
            "first_different_token": first_token,
            "different_tokens": int(unequal.sum()),
            "max_abs_error": float(difference.max()),
            "per_token_max_abs_error": [float(value) for value in per_token],
        }
        rows.append(row)
        if first_drift is None and not row["bitwise_equal"]:
            first_drift = row
    return rows, first_drift


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--request-indices", default="9,44")
    parser.add_argument("--proposal-tokens", type=int, default=15)
    parser.add_argument(
        "--attention", choices=["eager", "shape_invariant"], default="eager"
    )
    args = parser.parse_args()
    indices = [int(value) for value in args.request_indices.split(",") if value]
    if not indices or min(indices) < 0 or args.proposal_tokens < 0:
        parser.error("request indices and proposal count must be nonnegative")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    target = load_target(args.target)
    target.config._attn_implementation = "eager"
    modules = traced_modules(target)
    payload = json.loads(Path(args.proposals).read_text())
    candidates = {
        row["source_index"]: row for row in payload["rows"]
        if row["order"] == "priority" and row["fraction"] == .1
        and row["memory_mode"] == "exact" and row["source_index"] in indices
    }
    if set(candidates) != set(indices):
        raise ValueError(f"missing requested proposal rows: {set(indices) - set(candidates)}")

    results = []
    for request_index in indices:
        proposal = candidates[request_index]
        request = load_requests(
            proposal["source"], request_index + 1, proposal["prompt_tokens"]
        )[-1]
        ids = torch.tensor([request["tokens"]], device="cuda")
        # Prompt execution is outside the hooks and shared by both verifier paths.
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
            if args.attention == "shape_invariant" else "eager"
        )

        block_cache = fresh_cache(target, host)
        exact_linears = (
            row_invariant_verifier_ops(target)
            if args.attention == "shape_invariant" else nullcontext()
        )
        with exact_linears:
            with capture_outputs(modules) as batched:
                block_output = target.model(
                    block, past_key_values=block_cache, use_cache=True
                )
                block_logits = target.lm_head(block_output.last_hidden_state).float()

        sequence_cache = fresh_cache(target, host)
        sequence_logits = []
        with capture_outputs(modules) as sequential:
            for position in range(block.shape[1]):
                step = target.model(
                    block[:, position:position + 1],
                    past_key_values=sequence_cache,
                    use_cache=True,
                )
                sequence_logits.append(target.lm_head(step.last_hidden_state).float())
        sequential_logits = torch.cat(sequence_logits, dim=1)
        trace, first_drift = compare_trace(
            modules, batched, sequential, block.shape[1]
        )
        left_argmax = block_logits.argmax(-1)[0].tolist()
        right_argmax = sequential_logits.argmax(-1)[0].tolist()
        proposed = block[0, 1:].tolist()
        left_commit, left_accepted = greedy_commit(proposed, left_argmax)
        right_commit, right_accepted = greedy_commit(proposed, right_argmax)
        result = {
            "record_id": proposal["record_id"],
            "context_tokens": ids.shape[1],
            "verifier_input_tokens": block.shape[1],
            "first_drift": first_drift,
            "final_logits_bitwise_equal": torch.equal(block_logits, sequential_logits),
            "max_final_logit_abs_error": float(
                (block_logits - sequential_logits).abs().max()
            ),
            "batched_accepted": left_accepted,
            "sequential_accepted": right_accepted,
            "committed_output_equal": left_commit == right_commit,
            "trace": trace,
        }
        results.append(result)
        write_json(output / "progress.json", {"rows": results})
        print(json.dumps({
            "record_id": result["record_id"],
            "first_drift": first_drift["module"] if first_drift else None,
            "committed_output_equal": result["committed_output_equal"],
        }), flush=True)

    summary = {
        "requests": len(results),
        "first_drift_modules": [
            row["first_drift"]["module"] if row["first_drift"] else None
            for row in results
        ],
        "committed_output_mismatches": sum(
            not row["committed_output_equal"] for row in results
        ),
        "interpretation": (
            "The earliest unequal module identifies arithmetic-shape drift, not an "
            "algorithmic failure of standard speculative acceptance."
        ),
    }
    write_json(output / "results.json", {"summary": summary, "rows": results})
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
