"""Screen a pretrained small-model proposer against immutable Target packets.

This is the stage-zero gate for a structurally cheaper SparseCache proposer.
The small model receives the prompt token IDs already present at the decoder
and the exact producer seed, then autoregressively proposes a fixed suffix.
No Target KV adapter is used in this control.  References come exclusively
from immutable full-Target packets, so the screen never reruns the verifier or
silently changes its numerical trajectory.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from experiments.lossless_pd.pilot import bootstrap_delta
from experiments.lossless_pd.reference_packets import (
    digest_file,
    load_packet,
    read_index,
)


def accepted_prefix(
    proposal: list[int], reference: list[int], stop_ids: set[int]
) -> int:
    """Count the exact greedy prefix, stopping at the Target's EOS token."""

    accepted = 0
    for candidate, target in zip(proposal, reference, strict=False):
        if candidate != target:
            break
        accepted += 1
        if target in stop_ids:
            break
    return accepted


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile needs at least one value")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty screen")
    accepted = [int(row["accepted"]) for row in rows]
    prefill = [
        float(row["prefill_ms"]) for row in rows if row.get("prefill_ms") is not None
    ]
    decode = [
        float(row["draft_decode_ms"])
        for row in rows
        if row.get("draft_decode_ms") is not None
    ]
    total = [float(row["total_draft_ms"]) for row in rows]
    horizon = int(rows[0]["draft_tokens"])
    if any(int(row["draft_tokens"]) != horizon for row in rows):
        raise ValueError("screen rows disagree on the draft horizon")
    result = {
        "requests": len(rows),
        "draft_tokens": horizon,
        "mean_accepted_prefix": statistics.fmean(accepted),
        "accepted_prefix_ci95": bootstrap_delta(accepted),
        "zero_acceptance_rate": sum(value == 0 for value in accepted) / len(rows),
        "full_acceptance_rate": sum(value == horizon for value in accepted) / len(rows),
        "positive_acceptance_requests": sum(value > 0 for value in accepted),
        "mean_prefill_ms": statistics.fmean(prefill) if prefill else None,
        "p95_prefill_ms": percentile(prefill, 0.95) if prefill else None,
        "mean_draft_decode_ms": statistics.fmean(decode) if decode else None,
        "p95_draft_decode_ms": percentile(decode, 0.95) if decode else None,
        "mean_total_draft_ms": statistics.fmean(total),
        "p95_total_draft_ms": percentile(total, 0.95),
        "accepted_tokens_per_total_draft_ms": (
            statistics.fmean(accepted) / statistics.fmean(total)
        ),
    }
    branch_rows = [row for row in rows if "seed_branch_hit" in row]
    if branch_rows:
        if len(branch_rows) != len(rows):
            raise ValueError("screen rows mix seed-branch contracts")
        result["seed_branch_hit_rate"] = sum(
            bool(row["seed_branch_hit"]) for row in rows
        ) / len(rows)
    return result


@torch.inference_mode()
def generate_suffix(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    seed_token_id: int,
    draft_tokens: int,
) -> tuple[list[int], float, float]:
    """Prefill once, then draft after the authoritative producer seed."""

    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - started) * 1000

    cache = output.past_key_values
    token = torch.tensor([[seed_token_id]], dtype=torch.long, device=prompt_ids.device)
    proposal: list[int] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(draft_tokens):
        output = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        proposal.append(int(token.item()))
        cache = output.past_key_values
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - started) * 1000
    return proposal, prefill_ms, decode_ms


def write_result(
    args: argparse.Namespace,
    packet_root: Path,
    rows: list[dict[str, Any]],
    *,
    runtime: str,
) -> None:
    result = {
        "schema_version": 1,
        "purpose": (
            "unadapted small-model lossless-pipeline screen against immutable "
            "full-Target greedy references"
        ),
        "runtime": runtime,
        "proposal_origin": args.proposal_origin,
        "draft_model": str(args.draft_model.resolve()),
        "packet_root": str(packet_root),
        "packet_index_sha256": digest_file(packet_root / "index.json"),
        "attention_implementation": (
            args.attention_implementation if runtime == "transformers" else "vllm"
        ),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2))


@torch.inference_mode()
def run_vllm(
    args: argparse.Namespace,
    packet_root: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Measure the same proposer with vLLM's serving-grade dense runtime."""

    from vllm import LLM, SamplingParams

    max_prompt_tokens = max(int(entry["prompt_tokens"]) for entry in entries)
    concurrent_branch = args.proposal_origin == "concurrent_seed_branch"
    generated_tokens = args.draft_tokens + int(concurrent_branch)
    engine = LLM(
        model=str(args.draft_model),
        tokenizer=str(args.draft_model),
        dtype="bfloat16",
        max_model_len=max_prompt_tokens + generated_tokens + 2,
        max_num_seqs=1,
        block_size=64,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=generated_tokens,
        ignore_eos=True,
    )

    def request_for(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        packet = load_packet(packet_root, entry)
        prompt = packet["prompt_ids"][0].tolist()
        if not concurrent_branch:
            prompt.append(int(packet["seed"].item()))
        return packet, {"prompt_token_ids": prompt}

    # Compile and warm the exact long-context/decode geometry outside timing.
    warm_packet, warm_request = request_for(entries[0])
    engine.generate([warm_request], sampling, use_tqdm=False)
    del warm_packet, warm_request

    rows: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(entries):
        packet, request = request_for(entry)
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate([request], sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        generated = list(outputs[0].outputs[0].token_ids)
        if len(generated) != generated_tokens:
            raise RuntimeError("vLLM did not return the fixed draft horizon")
        seed_token_id = int(packet["seed"].item())
        seed_branch_hit = not concurrent_branch or generated[0] == seed_token_id
        proposal = generated[1:] if concurrent_branch else generated
        reference = packet["reference"][: args.draft_tokens].tolist()
        stop_ids = set(packet["stop_ids"].tolist())
        rows.append(
            {
                "ordinal": ordinal,
                "record_id": packet["record_id"],
                "packet_sha256": entry["sha256"],
                "input_sha256": packet["input_sha256"],
                "prompt_tokens": int(packet["prompt_ids"].shape[1]),
                "seed_token_id": seed_token_id,
                "seed_branch_hit": seed_branch_hit,
                "generated_seed_branch": generated if concurrent_branch else None,
                "reference": reference,
                "proposal": proposal,
                "accepted": (
                    accepted_prefix(proposal, reference, stop_ids)
                    if seed_branch_hit
                    else 0
                ),
                "draft_tokens": args.draft_tokens,
                "prefill_ms": None,
                "draft_decode_ms": None,
                "total_draft_ms": elapsed_ms,
            }
        )
        print(
            json.dumps(
                {
                    "event": "small_draft_vllm_screen",
                    "completed": ordinal + 1,
                    "total": len(entries),
                    "accepted": rows[-1]["accepted"],
                }
            ),
            flush=True,
        )
    write_result(args, packet_root, rows, runtime="vllm")


def run(args: argparse.Namespace) -> None:
    packet_root = args.packets.resolve()
    index = read_index(packet_root)
    entries = index["entries"][args.request_offset :]
    if args.num_requests is not None:
        entries = entries[: args.num_requests]
    if not entries:
        raise ValueError("the requested immutable packet slice is empty")
    if args.draft_tokens > int(index["draft_tokens"]):
        raise ValueError("draft horizon exceeds the immutable Target reference")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.runtime == "vllm":
        run_vllm(args, packet_root, entries)
        return
    if args.proposal_origin != "after_target_seed":
        raise ValueError("concurrent seed-branch screen currently requires vLLM")

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=args.attention_implementation,
        )
        .to("cuda")
        .eval()
        .requires_grad_(False)
    )
    if int(model.config.vocab_size) <= 0:
        raise ValueError("draft model has an invalid vocabulary")

    # Warm kernels without contaminating a measured packet or allocating its
    # full context cache twice.
    warmup_length = min(args.warmup_tokens, int(model.config.max_position_embeddings))
    warmup = torch.arange(warmup_length, device="cuda")[None] % int(
        model.config.vocab_size
    )
    generate_suffix(model, warmup, args.warmup_seed_token, args.draft_tokens)
    del warmup
    torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(entries):
        packet = load_packet(packet_root, entry)
        prompt_ids = packet["prompt_ids"].to("cuda", non_blocking=True)
        seed_token_id = int(packet["seed"].item())
        reference = packet["reference"][: args.draft_tokens].tolist()
        stop_ids = set(packet["stop_ids"].tolist())
        proposal, prefill_ms, decode_ms = generate_suffix(
            model, prompt_ids, seed_token_id, args.draft_tokens
        )
        rows.append(
            {
                "ordinal": ordinal,
                "record_id": packet["record_id"],
                "packet_sha256": entry["sha256"],
                "input_sha256": packet["input_sha256"],
                "prompt_tokens": int(prompt_ids.shape[1]),
                "seed_token_id": seed_token_id,
                "reference": reference,
                "proposal": proposal,
                "accepted": accepted_prefix(proposal, reference, stop_ids),
                "draft_tokens": args.draft_tokens,
                "prefill_ms": prefill_ms,
                "draft_decode_ms": decode_ms,
                "total_draft_ms": prefill_ms + decode_ms,
            }
        )
        del packet, prompt_ids
        print(
            json.dumps(
                {
                    "event": "small_draft_screen",
                    "completed": ordinal + 1,
                    "total": len(entries),
                    "accepted": rows[-1]["accepted"],
                }
            ),
            flush=True,
        )

    write_result(args, packet_root, rows, runtime="transformers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=512)
    parser.add_argument("--warmup-seed-token", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--runtime", choices=("transformers", "vllm"), default="transformers"
    )
    parser.add_argument(
        "--proposal-origin",
        choices=("after_target_seed", "concurrent_seed_branch"),
        default="after_target_seed",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    args = parser.parse_args()
    if args.request_offset < 0:
        parser.error("request offset cannot be negative")
    if args.num_requests is not None and args.num_requests <= 0:
        parser.error("num requests must be positive")
    if min(args.draft_tokens, args.warmup_tokens) <= 0:
        parser.error("draft and warmup token counts must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("gpu memory utilization must lie in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
