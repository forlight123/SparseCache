"""Evaluate a block-wise sparse Target-KV cross-attention drafter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from experiments.lossless_pd.reference_packets import (
    digest_file,
    load_packet,
    read_index,
)
from experiments.lossless_pd.screen_small_model_draft import accepted_prefix
from experiments.small_kv_adapter.cross_attention import (
    SparseTargetKVCrossAttention,
)
from experiments.small_kv_adapter.evaluate import sparse_view, summarize
from experiments.small_kv_adapter.train import validate_model_contract


@torch.inference_mode()
def propose(
    small: torch.nn.Module,
    adapter: SparseTargetKVCrossAttention | None,
    packet: dict[str, Any],
    *,
    draft_tokens: int,
    fraction: float,
    zero_kv: bool = False,
) -> tuple[list[int], dict[str, float]]:
    prompt_ids = packet["prompt_ids"].to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    prompt = small.model(prompt_ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - started) * 1000

    adapter_ms = 0.0
    memory = None
    if adapter is not None:
        keys, values, positions = sparse_view(packet, fraction)
        if zero_kv:
            keys = torch.zeros_like(keys)
            values = torch.zeros_like(values)
        torch.cuda.synchronize()
        started = time.perf_counter()
        memory = adapter.prepare_memory(keys, values, positions)
        torch.cuda.synchronize()
        adapter_ms = (time.perf_counter() - started) * 1000

    token = packet["seed"].reshape(1, 1).to("cuda", non_blocking=True)
    proposal = []
    cache = prompt.past_key_values
    prompt_tokens = int(packet["prompt_tokens"])
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(draft_tokens):
        if adapter is None:
            output = small(
                token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        else:
            query_positions = torch.tensor(
                [prompt_tokens + step], device="cuda", dtype=torch.long
            )
            with adapter.activate(small, memory, query_positions):
                output = small(
                    token,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        proposal.append(int(token.item()))
        cache = output.past_key_values
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - started) * 1000
    return proposal, {
        "prefill_ms": prefill_ms,
        "adapter_ms": adapter_ms,
        "decode_ms": decode_ms,
        "total_ms": prefill_ms + adapter_ms + decode_ms,
    }


def evaluate(args: argparse.Namespace) -> None:
    root = args.packets.resolve()
    index = read_index(root)
    entries = index["entries"][args.request_offset :]
    if args.num_requests is not None:
        entries = entries[: args.num_requests]
    if not entries:
        raise ValueError("evaluation packet slice is empty")
    if args.draft_tokens > int(index["draft_tokens"]):
        raise ValueError("draft horizon exceeds immutable references")

    small = (
        AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
        .requires_grad_(False)
    )
    adapter = (
        SparseTargetKVCrossAttention.load_checkpoint(args.checkpoint)
        .to("cuda", dtype=torch.bfloat16)
        .eval()
    )
    validate_model_contract(small, adapter.config)
    modes = ("base", "exact", "zero") if args.include_ablations else ("base", "exact")

    warm = load_packet(root, entries[0])
    propose(
        small,
        adapter,
        warm,
        draft_tokens=args.draft_tokens,
        fraction=args.fraction,
    )
    rows = []
    for ordinal, entry in enumerate(entries):
        packet = load_packet(root, entry)
        reference = packet["reference"][: args.draft_tokens].tolist()
        stop_ids = set(packet["stop_ids"].tolist())
        for mode in modes:
            proposal, timing = propose(
                small,
                None if mode == "base" else adapter,
                packet,
                draft_tokens=args.draft_tokens,
                fraction=args.fraction,
                zero_kv=mode == "zero",
            )
            rows.append(
                {
                    "ordinal": ordinal,
                    "record_id": packet["record_id"],
                    "packet_sha256": entry["sha256"],
                    "input_sha256": packet["input_sha256"],
                    "mode": mode,
                    "reference": reference,
                    "proposal": proposal,
                    "accepted": accepted_prefix(proposal, reference, stop_ids),
                    **timing,
                }
            )
        print(
            json.dumps(
                {
                    "event": "cross_attention_eval",
                    "completed": ordinal + 1,
                    "total": len(entries),
                }
            ),
            flush=True,
        )

    result = {
        "schema_version": 1,
        "architecture": "block_wise_sparse_target_kv_cross_attention",
        "metric_contract": (
            "exact greedy accepted prefix after the authoritative P seed; "
            "immutable full-Qwen3-8B Target references"
        ),
        "draft_model": str(args.draft_model.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest_file(
            args.checkpoint / "cross_attention_adapter.pt"
        ),
        "packet_root": str(root),
        "packet_index_sha256": digest_file(root / "index.json"),
        "draft_tokens": args.draft_tokens,
        "fraction": args.fraction,
        "cells": {mode: summarize(rows, mode) for mode in modes},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["cells"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--include-ablations", action="store_true")
    args = parser.parse_args()
    if args.request_offset < 0:
        parser.error("request offset cannot be negative")
    if args.num_requests is not None and args.num_requests <= 0:
        parser.error("num requests must be positive")
    if args.draft_tokens <= 0:
        parser.error("draft tokens must be positive")
    if not 0.0 < args.fraction <= 1.0:
        parser.error("fraction must lie in (0, 1]")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
