"""Evaluate a sparse Target-KV adapter against its exact small-model base."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from experiments.lossless_pd.core import page_order, visible_positions
from experiments.lossless_pd.pilot import bootstrap_delta
from experiments.lossless_pd.reference_packets import (
    digest_file,
    load_packet,
    read_index,
)
from experiments.lossless_pd.screen_small_model_draft import accepted_prefix
from experiments.small_kv_adapter.model import SparseTargetKVAdapter
from experiments.small_kv_adapter.train import validate_model_contract


def sparse_view(packet: dict[str, Any], fraction: float):
    prompt_tokens = int(packet["prompt_tokens"])
    order = page_order(
        prompt_tokens,
        int(packet["page_size"]),
        "priority",
        0,
        packet["priority_scores"],
    )
    positions = visible_positions(
        prompt_tokens,
        int(packet["page_size"]),
        fraction,
        order,
        torch.device("cpu"),
    )
    return (
        packet["keys"].index_select(-2, positions).to("cuda", non_blocking=True),
        packet["values"].index_select(-2, positions).to("cuda", non_blocking=True),
        positions.to("cuda", non_blocking=True),
    )


@torch.inference_mode()
def propose(
    small: torch.nn.Module,
    adapter: SparseTargetKVAdapter | None,
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
    cache = prompt.past_key_values
    if adapter is not None:
        keys, values, positions = sparse_view(packet, fraction)
        if zero_kv:
            keys = torch.zeros_like(keys)
            values = torch.zeros_like(values)
        torch.cuda.synchronize()
        started = time.perf_counter()
        cache = adapter(cache, keys, values, positions)
        torch.cuda.synchronize()
        adapter_ms = (time.perf_counter() - started) * 1000

    token = packet["seed"].reshape(1, 1).to("cuda", non_blocking=True)
    proposal = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(draft_tokens):
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


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    accepted = [int(row["accepted"]) for row in selected]
    base = {
        row["record_id"]: int(row["accepted"])
        for row in rows
        if row["mode"] == "base"
    }
    deltas = [value - base[row["record_id"]] for value, row in zip(accepted, selected)]
    return {
        "requests": len(selected),
        "mean_accepted": statistics.fmean(accepted),
        "accepted_ci95": bootstrap_delta(accepted),
        "mean_delta_vs_base": statistics.fmean(deltas),
        "delta_vs_base_ci95": bootstrap_delta(deltas),
        "zero_acceptance_rate": sum(value == 0 for value in accepted) / len(selected),
        "mean_prefill_ms": statistics.fmean(row["prefill_ms"] for row in selected),
        "mean_adapter_ms": statistics.fmean(row["adapter_ms"] for row in selected),
        "mean_decode_ms": statistics.fmean(row["decode_ms"] for row in selected),
        "mean_total_ms": statistics.fmean(row["total_ms"] for row in selected),
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
        SparseTargetKVAdapter.load_checkpoint(args.checkpoint)
        .to("cuda", dtype=torch.bfloat16)
        .eval()
    )
    validate_model_contract(small, adapter.config)
    modes = ("base", "exact", "zero") if args.include_ablations else ("base", "exact")

    # Warm both model and adapter shapes outside measured rows.
    warm = load_packet(root, entries[0])
    propose(
        small,
        adapter,
        warm,
        draft_tokens=args.draft_tokens,
        fraction=args.fraction,
    )
    rows: list[dict[str, Any]] = []
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
                    "event": "small_kv_adapter_eval",
                    "completed": ordinal + 1,
                    "total": len(entries),
                }
            ),
            flush=True,
        )

    result = {
        "schema_version": 1,
        "metric_contract": (
            "exact greedy accepted prefix after the authoritative P seed; "
            "immutable full-Qwen3-8B Target references"
        ),
        "draft_model": str(args.draft_model.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest_file(args.checkpoint / "small_kv_adapter.pt"),
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
