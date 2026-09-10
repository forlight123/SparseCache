"""Benchmark fixed-split batched qlen=1 attention against serial decode.

Run this with the validated LMCache/vLLM environment.  GPU selection is made
with ``CUDA_VISIBLE_DEVICES`` so the script always uses logical ``cuda:0``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch


def parse_positive_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw.split(",") if item.strip())
    if not values or min(values) <= 0:
        raise ValueError("expected a comma-separated list of positive integers")
    return values


def page_metadata(
    lengths: tuple[int, ...], page_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indptr = [0]
    indices: list[int] = []
    last_page_len = []
    for length in lengths:
        page_count = (length + page_size - 1) // page_size
        indices.extend(range(page_count))
        indptr.append(len(indices))
        last_page_len.append((length - 1) % page_size + 1)
    return tuple(
        torch.tensor(value, dtype=torch.int32, device=device)
        for value in (indptr, indices, last_page_len)
    )


def median_cuda_ms(function: Callable[[], Any], iterations: int) -> float:
    for _ in range(4):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


def run_serial_attention(wrappers, query, key, value):
    return [
        wrapper.run(query[row : row + 1], (key, value))
        for row, wrapper in enumerate(wrappers)
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed_values = sorted({int(row["fixed_split_pages"]) for row in rows})
    result = {}
    for fixed in fixed_values:
        selected = [row for row in rows if row["fixed_split_pages"] == fixed]
        result[str(fixed)] = {
            "exact_cells": sum(bool(row["bitwise_equal"]) for row in selected),
            "cells": len(selected),
            "min_speedup_vs_serial_q1": min(
                float(row["speedup_vs_serial_q1"]) for row in selected
            ),
            "geomean_speedup_vs_serial_q1": math.exp(
                statistics.fmean(
                    math.log(float(row["speedup_vs_serial_q1"]))
                    for row in selected
                )
            ),
        }
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix-lengths", default="509,2557,8189,32765,65529"
    )
    parser.add_argument("--draft-lengths", default="4,8,16")
    parser.add_argument("--fixed-split-pages", default="32,64,128,256")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    args = parser.parse_args()
    if min(
        args.page_size,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        args.seeds,
    ) <= 0:
        parser.error("shape and seed arguments must be positive")
    if args.num_q_heads % args.num_kv_heads:
        parser.error("--num-q-heads must be divisible by --num-kv-heads")
    if args.workspace_gib <= 0:
        parser.error("--workspace-gib must be positive")

    try:
        import flashinfer
    except ImportError:
        parser.error(
            "flashinfer is required; use the LMCache/.venv-vllm environment"
        )

    prefix_lengths = parse_positive_ints(args.prefix_lengths)
    draft_lengths = parse_positive_ints(args.draft_lengths)
    fixed_values = parse_positive_ints(args.fixed_split_pages)
    device = torch.device("cuda")
    workspace = torch.empty(
        int(args.workspace_gib * 1024**3), dtype=torch.uint8, device=device
    )

    def make_wrapper(lengths: tuple[int, ...], fixed: int):
        indptr, indices, last_page_len = page_metadata(
            lengths, args.page_size, device
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_tensor_cores=True,
            backend="auto",
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_dim,
            args.page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            fixed_split_size=fixed,
        )
        return wrapper

    rows = []
    for prefix_length in prefix_lengths:
        for draft_length in draft_lengths:
            lengths = tuple(
                prefix_length + offset
                for offset in range(1, draft_length + 1)
            )
            page_count = (lengths[-1] + args.page_size - 1) // args.page_size
            torch.manual_seed(100_000 + prefix_length * 10 + draft_length)
            key = torch.randn(
                page_count,
                args.page_size,
                args.num_kv_heads,
                args.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            value = torch.randn_like(key)
            for fixed in fixed_values:
                batched = make_wrapper(lengths, fixed)
                serial = tuple(make_wrapper((length,), fixed) for length in lengths)
                bitwise_equal = True
                unequal_elements = 0
                max_abs_error = 0.0
                for seed in range(args.seeds):
                    torch.manual_seed(seed + prefix_length + draft_length)
                    query = torch.randn(
                        draft_length,
                        args.num_q_heads,
                        args.head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    batched_output = batched.run(query, (key, value))
                    serial_output = torch.cat(
                        [
                            wrapper.run(query[row : row + 1], (key, value))
                            for row, wrapper in enumerate(serial)
                        ]
                    )
                    torch.cuda.synchronize()
                    unequal = int((batched_output != serial_output).sum())
                    unequal_elements += unequal
                    bitwise_equal &= unequal == 0
                    max_abs_error = max(
                        max_abs_error,
                        float(
                            (batched_output.float() - serial_output.float())
                            .abs()
                            .max()
                        ),
                    )

                query = torch.randn(
                    draft_length,
                    args.num_q_heads,
                    args.head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                iterations = 30 if prefix_length <= 8192 else 10
                batched_ms = median_cuda_ms(
                    partial(batched.run, query, (key, value)), iterations
                )
                serial_ms = median_cuda_ms(
                    partial(run_serial_attention, serial, query, key, value),
                    iterations,
                )
                row = {
                    "prefix_length": prefix_length,
                    "draft_length": draft_length,
                    "fixed_split_pages": fixed,
                    "fixed_split_tokens": fixed * args.page_size,
                    "seeds": args.seeds,
                    "bitwise_equal": bitwise_equal,
                    "unequal_elements": unequal_elements,
                    "max_abs_error": max_abs_error,
                    "batched_ms": batched_ms,
                    "serial_q1_ms": serial_ms,
                    "speedup_vs_serial_q1": serial_ms / batched_ms,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

    result = {
        "contract": (
            "one fixed-split FlashInfer batch with one qlen=1 sequence per "
            "draft position versus the same rows issued as separate qlen=1 calls"
        ),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "dtype": "bfloat16",
            "page_size": args.page_size,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
