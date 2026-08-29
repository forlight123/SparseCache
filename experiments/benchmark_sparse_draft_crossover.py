# SPDX-License-Identifier: Apache-2.0
"""Measure when exact sparse-page self-drafting becomes genuinely cheaper.

This is a compute-only mechanism experiment.  It deliberately excludes P/D
wire time and final verification so that the draft path itself cannot hide
behind transfer overlap.  A single PROGRESSIVE_KV engine executes a balanced
within-process sweep including a 100%-visible control; an ordinary FLASH_ATTN
engine supplies a secondary backend-overhead control.  Prefix-cache coverage
is a fail-closed gate, so timed requests contain decode work rather than a
repeated long prefill.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any


def parse_fractions(raw: str) -> tuple[float, ...]:
    """Parse unique visibility fractions and require the dense 100% control."""
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("visibility fractions must be numeric") from error
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("visibility fractions must lie in (0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("visibility fractions must be unique")
    if 1.0 not in values:
        raise ValueError("visibility fractions must include the 1.0 control")
    return tuple(sorted(values))


def balanced_fraction_schedule(
    fractions: tuple[float, ...], repeats: int
) -> list[tuple[int, int, float]]:
    """Build a deterministic rotation/reversal schedule for thermal fairness."""
    if repeats <= 0:
        raise ValueError("measurement repeats must be positive")
    schedule = []
    count = len(fractions)
    for repeat in range(repeats):
        ordered = list(fractions)
        if repeat % 2:
            ordered.reverse()
        shift = (repeat // 2) % count
        ordered = ordered[shift:] + ordered[:shift]
        schedule.extend(
            (repeat, order_index, fraction)
            for order_index, fraction in enumerate(ordered)
        )
    return schedule


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def bootstrap_mean_ci(
    values: list[float], *, samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a paired mean."""
    if not values:
        raise ValueError("bootstrap values must be non-empty")
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(
            statistics.fmean(generator.choice(values) for _ in range(len(values)))
        )
    means.sort()
    lower = means[math.floor(0.025 * (samples - 1))]
    upper = means[math.ceil(0.975 * (samples - 1))]
    return lower, upper


def describe(values: list[float]) -> dict[str, float]:
    """Summarize one non-empty latency vector."""
    if not values:
        raise ValueError("cannot describe an empty vector")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _request_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == offset:
                return json.loads(line)
    raise ValueError(f"request pack has no row at offset {offset}")


def _model_configuration(model: Path) -> tuple[int, int, int]:
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    max_position = int(config["max_position_embeddings"])
    vocab_size = int(config["vocab_size"])
    if max_position <= 0:
        raise ValueError("model max_position_embeddings must be positive")
    if vocab_size <= 1024:
        raise ValueError("model vocab_size is unexpectedly small")
    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    head_dim = int(
        config.get(
            "head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"])
        )
    )
    logical_bf16_kv_bytes_per_token = (
        2 * num_layers * num_kv_heads * head_dim * 2
    )
    return max_position, vocab_size, logical_bf16_kv_bytes_per_token


def measurement_seed_token(vocab_size: int, sequence_index: int) -> int:
    """Choose a valid changing token so only the exact prompt prefix is cached."""
    return 1024 + sequence_index % (vocab_size - 1024)


def prepare_prompt(
    *,
    requests_jsonl: Path,
    request_offset: int,
    model: Path,
    draft_tokens: int,
    block_size: int,
    max_context_tokens: int | None,
) -> dict[str, Any]:
    """Materialize one block-aligned prompt without exceeding model context."""
    request = _request_at(requests_jsonl, request_offset)
    prompt = request.get("prompt")
    if not isinstance(prompt, list) or not prompt or any(
        not isinstance(token, int) for token in prompt
    ):
        raise ValueError("request row must contain a non-empty integer prompt list")
    model_limit, vocab_size, kv_bytes_per_token = _model_configuration(model)
    safe_limit = model_limit - draft_tokens - 2
    if max_context_tokens is not None:
        safe_limit = min(safe_limit, max_context_tokens)
    safe_limit -= safe_limit % block_size
    if safe_limit < 2 * block_size:
        raise ValueError("model/output budget leaves fewer than two KV pages")
    effective = min(len(prompt), safe_limit)
    effective -= effective % block_size
    prompt = prompt[:effective]
    return {
        "prompt_token_ids": prompt,
        "source_prompt_tokens": len(request["prompt"]),
        "effective_prompt_tokens": len(prompt),
        "model_max_position_embeddings": model_limit,
        "vocab_size": vocab_size,
        "logical_bf16_kv_bytes_per_token": kv_bytes_per_token,
        "doc_start_token": 0,
        "doc_end_token": len(prompt),
        "source_request_offset": request_offset,
        "source_request_sha256": hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def publish_visibility(
    path: Path, *, request_id: str, fraction: float, trace_enabled: bool = False
) -> None:
    """Atomically publish one fixed-S1 visibility snapshot for a timed request."""
    payload = {
        "request_id": request_id,
        "visibility_mode": "fixed_s1",
        "completed_fraction": fraction,
        "epoch": request_id,
        "completed_at_ns": 0,
        "visible_token_ranges": [],
        "trace_enabled": trace_enabled,
        "trace_full_visibility": trace_enabled and fraction == 1.0,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def _assert_exclusive_gpu() -> tuple[int, int]:
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    used_bytes = total_bytes - free_bytes
    if used_bytes > 1 * 2**30:
        raise RuntimeError(
            f"selected GPU is not exclusive: {used_bytes / 2**30:.2f} GiB used"
        )
    return used_bytes, total_bytes


def run_worker(args: argparse.Namespace) -> None:
    """Run either the within-engine progressive sweep or FlashAttention control."""
    import torch
    from vllm import LLM, SamplingParams

    prompt = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
    prompt_ids = list(prompt["prompt_token_ids"])
    fractions = parse_fractions(args.visibility_fractions)
    schedule = balanced_fraction_schedule(fractions, args.measurement_repeats)
    initial_used = 0
    initial_total = 0
    if args.require_exclusive_gpu:
        initial_used, initial_total = _assert_exclusive_gpu()

    progressive = args.worker_mode == "progressive"
    backend = "PROGRESSIVE_KV" if progressive else "FLASH_ATTN"
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=len(prompt_ids) + args.draft_tokens + 2,
        max_num_seqs=args.batch_size,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=True,
        disable_log_stats=False,
        attention_config={"backend": backend},
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.draft_tokens,
        ignore_eos=True,
    )
    warmup_requests = [
        {"prompt_token_ids": prompt_ids} for _ in range(args.batch_size)
    ]
    completion = Path(args.completion_file) if args.completion_file else None
    if progressive and completion is None:
        raise ValueError("progressive worker requires a completion file")

    for warmup in range(args.warmup_repeats):
        if progressive:
            assert completion is not None
            publish_visibility(
                completion,
                request_id=f"warmup-{warmup}",
                fraction=1.0,
                trace_enabled=False,
            )
        engine.generate(warmup_requests, sampling, use_tqdm=False)

    rows = []
    worker_schedule = schedule if progressive else [
        (repeat, 0, 1.0) for repeat in range(args.measurement_repeats)
    ]
    for repeat, order_index, fraction in worker_schedule:
        request_id = f"{args.worker_mode}-r{repeat:03d}-o{order_index:02d}-f{fraction:.6f}"
        sequence_index = repeat * len(fractions) + order_index
        seed_token_ids = [
            measurement_seed_token(
                int(prompt["vocab_size"]), sequence_index * args.batch_size + index
            )
            for index in range(args.batch_size)
        ]
        # The block-aligned prompt was populated by warmup.  A changing final
        # token prevents the prefix cache from reusing the query itself, so the
        # timed request performs exactly one sparse/full seed forward followed
        # by the requested autoregressive draft horizon.
        requests = [
            {"prompt_token_ids": prompt_ids + [seed_token_id]}
            for seed_token_id in seed_token_ids
        ]
        if progressive:
            assert completion is not None
            publish_visibility(
                completion,
                request_id=request_id,
                fraction=fraction,
                trace_enabled=False,
            )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = perf_counter()
        if args.emit_nvtx:
            torch.cuda.nvtx.range_push(
                "sparsecache_draft|"
                f"backend={backend}|fraction={fraction:.6f}|"
                f"batch={args.batch_size}|repeat={repeat:03d}"
            )
        try:
            outputs = engine.generate(requests, sampling, use_tqdm=False)
        finally:
            if args.emit_nvtx:
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000
        rows.append(
            {
                "request_id": request_id,
                "worker_mode": args.worker_mode,
                "backend": backend,
                "repeat": repeat,
                "order_index": order_index,
                "visible_fraction": fraction,
                "elapsed_ms": elapsed_ms,
                "batch_size": args.batch_size,
                "prompt_tokens": len(prompt_ids),
                "draft_tokens": args.draft_tokens,
                "seed_token_ids": seed_token_ids,
                "num_cached_tokens": min(
                    int(output.num_cached_tokens or 0) for output in outputs
                ),
                "batch_num_cached_tokens": [
                    int(output.num_cached_tokens or 0) for output in outputs
                ],
                "token_ids": list(outputs[0].outputs[0].token_ids),
                "batch_token_ids": [
                    list(output.outputs[0].token_ids) for output in outputs
                ],
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "initial_gpu_used_bytes": initial_used,
                "initial_gpu_total_bytes": initial_total,
            }
        )
    if progressive:
        # Mechanism telemetry performs Python hashing and file I/O. Keep it
        # entirely outside the publishable latency interval, then audit one
        # prefix-cache-hit request per fraction in the already-loaded engine.
        assert completion is not None
        audit_base = args.measurement_repeats * len(fractions)
        for audit_index, fraction in enumerate(fractions):
            request_id = f"mechanism-audit-f{fraction:.6f}"
            seed_token_ids = [
                measurement_seed_token(
                    int(prompt["vocab_size"]),
                    (audit_base + audit_index) * args.batch_size + index,
                )
                for index in range(args.batch_size)
            ]
            requests = [
                {"prompt_token_ids": prompt_ids + [seed_token_id]}
                for seed_token_id in seed_token_ids
            ]
            publish_visibility(
                completion,
                request_id=request_id,
                fraction=fraction,
                trace_enabled=True,
            )
            engine.generate(requests, sampling, use_tqdm=False)
    write_jsonl(args.result_file, rows)


def _attention_by_request(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        request_id = row.get("request_id")
        if isinstance(request_id, str) and not request_id.startswith("warmup-"):
            grouped.setdefault(request_id, []).append(row)
    return grouped


def aggregate(
    *,
    prompt: dict[str, Any],
    progressive_rows: list[dict[str, Any]],
    flash_rows: list[dict[str, Any]],
    attention_path: Path,
    fractions: tuple[float, ...],
    repeats: int,
    block_size: int,
) -> dict[str, Any]:
    """Validate and summarize one crossover cell."""
    errors: list[str] = []
    expected_keys = {
        (repeat, fraction) for repeat in range(repeats) for fraction in fractions
    }
    actual_keys = {
        (int(row["repeat"]), float(row["visible_fraction"]))
        for row in progressive_rows
    }
    if actual_keys != expected_keys or len(progressive_rows) != len(expected_keys):
        errors.append("progressive rows do not form a complete repeat/fraction matrix")
    if len(flash_rows) != repeats or {
        int(row["repeat"]) for row in flash_rows
    } != set(range(repeats)):
        errors.append("FlashAttention control does not cover every repeat")

    minimum_cached = int(prompt["effective_prompt_tokens"])
    all_rows = progressive_rows + flash_rows
    batch_sizes = {int(row.get("batch_size", 1)) for row in all_rows}
    if len(batch_sizes) != 1:
        errors.append("timed rows do not use one uniform batch size")
    batch_size = next(iter(batch_sizes), 1)
    if any(int(row.get("num_cached_tokens", -1)) < minimum_cached for row in all_rows):
        errors.append("a timed request recomputed more than one prompt KV page")
    if any(
        len(row.get("batch_token_ids", [row.get("token_ids", [])])) != batch_size
        or any(
            len(token_ids) != int(row["draft_tokens"])
            for token_ids in row.get(
                "batch_token_ids", [row.get("token_ids", [])]
            )
        )
        for row in all_rows
    ):
        errors.append("a timed request did not produce the fixed draft horizon")

    attention = _attention_by_request(attention_path)
    trace_steps = {}
    logical_payload_by_fraction: dict[float, dict[str, int]] = {}
    expected_draft_tokens = (
        int(progressive_rows[0]["draft_tokens"]) if progressive_rows else 0
    )
    timed_request_ids = {str(row["request_id"]) for row in progressive_rows}
    traced_timed_requests = timed_request_ids.intersection(attention)
    if traced_timed_requests:
        errors.append("timed requests contain mechanism trace I/O")
    for fraction in fractions:
        request_id = f"mechanism-audit-f{fraction:.6f}"
        records = attention.get(request_id, [])
        steps = {int(record["decode_step"]): record for record in records}
        if len(steps) != expected_draft_tokens:
            errors.append(f"{request_id}: attention trace misses draft positions")
            continue
        for record in records:
            if record.get("trace_scope") != "representative_layer_0":
                errors.append(f"{request_id}: non-representative trace scope")
            if not isinstance(record.get("visible_logical_blocks_sha256"), str):
                errors.append(f"{request_id}: missing exact page-set digest")
            visible = int(record["visible_pages"])
            candidates = int(record["candidate_pages"])
            if not 0 < visible <= candidates:
                errors.append(f"{request_id}: invalid physical page counts")
            if fraction == 1.0 and visible != candidates:
                errors.append(f"{request_id}: 100% control is not physically dense")
            if fraction < 1.0 and visible >= candidates:
                errors.append(f"{request_id}: sparse control did not remove a page")
            if fraction < 1.0 and int(record["sparse_seq_len"]) >= int(
                record["full_seq_len"]
            ):
                errors.append(f"{request_id}: sparse sequence length was not reduced")
        trace_steps[request_id] = len(steps)
        bytes_per_token = int(prompt["logical_bf16_kv_bytes_per_token"])
        logical_payload_by_fraction[fraction] = {
            "addressed_bytes": sum(
                int(record["sparse_seq_len"]) * bytes_per_token
                for record in records
            )
            * batch_size,
            "dense_equivalent_bytes": sum(
                int(record["full_seq_len"]) * bytes_per_token for record in records
            )
            * batch_size,
        }

    by_fraction: dict[float, list[dict[str, Any]]] = {value: [] for value in fractions}
    for row in progressive_rows:
        by_fraction[float(row["visible_fraction"])].append(row)
    for rows in by_fraction.values():
        rows.sort(key=lambda item: int(item["repeat"]))
    dense_by_repeat = {
        int(row["repeat"]): float(row["elapsed_ms"]) for row in by_fraction[1.0]
    }
    fraction_summaries = {}
    crossover_observed = False
    for fraction in fractions:
        rows = by_fraction[fraction]
        latencies = [float(row["elapsed_ms"]) for row in rows]
        deltas = [
            dense_by_repeat[int(row["repeat"])] - float(row["elapsed_ms"])
            for row in rows
        ]
        lower, upper = bootstrap_mean_ci(deltas)
        dense_mean = statistics.fmean(dense_by_repeat.values())
        sparse_mean = statistics.fmean(latencies)
        speedup = dense_mean / sparse_mean
        payload = logical_payload_by_fraction.get(fraction)
        if fraction <= 0.1 and speedup >= 1.1 and lower > 0:
            crossover_observed = True
        logical_payload = None
        if payload is not None:
            logical_payload = {
                "mean_addressed_gib": payload["addressed_bytes"] / 2**30,
                "mean_dense_equivalent_gib": payload["dense_equivalent_bytes"]
                / 2**30,
                "addressed_fraction": payload["addressed_bytes"]
                / payload["dense_equivalent_bytes"],
                "interpretation": (
                    "exact BF16 K/V elements addressed assuming one read; "
                    "not measured DRAM transactions"
                ),
            }
        fraction_summaries[f"{fraction:.6f}"] = {
            "visible_fraction": fraction,
            "latency_ms": describe(latencies),
            "latency_per_request_ms": describe(
                [latency / batch_size for latency in latencies]
            ),
            "aggregate_draft_tokens_per_second": describe(
                [
                    batch_size * int(row["draft_tokens"]) * 1000.0
                    / float(row["elapsed_ms"])
                    for row in rows
                ]
            ),
            "paired_saved_vs_100pct_ms": {
                **describe(deltas),
                "bootstrap_95pct_ci": [lower, upper],
            },
            "speedup_vs_100pct_ratio_of_means": speedup,
            "mean_cached_tokens": statistics.fmean(
                int(row["num_cached_tokens"]) for row in rows
            ),
            "logical_kv_payload": logical_payload,
        }

    flash_latencies = [float(row["elapsed_ms"]) for row in flash_rows]
    progressive_dense_mean = fraction_summaries["1.000000"]["latency_ms"]["mean"]
    flash_mean = statistics.fmean(flash_latencies)
    backend_overhead_ratio = progressive_dense_mean / flash_mean
    gates = {
        "complete_balanced_matrix": not any("matrix" in error for error in errors),
        "uniform_batch_geometry": not any(
            "uniform batch size" in error for error in errors
        ),
        "flash_control_coverage": not any(
            "FlashAttention control" in error for error in errors
        ),
        "prefix_cache_coverage": not any("recomputed" in error for error in errors),
        "fixed_draft_horizon": not any("fixed draft horizon" in error for error in errors),
        "request_attributed_attention": not any(
            "attention trace" in error for error in errors
        ),
        "physical_sparse_pages": not any(
            "physically dense" in error
            or "did not remove" in error
            or "not reduced" in error
            for error in errors
        ),
        "compact_trace_scope": not any(
            "trace scope" in error or "page-set digest" in error for error in errors
        ),
        "timing_trace_isolation": not any(
            "trace I/O" in error for error in errors
        ),
    }
    valid = not errors and all(gates.values())
    return {
        "schema_version": 1,
        "status": "valid" if valid else "invalid",
        "purpose": (
            "compute-only exact sparse-page draft crossover with homogeneous "
            "batch geometry; no P/D wire or verifier"
        ),
        "prompt": {key: value for key, value in prompt.items() if key != "prompt_token_ids"},
        "batch_size": batch_size,
        "repeats": repeats,
        "fractions": list(fractions),
        "gates": gates,
        "errors": errors,
        "decision_metrics": {
            "economic_crossover_observed_at_or_below_10pct": crossover_observed,
            "progressive_100pct_over_flash_ratio": backend_overhead_ratio,
            "progressive_backend_overhead_within_10pct": backend_overhead_ratio <= 1.1,
        },
        "progressive": fraction_summaries,
        "flash_dense_control": {
            "latency_ms": describe(flash_latencies),
            "latency_per_request_ms": describe(
                [latency / batch_size for latency in flash_latencies]
            ),
            "aggregate_draft_tokens_per_second": describe(
                [
                    batch_size * int(row["draft_tokens"]) * 1000.0
                    / float(row["elapsed_ms"])
                    for row in flash_rows
                ]
            ),
            "mean_cached_tokens": statistics.fmean(
                int(row["num_cached_tokens"]) for row in flash_rows
            ),
        },
        "trace_steps_per_request": trace_steps,
    }


def write_summary_artifacts(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Exact sparse-draft crossover",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This cell is compute-only; it excludes P/D wire time and final verification.",
        f"Homogeneous decode batch size: **{summary['batch_size']}**.",
        "",
        "| visible KV | mean draft ms | speedup vs 100% | paired saved ms | 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["progressive"].values():
        paired = item["paired_saved_vs_100pct_ms"]
        interval = paired["bootstrap_95pct_ci"]
        lines.append(
            f"| {item['visible_fraction']:.1%} | {item['latency_ms']['mean']:.3f} "
            f"| {item['speedup_vs_100pct_ratio_of_means']:.3f}x "
            f"| {paired['mean']:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] |"
        )
    if summary["errors"]:
        lines.extend(["", "## Invalidity reasons", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output_dir / "paper_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "visible_fraction",
                "mean_draft_ms",
                "speedup_vs_100pct",
                "paired_saved_ms",
                "ci95_low_ms",
                "ci95_high_ms",
            ),
        )
        writer.writeheader()
        for item in summary["progressive"].values():
            paired = item["paired_saved_vs_100pct_ms"]
            writer.writerow(
                {
                    "visible_fraction": item["visible_fraction"],
                    "mean_draft_ms": item["latency_ms"]["mean"],
                    "speedup_vs_100pct": item[
                        "speedup_vs_100pct_ratio_of_means"
                    ],
                    "paired_saved_ms": paired["mean"],
                    "ci95_low_ms": paired["bootstrap_95pct_ci"][0],
                    "ci95_high_ms": paired["bootstrap_95pct_ci"][1],
                }
            )


def launch_worker(
    args: argparse.Namespace,
    *,
    mode: str,
    prompt_file: Path,
    result_file: Path,
    stats_file: Path,
    completion_file: Path,
) -> None:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "vllm"
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "PYTHONPATH": (
                str(source_root)
                if not old_pythonpath
                else str(source_root) + os.pathsep + old_pythonpath
            ),
        }
    )
    prompt = json.loads(prompt_file.read_text(encoding="utf-8"))
    if mode == "progressive":
        environment.update(
            {
                "VLLM_PROGRESSIVE_KV_DOC_START_TOKEN": str(
                    prompt["doc_start_token"]
                ),
                "VLLM_PROGRESSIVE_KV_DOC_END_TOKEN": str(prompt["doc_end_token"]),
                "VLLM_PROGRESSIVE_KV_VISIBLE_FRACTIONS": "1.0",
                "VLLM_PROGRESSIVE_KV_PAGE_ORDER": args.page_order,
                "VLLM_PROGRESSIVE_KV_STATS_PATH": str(stats_file),
                "VLLM_PROGRESSIVE_KV_COMPLETION_PATH": str(completion_file),
                "VLLM_PROGRESSIVE_KV_VISIBILITY_MODE": "fixed_s1",
            }
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
        "--prompt-file",
        str(prompt_file),
        "--result-file",
        str(result_file),
        "--stats-file",
        str(stats_file),
        "--completion-file",
        str(completion_file),
        "--visibility-fractions",
        args.visibility_fractions,
        "--draft-tokens",
        str(args.draft_tokens),
        "--block-size",
        str(args.block_size),
        "--warmup-repeats",
        str(args.warmup_repeats),
        "--measurement-repeats",
        str(args.measurement_repeats),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.require_exclusive_gpu:
        command.append("--require-exclusive-gpu")
    if args.emit_nvtx:
        command.append("--emit-nvtx")
    subprocess.run(command, env=environment, check=True)


def run_parent(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = prepare_prompt(
        requests_jsonl=Path(args.requests_jsonl),
        request_offset=args.request_offset,
        model=Path(args.model),
        draft_tokens=args.draft_tokens,
        block_size=args.block_size,
        max_context_tokens=args.max_context_tokens,
    )
    prompt_file = output_dir / "prompt.json"
    prompt_file.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
    progressive_file = output_dir / "progressive.jsonl"
    flash_file = output_dir / "flash_dense.jsonl"
    stats_file = output_dir / "attention.jsonl"
    completion_file = output_dir / "completion.json"
    launch_worker(
        args,
        mode="progressive",
        prompt_file=prompt_file,
        result_file=progressive_file,
        stats_file=stats_file,
        completion_file=completion_file,
    )
    launch_worker(
        args,
        mode="flash",
        prompt_file=prompt_file,
        result_file=flash_file,
        stats_file=output_dir / "unused_flash_attention.jsonl",
        completion_file=completion_file,
    )
    summary = aggregate(
        prompt=prompt,
        progressive_rows=read_jsonl(progressive_file),
        flash_rows=read_jsonl(flash_file),
        attention_path=stats_file,
        fractions=parse_fractions(args.visibility_fractions),
        repeats=args.measurement_repeats,
        block_size=args.block_size,
    )
    write_summary_artifacts(output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "valid":
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--requests-jsonl")
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument(
        "--visibility-fractions", default="0.02,0.05,0.1,0.25,0.5,1.0"
    )
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--page-order", default="uniform")
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--measurement-repeats", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument(
        "--emit-nvtx",
        action="store_true",
        help="wrap each timed request in an NVTX range for an external nsys run",
    )
    parser.add_argument(
        "--worker-mode", choices=("progressive", "flash"), help=argparse.SUPPRESS
    )
    parser.add_argument("--prompt-file", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--stats-file", help=argparse.SUPPRESS)
    parser.add_argument("--completion-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        parse_fractions(args.visibility_fractions)
    except ValueError as error:
        parser.error(str(error))
    if min(
        args.draft_tokens,
        args.block_size,
        args.batch_size,
        args.warmup_repeats,
        args.measurement_repeats,
    ) <= 0:
        parser.error("token sizes and repeat counts must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("gpu-memory-utilization must lie in (0, 1]")
    if args.worker_mode is None and not args.requests_jsonl:
        parser.error("parent mode requires --requests-jsonl")
    if args.worker_mode is not None and not all(
        (args.prompt_file, args.result_file, args.stats_file)
    ):
        parser.error("worker mode requires prompt/result/stats paths")
    return args


def main() -> None:
    args = parse_args()
    if args.worker_mode is None:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
