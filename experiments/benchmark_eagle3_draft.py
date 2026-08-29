# SPDX-License-Identifier: Apache-2.0
"""Screen a learned EAGLE3 draft before progressive P/D integration.

The benchmark runs an ordinary target and several native EAGLE3 configurations
in isolated subprocesses on one physical GPU. Long-prefill work is removed by
an explicit prefix-cache gate. Speculative acceptance is read directly from
vLLM's in-memory Prometheus counters around each timed request.
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
from time import perf_counter, time_ns
from typing import Any


SPEC_COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
    "accepted_per_position": "vllm:spec_decode_num_accepted_tokens_per_pos",
}


def parse_horizons(raw: str) -> tuple[int, ...]:
    """Parse unique positive speculative horizons in ascending order."""
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("speculative horizons must be integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("speculative horizons must be positive")
    if len(values) != len(set(values)):
        raise ValueError("speculative horizons must be unique")
    return tuple(sorted(values))


def common_prefix_length(left: list[int], right: list[int]) -> int:
    """Return the number of equal tokens before the first mismatch."""
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: list[float]) -> dict[str, float]:
    """Summarize a non-empty numeric vector."""
    if not values:
        raise ValueError("cannot summarize an empty vector")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def bootstrap_mean_ci(
    values: list[float], *, samples: int = 10_000, seed: int = 20260827
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a mean."""
    if not values:
        raise ValueError("bootstrap values must be non-empty")
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    ]
    means.sort()
    return (
        means[math.floor(0.025 * (samples - 1))],
        means[math.ceil(0.975 * (samples - 1))],
    )


def _model_limit(model: Path) -> int:
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    return int(config["max_position_embeddings"])


def draft_weight_file(model: Path) -> Path:
    """Resolve the immutable local weight artifact for either EAGLE3 format."""
    candidates = (model / "model.safetensors", model / "pytorch_model.bin")
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            f"draft model must contain exactly one supported weight file: {model}"
        )
    return existing[0]


def load_prompts(
    *,
    requests_jsonl: Path,
    request_offset: int,
    num_requests: int,
    max_context_tokens: int,
    block_size: int,
    output_tokens: int,
    model_limit: int,
) -> list[dict[str, Any]]:
    """Load one prebuilt, block-aligned request slice without unsafe truncation."""
    source = read_jsonl(requests_jsonl)
    selected = source[request_offset : request_offset + num_requests]
    if len(selected) != num_requests:
        raise ValueError("request pack does not contain the requested slice")
    safe_limit = min(max_context_tokens, model_limit - output_tokens - 2)
    safe_limit -= safe_limit % block_size
    prompts = []
    for local_index, row in enumerate(selected):
        prompt = row.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(token) is not int for token in prompt)
        ):
            raise ValueError("every request must contain integer prompt tokens")
        if len(prompt) > safe_limit:
            raise ValueError(
                "prebuilt prompt exceeds the configured context cap; use a "
                "properly middle-truncated request pack"
            )
        if len(prompt) % block_size:
            raise ValueError("prebuilt prompt is not block aligned")
        prompt = list(prompt)
        prompts.append(
            {
                "request_offset": request_offset + local_index,
                "prompt_token_ids": prompt,
                "prompt_tokens": len(prompt),
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompt, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return prompts


def _assert_exclusive_gpu() -> tuple[int, int]:
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    used_bytes = total_bytes - free_bytes
    if used_bytes > 1 * 2**30:
        raise RuntimeError(
            f"selected GPU is not exclusive: {used_bytes / 2**30:.2f} GiB used"
        )
    return used_bytes, total_bytes


def spec_metric_snapshot(engine: Any) -> dict[str, Any]:
    """Read the four EAGLE acceptance counters from one vLLM engine."""
    result: dict[str, Any] = {
        "drafts": 0,
        "draft_tokens": 0,
        "accepted_tokens": 0,
        "accepted_per_position": [],
    }
    by_name: dict[str, list[Any]] = {}
    for metric in engine.get_metrics():
        by_name.setdefault(str(metric.name), []).append(metric)
    for field in ("drafts", "draft_tokens", "accepted_tokens"):
        metrics = by_name.get(SPEC_COUNTERS[field], [])
        result[field] = sum(int(metric.value) for metric in metrics)
    vectors = by_name.get(SPEC_COUNTERS["accepted_per_position"], [])
    if vectors:
        width = max(len(metric.values) for metric in vectors)
        values = [0] * width
        for metric in vectors:
            for index, value in enumerate(metric.values):
                values[index] += int(value)
        result["accepted_per_position"] = values
    return result


def subtract_spec_metrics(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    """Subtract two monotonic speculative-counter snapshots."""
    result = {
        field: int(after[field]) - int(before[field])
        for field in ("drafts", "draft_tokens", "accepted_tokens")
    }
    before_vector = list(before["accepted_per_position"])
    after_vector = list(after["accepted_per_position"])
    width = max(len(before_vector), len(after_vector))
    before_vector.extend([0] * (width - len(before_vector)))
    after_vector.extend([0] * (width - len(after_vector)))
    result["accepted_per_position"] = [
        after_vector[index] - before_vector[index] for index in range(width)
    ]
    if any(value < 0 for value in result.values() if isinstance(value, int)) or any(
        value < 0 for value in result["accepted_per_position"]
    ):
        raise RuntimeError("speculative counters are not monotonic")
    return result


def run_worker(args: argparse.Namespace) -> None:
    """Run one ordinary or EAGLE3 cell in a fresh process."""
    prompts = load_prompts(
        requests_jsonl=Path(args.requests_jsonl),
        request_offset=args.request_offset,
        num_requests=args.num_requests,
        max_context_tokens=args.max_context_tokens,
        block_size=args.block_size,
        output_tokens=args.output_tokens,
        model_limit=_model_limit(Path(args.model)),
    )
    maximum_prompt_tokens = max(int(prompt["prompt_tokens"]) for prompt in prompts)

    import torch

    from vllm import LLM, SamplingParams

    initial_used = 0
    initial_total = 0
    if args.require_exclusive_gpu:
        initial_used, initial_total = _assert_exclusive_gpu()
    engine_kwargs: dict[str, Any] = {}
    if args.worker_mode == "eagle3":
        engine_kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": args.worker_horizon,
        }
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=maximum_prompt_tokens + args.output_tokens + 2,
        max_num_seqs=1,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        disable_log_stats=False,
        enforce_eager=args.enforce_eager,
        **engine_kwargs,
    )
    loaded_allocated_bytes = int(torch.cuda.memory_allocated())
    loaded_reserved_bytes = int(torch.cuda.memory_reserved())
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=args.ignore_eos,
    )
    warmup_sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    # Exercise the capped EOS-aware output path outside every measured interval.
    calibration_prompt = list(prompts[0]["prompt_token_ids"])
    engine.generate(
        [{"prompt_token_ids": calibration_prompt}], warmup_sampling, use_tqdm=False
    )
    engine.generate(
        [{"prompt_token_ids": calibration_prompt}], sampling, use_tqdm=False
    )
    torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_ids = list(prompt["prompt_token_ids"])
        # Populate this exact prefix immediately before the timed decode.
        engine.generate(
            [{"prompt_token_ids": prompt_ids}], warmup_sampling, use_tqdm=False
        )
        torch.cuda.synchronize()
        before = spec_metric_snapshot(engine)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = perf_counter()
        output = engine.generate(
            [{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False
        )[0]
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000.0
        after = spec_metric_snapshot(engine)
        counters = subtract_spec_metrics(after, before)
        token_ids = list(output.outputs[0].token_ids)
        rows.append(
            {
                "request_offset": prompt["request_offset"],
                "prompt_sha256": prompt["prompt_sha256"],
                "prompt_tokens": prompt["prompt_tokens"],
                "mode": args.worker_mode,
                "speculative_horizon": (
                    args.worker_horizon if args.worker_mode == "eagle3" else 0
                ),
                "output_tokens": args.output_tokens,
                "token_ids": token_ids,
                "elapsed_ms": elapsed_ms,
                "num_cached_tokens": int(output.num_cached_tokens or 0),
                "recomputed_prompt_tokens": int(prompt["prompt_tokens"])
                - int(output.num_cached_tokens or 0),
                "spec_metrics": counters,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "loaded_allocated_bytes": loaded_allocated_bytes,
                "loaded_reserved_bytes": loaded_reserved_bytes,
                "initial_gpu_used_bytes": initial_used,
                "initial_gpu_total_bytes": initial_total,
            }
        )
    write_jsonl(args.result_file, rows)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed on the matrix, then build paired EAGLE summaries."""
    horizons = parse_horizons(args.speculative_horizons)
    output_dir = Path(args.output_dir)
    dense_rows = read_jsonl(output_dir / "dense.jsonl")
    errors: list[str] = []
    expected_offsets = set(
        range(args.request_offset, args.request_offset + args.num_requests)
    )
    dense_offsets = {int(row["request_offset"]) for row in dense_rows}
    if len(dense_rows) != args.num_requests or dense_offsets != expected_offsets:
        errors.append("ordinary target rows do not cover the requested slice")
    dense_by_offset = {int(row["request_offset"]): row for row in dense_rows}
    all_rows = list(dense_rows)
    candidate_summaries: list[dict[str, Any]] = []
    for horizon in horizons:
        rows = read_jsonl(output_dir / f"eagle3_k{horizon}.jsonl")
        all_rows.extend(rows)
        offsets = {int(row["request_offset"]) for row in rows}
        if len(rows) != args.num_requests or offsets != expected_offsets:
            errors.append(f"EAGLE3 k={horizon} rows do not cover the requested slice")
            continue
        rows_by_offset = {int(row["request_offset"]): row for row in rows}
        if any(
            row.get("prompt_sha256")
            != dense_by_offset[int(row["request_offset"])].get("prompt_sha256")
            for row in rows
        ):
            errors.append(f"EAGLE3 k={horizon} prompt digest mismatch")
        exact = [
            list(rows_by_offset[offset]["token_ids"])
            == list(dense_by_offset[offset]["token_ids"])
            for offset in sorted(expected_offsets)
        ]
        prefixes = [
            common_prefix_length(
                list(rows_by_offset[offset]["token_ids"]),
                list(dense_by_offset[offset]["token_ids"]),
            )
            for offset in sorted(expected_offsets)
        ]
        length_matches = [
            len(rows_by_offset[offset]["token_ids"])
            == len(dense_by_offset[offset]["token_ids"])
            for offset in sorted(expected_offsets)
        ]
        candidate_latencies = [
            float(rows_by_offset[offset]["elapsed_ms"])
            for offset in sorted(expected_offsets)
        ]
        dense_latencies = [
            float(dense_by_offset[offset]["elapsed_ms"])
            for offset in sorted(expected_offsets)
        ]
        saved = [
            dense_latency - candidate_latency
            for dense_latency, candidate_latency in zip(
                dense_latencies, candidate_latencies, strict=True
            )
        ]
        ci_low, ci_high = bootstrap_mean_ci(saved)
        drafts = sum(int(row["spec_metrics"]["drafts"]) for row in rows)
        drafted = sum(int(row["spec_metrics"]["draft_tokens"]) for row in rows)
        accepted = sum(int(row["spec_metrics"]["accepted_tokens"]) for row in rows)
        per_position = [0] * horizon
        for row in rows:
            values = list(row["spec_metrics"]["accepted_per_position"])
            if len(values) != horizon:
                errors.append(
                    f"EAGLE3 k={horizon} acceptance vector has width {len(values)}"
                )
                continue
            for index, value in enumerate(values):
                per_position[index] += int(value)
        acceptance_rate = accepted / drafted if drafted else 0.0
        mean_acceptance_length = 1.0 + accepted / drafts if drafts else 1.0
        speedup = statistics.fmean(dense_latencies) / statistics.fmean(
            candidate_latencies
        )
        exact_fraction = sum(exact) / len(exact)
        length_match_fraction = sum(length_matches) / len(length_matches)
        gates = {
            "native_target_verifier_active": drafts > 0 and 0 <= accepted <= drafted,
            "paired_output_work_match": length_match_fraction == 1.0,
            "mean_acceptance_length_at_least_2": mean_acceptance_length >= 2.0,
            "speedup_at_least_1p10x": speedup >= 1.10,
            "paired_ci_strictly_positive": ci_low > 0.0,
        }
        candidate_summaries.append(
            {
                "speculative_horizon": horizon,
                "samples": len(rows),
                "latency_ms": describe(candidate_latencies),
                "output_tokens": describe(
                    [float(len(row["token_ids"])) for row in rows]
                ),
                "tokens_per_second_ratio_of_sums": sum(
                    len(row["token_ids"]) for row in rows
                )
                * 1000.0
                / sum(candidate_latencies),
                "ordinary_target_latency_ms": describe(dense_latencies),
                "speedup_ratio_of_means": speedup,
                "paired_saved_ms": {
                    **describe(saved),
                    "bootstrap_95pct_ci": [ci_low, ci_high],
                    "faster_pairs": sum(value > 0 for value in saved),
                },
                "correctness": {
                    "exact_match_fraction": exact_fraction,
                    "output_length_match_fraction": length_match_fraction,
                    "common_prefix_tokens": describe([float(value) for value in prefixes]),
                    "mismatched_request_offsets": [
                        offset
                        for offset, matches in zip(
                            sorted(expected_offsets), exact, strict=True
                        )
                        if not matches
                    ],
                    "output_length_mismatched_request_offsets": [
                        offset
                        for offset, matches in zip(
                            sorted(expected_offsets), length_matches, strict=True
                        )
                        if not matches
                    ],
                    "interpretation": (
                        "native EAGLE emissions are target-verified; exact "
                        "cross-engine greedy equality is a numerical diagnostic"
                    ),
                },
                "acceptance": {
                    "draft_cycles": drafts,
                    "drafted_tokens": drafted,
                    "accepted_tokens": accepted,
                    "draft_token_acceptance_rate": acceptance_rate,
                    "mean_acceptance_length_including_bonus": mean_acceptance_length,
                    "per_position_acceptance_rate": [
                        value / drafts if drafts else 0.0 for value in per_position
                    ],
                },
                "memory": {
                    "loaded_allocated_gib": statistics.fmean(
                        int(row["loaded_allocated_bytes"]) for row in rows
                    )
                    / 2**30,
                    "loaded_reserved_gib": statistics.fmean(
                        int(row["loaded_reserved_bytes"]) for row in rows
                    )
                    / 2**30,
                    "peak_allocated_gib": max(
                        int(row["peak_allocated_bytes"]) for row in rows
                    )
                    / 2**30,
                },
                "pre_registered_gates": gates,
                "screen_pass": all(gates.values()),
            }
        )

    if any(
        not 0 < len(row.get("token_ids", [])) <= args.output_tokens for row in all_rows
    ):
        errors.append("a request returned an invalid EOS-aware output length")
    if any(
        int(row.get("num_cached_tokens", -1))
        < int(row.get("prompt_tokens", 0)) - 2 * args.block_size
        for row in all_rows
    ):
        errors.append("a timed request recomputed more than two prompt blocks")
    if any(int(row.get("initial_gpu_used_bytes", 0)) > 1 * 2**30 for row in all_rows):
        errors.append("a worker did not start on an exclusive GPU")

    dense_latencies = [float(row["elapsed_ms"]) for row in dense_rows]
    return {
        "schema_version": 1,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "scope": "learned draft selection; not an end-to-end P/D claim",
        "target_model": args.model,
        "draft_model": args.draft_model,
        "requests_jsonl": args.requests_jsonl,
        "request_offset": args.request_offset,
        "num_requests": args.num_requests,
        "prompt_tokens": (
            describe([float(row["prompt_tokens"]) for row in dense_rows])
            if dense_rows
            else None
        ),
        "maximum_output_tokens": args.output_tokens,
        "speculative_horizons": list(horizons),
        "engine": {
            "dtype": "bfloat16",
            "batch_size": 1,
            "prefix_caching": True,
            "enforce_eager": args.enforce_eager,
            "ignore_eos": args.ignore_eos,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "ordinary_target": {
            "latency_ms": describe(dense_latencies) if dense_latencies else None,
            "output_tokens": (
                describe([float(len(row["token_ids"])) for row in dense_rows])
                if dense_rows
                else None
            ),
            "tokens_per_second_ratio_of_sums": (
                sum(len(row["token_ids"]) for row in dense_rows)
                * 1000.0
                / sum(dense_latencies)
                if dense_latencies
                else None
            ),
            "loaded_allocated_gib": (
                statistics.fmean(
                    int(row["loaded_allocated_bytes"]) for row in dense_rows
                )
                / 2**30
                if dense_rows
                else None
            ),
        },
        "candidates": candidate_summaries,
        "pre_registered_rule": {
            "native_target_verifier_required": True,
            "minimum_output_length_match_fraction": 1.0,
            "minimum_mean_acceptance_length_including_bonus": 2.0,
            "minimum_speedup": 1.10,
            "paired_saved_ms_ci_must_exclude_zero": True,
            "passing_horizons": [
                cell["speculative_horizon"]
                for cell in candidate_summaries
                if cell["screen_pass"]
            ],
        },
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "paper_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "speculative_horizon",
            "latency_ms_mean",
            "speedup",
            "saved_ms_mean",
            "saved_ms_ci_low",
            "saved_ms_ci_high",
            "acceptance_rate",
            "mean_acceptance_length",
            "exact_match_fraction",
            "screen_pass",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cell in summary["candidates"]:
            writer.writerow(
                {
                    "speculative_horizon": cell["speculative_horizon"],
                    "latency_ms_mean": cell["latency_ms"]["mean"],
                    "speedup": cell["speedup_ratio_of_means"],
                    "saved_ms_mean": cell["paired_saved_ms"]["mean"],
                    "saved_ms_ci_low": cell["paired_saved_ms"][
                        "bootstrap_95pct_ci"
                    ][0],
                    "saved_ms_ci_high": cell["paired_saved_ms"][
                        "bootstrap_95pct_ci"
                    ][1],
                    "acceptance_rate": cell["acceptance"][
                        "draft_token_acceptance_rate"
                    ],
                    "mean_acceptance_length": cell["acceptance"][
                        "mean_acceptance_length_including_bonus"
                    ],
                    "exact_match_fraction": cell["correctness"][
                        "exact_match_fraction"
                    ],
                    "screen_pass": cell["screen_pass"],
                }
            )
    lines = [
        "# EAGLE3 learned-draft screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| k | mean ms | speedup | accepted | mean length | exact | pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for cell in summary["candidates"]:
        lines.append(
            f"| {cell['speculative_horizon']} | "
            f"{cell['latency_ms']['mean']:.3f} | "
            f"{cell['speedup_ratio_of_means']:.3f}x | "
            f"{cell['acceptance']['draft_token_acceptance_rate']:.3f} | "
            f"{cell['acceptance']['mean_acceptance_length_including_bonus']:.3f} | "
            f"{cell['correctness']['exact_match_fraction']:.3f} | "
            f"{'yes' if cell['screen_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "This is a native learned-draft selection result, not P/D overlap evidence.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_worker(
    args: argparse.Namespace, *, mode: str, horizon: int, result_file: Path
) -> None:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "vllm"
    old_pythonpath = environment.get("PYTHONPATH")
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
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-horizon",
        str(horizon),
        "--result-file",
        str(result_file),
        "--model",
        args.model,
        "--draft-model",
        args.draft_model,
        "--requests-jsonl",
        args.requests_jsonl,
        "--output-dir",
        args.output_dir,
        "--request-offset",
        str(args.request_offset),
        "--num-requests",
        str(args.num_requests),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--output-tokens",
        str(args.output_tokens),
        "--block-size",
        str(args.block_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.ignore_eos:
        command.append("--ignore-eos")
    if args.require_exclusive_gpu:
        command.append("--require-exclusive-gpu")
    subprocess.run(command, env=environment, check=True)


def run_parent(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    horizons = parse_horizons(args.speculative_horizons)
    manifest_path = output_dir / "execution_manifest.json"
    hashed_inputs = {
        "requests_jsonl": Path(args.requests_jsonl).resolve(),
        "target_config": Path(args.model).resolve() / "config.json",
        "target_weight_index": Path(args.model).resolve()
        / "model.safetensors.index.json",
        "draft_config": Path(args.draft_model).resolve() / "config.json",
        "draft_weights": draft_weight_file(Path(args.draft_model).resolve()),
        "benchmark": Path(__file__).resolve(),
        "protocol": root / "docs/EAGLE3_DRAFT_SCREEN_PROTOCOL.md",
        "vllm_eagle": root / "vllm/vllm/v1/spec_decode/eagle.py",
        "vllm_metrics": root / "vllm/vllm/v1/spec_decode/metrics.py",
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "physical_gpu": args.cuda_visible_devices,
        "started_at_unix_ns": time_ns(),
        "input_sha256": {
            name: sha256_file(path) for name, path in hashed_inputs.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    launch_worker(args, mode="dense", horizon=0, result_file=output_dir / "dense.jsonl")
    for horizon in horizons:
        launch_worker(
            args,
            mode="eagle3",
            horizon=horizon,
            result_file=output_dir / f"eagle3_k{horizon}.jsonl",
        )
    summary = aggregate(args)
    write_summary(output_dir, summary)
    artifacts = (
        output_dir / "summary.json",
        output_dir / "summary.md",
        output_dir / "paper_table.csv",
        output_dir / "dense.jsonl",
        *(output_dir / f"eagle3_k{horizon}.jsonl" for horizon in horizons),
    )
    manifest["status"] = (
        "completed_valid" if summary["status"] == "valid" else "completed_invalid"
    )
    manifest["finished_at_unix_ns"] = time_ns()
    manifest["artifact_sha256"] = {
        path.name: sha256_file(path) for path in artifacts
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "valid":
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--speculative-horizons", default="2,3,4,8")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--worker-mode", choices=("dense", "eagle3"))
    parser.add_argument("--worker-horizon", type=int, default=0)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        parse_horizons(args.speculative_horizons)
    except ValueError as error:
        parser.error(str(error))
    if min(
        args.num_requests,
        args.max_context_tokens,
        args.output_tokens,
        args.block_size,
    ) <= 0:
        parser.error("request and token counts must be positive")
    if args.request_offset < 0:
        parser.error("request offset must be non-negative")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("gpu memory utilization must lie in (0, 1]")
    if "," in args.cuda_visible_devices or not args.cuda_visible_devices.strip():
        parser.error("selection timing requires exactly one physical GPU")
    if args.worker_mode is not None:
        if not args.result_file:
            parser.error("worker mode requires --result-file")
        if args.worker_mode == "eagle3" and args.worker_horizon <= 0:
            parser.error("EAGLE3 worker requires a positive horizon")
    return args


def main() -> None:
    args = parse_args()
    if args.worker_mode is None:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
