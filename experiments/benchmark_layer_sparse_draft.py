# SPDX-License-Identifier: Apache-2.0
"""Screen layer-sparse x page-sparse drafting on one frozen target model.

The benchmark truncates Llama to its first N original transformer layers and
reuses the original final norm/language-model head.  This is intentionally an
untrained early-exit screen, not a claimed production draft model.  Every
candidate is compared with the 100%-visible full-depth greedy continuation.
Timed requests hit vLLM's prefix cache, while one post-timing request per cell
records the exact physical page set selected by PROGRESSIVE_KV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any


def parse_ints(raw: str, *, field: str) -> tuple[int, ...]:
    """Parse a comma-separated tuple of unique positive integers."""
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError(f"{field} must contain integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{field} must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(values))


def parse_fractions(raw: str) -> tuple[float, ...]:
    """Parse unique page fractions and require the exact 100% control."""
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("visibility fractions must be numeric") from error
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("visibility fractions must lie in (0, 1]")
    if len(values) != len(set(values)):
        raise ValueError("visibility fractions must be unique")
    if 1.0 not in values:
        raise ValueError("visibility fractions must include 1.0")
    return tuple(sorted(values))


def common_prefix_length(left: list[int], right: list[int]) -> int:
    """Return the number of equal tokens before the first mismatch."""
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def fraction_order(
    fractions: tuple[float, ...], request_index: int
) -> tuple[float, ...]:
    """Rotate and reverse page fractions to limit order and thermal bias."""
    ordered = list(fractions)
    if request_index % 2:
        ordered.reverse()
    shift = (request_index // 2) % len(ordered)
    return tuple(ordered[shift:] + ordered[:shift])


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
    """Hash one immutable experiment input or output."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_visibility(
    path: Path, *, request_id: str, fraction: float, trace_enabled: bool
) -> None:
    """Atomically publish a fixed-S1 visibility snapshot."""
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


def _model_config(model: Path) -> dict[str, int]:
    raw = json.loads((model / "config.json").read_text(encoding="utf-8"))
    return {
        "num_hidden_layers": int(raw["num_hidden_layers"]),
        "max_position_embeddings": int(raw["max_position_embeddings"]),
        "num_key_value_heads": int(
            raw.get("num_key_value_heads", raw["num_attention_heads"])
        ),
        "head_dim": int(
            raw.get(
                "head_dim",
                int(raw["hidden_size"]) // int(raw["num_attention_heads"]),
            )
        ),
    }


def _load_prompts(
    *,
    requests_jsonl: Path,
    request_offset: int,
    num_requests: int,
    max_context_tokens: int,
    block_size: int,
    output_tokens: int,
    model_limit: int,
) -> list[dict[str, Any]]:
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
            or any(not isinstance(token, int) for token in prompt)
        ):
            raise ValueError("every request must contain integer prompt tokens")
        effective = min(len(prompt), safe_limit)
        effective -= effective % block_size
        prompt = prompt[:effective]
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
    lengths = {int(prompt["prompt_tokens"]) for prompt in prompts}
    if len(lengths) != 1:
        raise ValueError("screen requires one fixed effective prompt length per worker")
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


def run_worker(args: argparse.Namespace) -> None:
    """Run all page fractions for one model depth in one loaded engine."""
    model = Path(args.model)
    config = _model_config(model)
    prompts = _load_prompts(
        requests_jsonl=Path(args.requests_jsonl),
        request_offset=args.request_offset,
        num_requests=args.num_requests,
        max_context_tokens=args.max_context_tokens,
        block_size=args.block_size,
        output_tokens=args.draft_tokens,
        model_limit=config["max_position_embeddings"],
    )
    prompt_tokens = int(prompts[0]["prompt_tokens"])
    stats_path = Path(args.stats_file).resolve()
    completion_path = Path(args.completion_file).resolve()
    target_reference_path = Path(args.target_reference_file).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "VLLM_PROGRESSIVE_KV_DOC_START_TOKEN": "0",
            "VLLM_PROGRESSIVE_KV_DOC_END_TOKEN": str(prompt_tokens),
            "VLLM_PROGRESSIVE_KV_VISIBLE_FRACTIONS": "1.0",
            "VLLM_PROGRESSIVE_KV_PAGE_ORDER": args.page_order,
            "VLLM_PROGRESSIVE_KV_STATS_PATH": str(stats_path),
            "VLLM_PROGRESSIVE_KV_COMPLETION_PATH": str(completion_path),
            "VLLM_PROGRESSIVE_KV_VISIBILITY_MODE": "fixed_s1",
        }
    )

    import torch

    from vllm import LLM, SamplingParams

    initial_used = 0
    initial_total = 0
    if args.require_exclusive_gpu:
        initial_used, initial_total = _assert_exclusive_gpu()
    engine_kwargs: dict[str, Any] = {}
    if args.depth < config["num_hidden_layers"]:
        engine_kwargs["hf_overrides"] = {
            "num_hidden_layers": args.depth,
            "sparsecache_allow_truncated_layers": True,
        }
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=prompt_tokens + args.draft_tokens + 2,
        max_num_seqs=1,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=True,
        disable_log_stats=False,
        attention_config={"backend": "PROGRESSIVE_KV"},
        **engine_kwargs,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.draft_tokens,
        ignore_eos=True,
    )
    warmup_sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    reference_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.draft_tokens + 1,
        ignore_eos=True,
    )
    fractions = parse_fractions(args.visibility_fractions)
    rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    reference_by_offset: dict[int, dict[str, Any]] = {}
    if args.depth < config["num_hidden_layers"]:
        loaded_reference = read_jsonl(target_reference_path)
        reference_by_offset = {
            int(row["request_offset"]): row for row in loaded_reference
        }
        if len(reference_by_offset) != args.num_requests:
            raise ValueError("full-depth target reference is incomplete")
    for request_index, prompt in enumerate(prompts):
        prompt_ids = list(prompt["prompt_token_ids"])
        warmup_id = f"d{args.depth}-q{prompt['request_offset']}-warmup"
        publish_visibility(
            completion_path,
            request_id=warmup_id,
            fraction=1.0,
            trace_enabled=False,
        )
        engine.generate(
            [{"prompt_token_ids": prompt_ids}], warmup_sampling, use_tqdm=False
        )
        if args.depth == config["num_hidden_layers"]:
            reference_id = f"target-q{prompt['request_offset']}"
            publish_visibility(
                completion_path,
                request_id=reference_id,
                fraction=1.0,
                trace_enabled=False,
            )
            reference_output = engine.generate(
                [{"prompt_token_ids": prompt_ids}],
                reference_sampling,
                use_tqdm=False,
            )[0]
            reference_tokens = list(reference_output.outputs[0].token_ids)
            if len(reference_tokens) != args.draft_tokens + 1:
                raise RuntimeError("target reference did not return seed + horizon")
            reference = {
                "request_offset": prompt["request_offset"],
                "prompt_sha256": prompt["prompt_sha256"],
                "seed_token_id": reference_tokens[0],
                "target_token_ids": reference_tokens[1:],
            }
            reference_rows.append(reference)
            reference_by_offset[int(prompt["request_offset"])] = reference
        reference = reference_by_offset[int(prompt["request_offset"])]
        if reference.get("prompt_sha256") != prompt["prompt_sha256"]:
            raise ValueError("target reference prompt digest mismatch")
        seed_token_id = int(reference["seed_token_id"])
        draft_input_ids = prompt_ids + [seed_token_id]
        for order_index, fraction in enumerate(
            fraction_order(fractions, request_index)
        ):
            request_id = (
                f"d{args.depth}-q{prompt['request_offset']}-"
                f"o{order_index}-f{fraction:.6f}"
            )
            publish_visibility(
                completion_path,
                request_id=request_id,
                fraction=fraction,
                trace_enabled=False,
            )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = perf_counter()
            outputs = engine.generate(
                [{"prompt_token_ids": draft_input_ids}], sampling, use_tqdm=False
            )
            torch.cuda.synchronize()
            elapsed_ms = (perf_counter() - started) * 1000
            output = outputs[0]
            rows.append(
                {
                    "request_id": request_id,
                    "request_offset": prompt["request_offset"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "prompt_tokens": prompt_tokens,
                    "depth": args.depth,
                    "original_depth": config["num_hidden_layers"],
                    "visible_fraction": fraction,
                    "order_index": order_index,
                    "draft_tokens": args.draft_tokens,
                    "seed_token_id": seed_token_id,
                    "token_ids": list(output.outputs[0].token_ids),
                    "elapsed_ms": elapsed_ms,
                    "num_cached_tokens": int(output.num_cached_tokens or 0),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "initial_gpu_used_bytes": initial_used,
                    "initial_gpu_total_bytes": initial_total,
                }
            )

    # Trace I/O is deliberately outside all latency intervals.
    first_reference = reference_by_offset[int(prompts[0]["request_offset"])]
    audit_prompt = list(prompts[0]["prompt_token_ids"]) + [
        int(first_reference["seed_token_id"])
    ]
    for fraction in fractions:
        request_id = f"audit-d{args.depth}-f{fraction:.6f}"
        publish_visibility(
            completion_path,
            request_id=request_id,
            fraction=fraction,
            trace_enabled=True,
        )
        engine.generate([{"prompt_token_ids": audit_prompt}], sampling, use_tqdm=False)
    if reference_rows:
        write_jsonl(target_reference_path, reference_rows)
    write_jsonl(args.result_file, rows)


def _cell_summary(
    rows: list[dict[str, Any]], target_by_request: dict[int, list[int]]
) -> dict[str, Any]:
    latencies = [float(row["elapsed_ms"]) for row in rows]
    accepted = [
        common_prefix_length(
            list(row["token_ids"]), target_by_request[int(row["request_offset"])]
        )
        for row in rows
    ]
    proposed = sum(len(row["token_ids"]) for row in rows)
    return {
        "samples": len(rows),
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "accepted_draft_tokens": sum(accepted),
        "proposed_draft_tokens": proposed,
        "acceptance_rate": sum(accepted) / proposed,
        "accepted_prefix_mean": statistics.fmean(accepted),
        "zero_accept_fraction": sum(value == 0 for value in accepted) / len(accepted),
        "full_proposal_match_fraction": (
            sum(
                value == len(rows[index]["token_ids"])
                for index, value in enumerate(accepted)
            )
            / len(accepted)
        ),
        "accepted_prefix_by_request": accepted,
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the full matrix and compare every cell with dense target."""
    depths = parse_ints(args.depths, field="depths")
    fractions = parse_fractions(args.visibility_fractions)
    original_depth = _model_config(Path(args.model))["num_hidden_layers"]
    errors: list[str] = []
    if original_depth not in depths:
        errors.append("depth matrix omits the original full-depth control")
    all_rows: list[dict[str, Any]] = []
    trace_summary: dict[str, Any] = {}
    for depth in depths:
        rows = read_jsonl(Path(args.output_dir) / f"depth_{depth:02d}.jsonl")
        all_rows.extend(rows)
        traces = read_jsonl(
            Path(args.output_dir) / f"attention_depth_{depth:02d}.jsonl"
        )
        by_fraction: dict[str, Any] = {}
        for fraction in fractions:
            records = [
                row
                for row in traces
                if row.get("request_id") == f"audit-d{depth}-f{fraction:.6f}"
            ]
            by_fraction[f"{fraction:.6f}"] = {
                "records": len(records),
                "visible_pages": sorted({int(row["visible_pages"]) for row in records}),
                "candidate_pages": sorted(
                    {int(row["candidate_pages"]) for row in records}
                ),
            }
            if len(records) != args.draft_tokens:
                errors.append(
                    f"depth={depth}, fraction={fraction}: incomplete physical trace"
                )
        trace_summary[str(depth)] = by_fraction

    expected = {
        (depth, fraction, offset)
        for depth in depths
        for fraction in fractions
        for offset in range(
            args.request_offset, args.request_offset + args.num_requests
        )
    }
    actual = {
        (
            int(row["depth"]),
            float(row["visible_fraction"]),
            int(row["request_offset"]),
        )
        for row in all_rows
    }
    if expected != actual or len(all_rows) != len(expected):
        errors.append("rows do not form the requested depth/fraction/request matrix")
    if any(
        int(row.get("num_cached_tokens", -1)) < int(row["prompt_tokens"])
        for row in all_rows
    ):
        errors.append("a timed request missed part of its exact prompt prefix")
    if any(len(row.get("token_ids", [])) != args.draft_tokens for row in all_rows):
        errors.append("a timed request did not return the fixed draft horizon")

    seed_reference_rows = read_jsonl(Path(args.output_dir) / "target_reference.jsonl")
    monolithic_by_request = {
        int(row["request_offset"]): list(row["target_token_ids"])
        for row in seed_reference_rows
    }
    if len(monolithic_by_request) != args.num_requests:
        errors.append("full-depth monolithic seed reference is incomplete")
    dense_control_rows = [
        row
        for row in all_rows
        if int(row["depth"]) == original_depth
        and math.isclose(float(row["visible_fraction"]), 1.0)
    ]
    target_by_request = {
        int(row["request_offset"]): list(row["token_ids"]) for row in dense_control_rows
    }
    if len(target_by_request) != args.num_requests:
        errors.append("full-depth 100%-visible verifier replay is incomplete")
    monolithic_replay: dict[str, Any] | None = None
    if (
        len(monolithic_by_request) == args.num_requests
        and len(dense_control_rows) == args.num_requests
    ):
        replay = _cell_summary(dense_control_rows, monolithic_by_request)
        monolithic_replay = {
            "acceptance_rate": replay["acceptance_rate"],
            "accepted_prefix_mean": replay["accepted_prefix_mean"],
            "full_proposal_match_fraction": replay["full_proposal_match_fraction"],
            "interpretation": (
                "diagnostic only: greedy continuations can diverge between a "
                "monolithic seed+horizon pass and verifier replay"
            ),
        }
    cells: list[dict[str, Any]] = []
    if len(target_by_request) == args.num_requests:
        dense_latency = statistics.fmean(
            float(row["elapsed_ms"]) for row in dense_control_rows
        )
        for depth in depths:
            for fraction in fractions:
                rows = [
                    row
                    for row in all_rows
                    if int(row["depth"]) == depth
                    and math.isclose(float(row["visible_fraction"]), fraction)
                ]
                if not rows:
                    continue
                cell = _cell_summary(rows, target_by_request)
                cell.update(
                    {
                        "depth": depth,
                        "depth_fraction": depth / original_depth,
                        "visible_fraction": fraction,
                        "logical_kv_fraction_of_dense": (
                            depth / original_depth * fraction
                        ),
                        "speedup_vs_full_depth_full_pages": (
                            dense_latency / cell["latency_ms_mean"]
                        ),
                    }
                )
                cells.append(cell)

    raw_candidates = [
        cell
        for cell in cells
        if int(cell["depth"]) < original_depth
        and math.isclose(float(cell["visible_fraction"]), 1.0)
    ]
    combined_candidates = [
        cell
        for cell in cells
        if int(cell["depth"]) < original_depth
        and float(cell["visible_fraction"]) <= 0.1
    ]
    raw_viable = any(
        float(cell["acceptance_rate"]) >= 0.5
        and float(cell["speedup_vs_full_depth_full_pages"]) >= 1.5
        for cell in raw_candidates
    )
    combined_viable = any(
        float(cell["acceptance_rate"]) >= 0.5
        and float(cell["speedup_vs_full_depth_full_pages"]) >= 1.5
        for cell in combined_candidates
    )
    return {
        "schema_version": 1,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "scope": "selection screen; not an end-to-end P/D latency claim",
        "draft_definition": (
            "first N frozen target layers plus original final norm/lm_head; "
            "no early-exit training"
        ),
        "target_definition": (
            "full-depth, 100%-visible greedy verifier replay from a shared "
            "committed seed token"
        ),
        "model": args.model,
        "requests_jsonl": args.requests_jsonl,
        "request_offset": args.request_offset,
        "num_requests": args.num_requests,
        "prompt_tokens": (int(all_rows[0]["prompt_tokens"]) if all_rows else None),
        "draft_tokens": args.draft_tokens,
        "depths": list(depths),
        "visibility_fractions": list(fractions),
        "cells": cells,
        "monolithic_vs_verifier_replay": monolithic_replay,
        "mechanism_trace": trace_summary,
        "pre_registered_screening_rule": {
            "minimum_acceptance_rate": 0.5,
            "minimum_speedup": 1.5,
            "raw_early_exit_viable": raw_viable,
            "combined_page_layer_viable": combined_viable,
            "interpretation": (
                "failure means the untrained exit is rejected; it does not "
                "falsify a trained early-exit or external draft model"
            ),
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
            "depth",
            "depth_fraction",
            "visible_fraction",
            "logical_kv_fraction_of_dense",
            "latency_ms_mean",
            "speedup_vs_full_depth_full_pages",
            "acceptance_rate",
            "accepted_prefix_mean",
            "zero_accept_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["cells"])
    lines = [
        "# Layer-sparse draft screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| depth | visible pages | mean ms | speedup | acceptance | mean prefix |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lines.append(
            f"| {cell['depth']} | {cell['visible_fraction']:.0%} | "
            f"{cell['latency_ms_mean']:.3f} | "
            f"{cell['speedup_vs_full_depth_full_pages']:.3f}x | "
            f"{cell['acceptance_rate']:.3f} | "
            f"{cell['accepted_prefix_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "This is an untrained early-exit selection screen, not an "
                "end-to-end P/D latency result."
            ),
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_worker(args: argparse.Namespace, *, depth: int, gpu: str) -> None:
    output_dir = Path(args.output_dir).resolve()
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "vllm"
    old_pythonpath = environment.get("PYTHONPATH")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
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
        "--worker-depth",
        str(depth),
        "--model",
        args.model,
        "--requests-jsonl",
        args.requests_jsonl,
        "--output-dir",
        str(output_dir),
        "--request-offset",
        str(args.request_offset),
        "--num-requests",
        str(args.num_requests),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--visibility-fractions",
        args.visibility_fractions,
        "--draft-tokens",
        str(args.draft_tokens),
        "--block-size",
        str(args.block_size),
        "--page-order",
        args.page_order,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--result-file",
        str(output_dir / f"depth_{depth:02d}.jsonl"),
        "--stats-file",
        str(output_dir / f"attention_depth_{depth:02d}.jsonl"),
        "--completion-file",
        str(output_dir / f"completion_depth_{depth:02d}.json"),
        "--target-reference-file",
        str(output_dir / "target_reference.jsonl"),
    ]
    if args.require_exclusive_gpu:
        command.append("--require-exclusive-gpu")
    subprocess.run(command, env=environment, check=True)


def run_parent(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _model_config(Path(args.model))
    depths = parse_ints(args.depths, field="depths")
    if any(depth > config["num_hidden_layers"] for depth in depths):
        raise ValueError("a requested depth exceeds the frozen target depth")
    if config["num_hidden_layers"] not in depths:
        raise ValueError("depths must include the original full-depth control")
    gpus = tuple(item.strip() for item in args.cuda_visible_devices.split(","))
    if len(gpus) != 1 or not gpus[0]:
        raise ValueError("selection timing currently requires exactly one physical GPU")
    root = Path(__file__).resolve().parents[1]
    manifest_path = output_dir / "execution_manifest.json"
    hashed_inputs = {
        "requests_jsonl": Path(args.requests_jsonl).resolve(),
        "model_config": Path(args.model).resolve() / "config.json",
        "benchmark": Path(__file__).resolve(),
        "protocol": root / "docs/LAYER_SPARSE_DRAFT_SCREEN_PROTOCOL.md",
        "llama_loader": root / "vllm/vllm/model_executor/models/llama.py",
        "progressive_backend": (
            root / "vllm/vllm/v1/attention/backends/progressive_kv.py"
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "physical_gpu": gpus[0],
        "started_at_unix_ns": time_ns(),
        "input_sha256": {
            name: sha256_file(path) for name, path in hashed_inputs.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    execution_order = (config["num_hidden_layers"],) + tuple(
        depth for depth in depths if depth != config["num_hidden_layers"]
    )
    for depth in execution_order:
        launch_worker(args, depth=depth, gpu=gpus[0])
    summary = aggregate(args)
    write_summary(output_dir, summary)
    artifacts = (
        output_dir / "summary.json",
        output_dir / "summary.md",
        output_dir / "paper_table.csv",
        output_dir / "target_reference.jsonl",
    )
    manifest["status"] = (
        "completed_valid" if summary["status"] == "valid" else "completed_invalid"
    )
    manifest["finished_at_unix_ns"] = time_ns()
    manifest["artifact_sha256"] = {path.name: sha256_file(path) for path in artifacts}
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
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--max-context-tokens", type=int, default=65536)
    parser.add_argument("--depths", default="8,16,24,32")
    parser.add_argument("--visibility-fractions", default="0.05,1.0")
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--page-order", default="uniform")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--worker-depth", dest="depth", type=int)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--stats-file", help=argparse.SUPPRESS)
    parser.add_argument("--completion-file", help=argparse.SUPPRESS)
    parser.add_argument("--target-reference-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        parse_ints(args.depths, field="depths")
        parse_fractions(args.visibility_fractions)
    except ValueError as error:
        parser.error(str(error))
    if (
        min(
            args.num_requests,
            args.max_context_tokens,
            args.draft_tokens,
            args.block_size,
        )
        <= 0
    ):
        parser.error("request and token counts must be positive")
    if args.request_offset < 0:
        parser.error("request offset must be non-negative")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("gpu memory utilization must lie in (0, 1]")
    if args.depth is not None and not all(
        (
            args.result_file,
            args.stats_file,
            args.completion_file,
            args.target_reference_file,
        )
    ):
        parser.error("worker mode requires result/stats/completion paths")
    return args


def main() -> None:
    args = parse_args()
    if args.depth is None:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
