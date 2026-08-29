# SPDX-License-Identifier: Apache-2.0
"""Screen frozen non-contiguous layer subsets as cheap draft networks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any

try:
    from experiments.benchmark_layer_sparse_draft import (
        _assert_exclusive_gpu,
        _cell_summary,
        _load_prompts,
        _model_config,
        fraction_order,
        parse_fractions,
        publish_visibility,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )
except ModuleNotFoundError:  # Direct execution from experiments/.
    from benchmark_layer_sparse_draft import (
        _assert_exclusive_gpu,
        _cell_summary,
        _load_prompts,
        _model_config,
        fraction_order,
        parse_fractions,
        publish_visibility,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )


def load_patterns(path: str | Path, *, full_depth: int) -> list[dict[str, Any]]:
    """Load and validate a frozen list of layer-subset candidates."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    patterns = raw.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("patterns JSON must contain a non-empty patterns list")
    validated = []
    identifiers: set[str] = set()
    for pattern in patterns:
        identifier = pattern.get("id") if isinstance(pattern, dict) else None
        indices = pattern.get("layer_indices") if isinstance(pattern, dict) else None
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[a-z0-9_]+", identifier) is None
        ):
            raise ValueError("pattern id must contain lowercase letters/digits/_")
        if identifier in identifiers:
            raise ValueError("pattern ids must be unique")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"{identifier}: layer_indices must be non-empty")
        if any(type(index) is not int for index in indices):
            raise ValueError(f"{identifier}: layer indices must be integers")
        if indices != sorted(set(indices)):
            raise ValueError(f"{identifier}: layer indices must be sorted and unique")
        if indices[0] != 0 or indices[-1] != full_depth - 1:
            raise ValueError(f"{identifier}: screen requires first and last layer")
        if indices[-1] >= full_depth:
            raise ValueError(f"{identifier}: layer index exceeds target depth")
        identifiers.add(identifier)
        validated.append(
            {
                "id": identifier,
                "layer_indices": indices,
                "active_layers": len(indices),
                "description": str(pattern.get("description", "")),
            }
        )
    return validated


def run_worker(args: argparse.Namespace) -> None:
    """Run one fixed layer subset over all requests and page fractions."""
    config = _model_config(Path(args.model))
    indices = json.loads(args.layer_indices_json)
    pattern = load_patterns(
        args.single_pattern_file, full_depth=config["num_hidden_layers"]
    )[0]
    if indices != pattern["layer_indices"] or args.candidate_id != pattern["id"]:
        raise ValueError("worker candidate differs from frozen single-pattern input")
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
    target_dir = Path(args.target_run_dir)
    seed_rows = read_jsonl(target_dir / "target_reference.jsonl")
    seed_by_offset = {int(row["request_offset"]): row for row in seed_rows}
    if len(seed_by_offset) < args.num_requests:
        raise ValueError("target run does not provide all committed seeds")

    stats_path = Path(args.stats_file).resolve()
    completion_path = Path(args.completion_file).resolve()
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
        hf_overrides={"sparsecache_draft_layer_indices": indices},
    )
    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.draft_tokens, ignore_eos=True
    )
    warmup_sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    fractions = parse_fractions(args.visibility_fractions)
    rows: list[dict[str, Any]] = []
    for request_index, prompt in enumerate(prompts):
        prompt_ids = list(prompt["prompt_token_ids"])
        seed = seed_by_offset[int(prompt["request_offset"])]
        if seed.get("prompt_sha256") != prompt["prompt_sha256"]:
            raise ValueError("target seed prompt digest mismatch")
        seed_token_id = int(seed["seed_token_id"])
        warmup_id = f"{args.candidate_id}-q{prompt['request_offset']}-warmup"
        publish_visibility(
            completion_path,
            request_id=warmup_id,
            fraction=1.0,
            trace_enabled=False,
        )
        engine.generate(
            [{"prompt_token_ids": prompt_ids}], warmup_sampling, use_tqdm=False
        )
        draft_input = prompt_ids + [seed_token_id]
        for order_index, fraction in enumerate(
            fraction_order(fractions, request_index)
        ):
            request_id = (
                f"{args.candidate_id}-q{prompt['request_offset']}-"
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
            output = engine.generate(
                [{"prompt_token_ids": draft_input}], sampling, use_tqdm=False
            )[0]
            torch.cuda.synchronize()
            rows.append(
                {
                    "request_id": request_id,
                    "request_offset": prompt["request_offset"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "prompt_tokens": prompt_tokens,
                    "candidate_id": args.candidate_id,
                    "layer_indices": indices,
                    "active_layers": len(indices),
                    "full_depth": config["num_hidden_layers"],
                    "visible_fraction": fraction,
                    "order_index": order_index,
                    "draft_tokens": args.draft_tokens,
                    "seed_token_id": seed_token_id,
                    "token_ids": list(output.outputs[0].token_ids),
                    "elapsed_ms": (perf_counter() - started) * 1000,
                    "num_cached_tokens": int(output.num_cached_tokens or 0),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "initial_gpu_used_bytes": initial_used,
                    "initial_gpu_total_bytes": initial_total,
                }
            )

    audit_seed = seed_by_offset[int(prompts[0]["request_offset"])]
    audit_prompt = list(prompts[0]["prompt_token_ids"]) + [
        int(audit_seed["seed_token_id"])
    ]
    for fraction in fractions:
        request_id = f"audit-{args.candidate_id}-f{fraction:.6f}"
        publish_visibility(
            completion_path,
            request_id=request_id,
            fraction=fraction,
            trace_enabled=True,
        )
        engine.generate([{"prompt_token_ids": audit_prompt}], sampling, use_tqdm=False)
    write_jsonl(args.result_file, rows)


def target_controls(
    target_dir: Path, *, num_requests: int, request_offset: int
) -> tuple[dict[int, list[int]], float]:
    """Load the frozen full-depth/full-page verifier replay control."""
    rows = [
        row
        for row in read_jsonl(target_dir / "depth_32.jsonl")
        if math.isclose(float(row["visible_fraction"]), 1.0)
        and request_offset <= int(row["request_offset"]) < request_offset + num_requests
    ]
    if len(rows) != num_requests:
        raise ValueError("target run lacks the requested dense verifier slice")
    return (
        {int(row["request_offset"]): list(row["token_ids"]) for row in rows},
        statistics.fmean(float(row["elapsed_ms"]) for row in rows),
    )


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate and summarize all subset/fraction cells."""
    config = _model_config(Path(args.model))
    patterns = load_patterns(args.patterns_json, full_depth=config["num_hidden_layers"])
    fractions = parse_fractions(args.visibility_fractions)
    target_by_request, dense_latency = target_controls(
        Path(args.target_run_dir),
        num_requests=args.num_requests,
        request_offset=args.request_offset,
    )
    errors: list[str] = []
    all_rows: list[dict[str, Any]] = []
    trace_summary: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []
    for pattern in patterns:
        identifier = str(pattern["id"])
        rows = read_jsonl(Path(args.output_dir) / f"candidate_{identifier}.jsonl")
        all_rows.extend(rows)
        traces = read_jsonl(Path(args.output_dir) / f"attention_{identifier}.jsonl")
        trace_summary[identifier] = {}
        for fraction in fractions:
            records = [
                row
                for row in traces
                if row.get("request_id") == f"audit-{identifier}-f{fraction:.6f}"
            ]
            trace_summary[identifier][f"{fraction:.6f}"] = {
                "records": len(records),
                "visible_pages": sorted({int(row["visible_pages"]) for row in records}),
                "candidate_pages": sorted(
                    {int(row["candidate_pages"]) for row in records}
                ),
            }
            if len(records) != args.draft_tokens:
                errors.append(f"{identifier}/{fraction}: incomplete physical trace")
            cell_rows = [
                row
                for row in rows
                if math.isclose(float(row["visible_fraction"]), fraction)
            ]
            if len(cell_rows) != args.num_requests:
                errors.append(f"{identifier}/{fraction}: incomplete request slice")
                continue
            cell = _cell_summary(cell_rows, target_by_request)
            cell.update(
                {
                    "candidate_id": identifier,
                    "description": pattern["description"],
                    "layer_indices": pattern["layer_indices"],
                    "active_layers": pattern["active_layers"],
                    "active_layer_fraction": (
                        pattern["active_layers"] / config["num_hidden_layers"]
                    ),
                    "visible_fraction": fraction,
                    "logical_kv_fraction_of_dense": (
                        pattern["active_layers"]
                        / config["num_hidden_layers"]
                        * fraction
                    ),
                    "speedup_vs_frozen_dense_control": (
                        dense_latency / cell["latency_ms_mean"]
                    ),
                }
            )
            cells.append(cell)
    if any(
        int(row.get("num_cached_tokens", -1)) < int(row["prompt_tokens"])
        for row in all_rows
    ):
        errors.append("a timed request missed part of its exact prompt prefix")
    if any(len(row.get("token_ids", [])) != args.draft_tokens for row in all_rows):
        errors.append("a timed request did not return the fixed draft horizon")
    viable = [
        cell
        for cell in cells
        if float(cell["acceptance_rate"]) >= 0.5
        and float(cell["speedup_vs_frozen_dense_control"]) >= 1.5
    ]
    return {
        "schema_version": 1,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "scope": "selection screen; dense timing is frozen from target run",
        "model": args.model,
        "requests_jsonl": args.requests_jsonl,
        "target_run_dir": args.target_run_dir,
        "patterns_json": args.patterns_json,
        "request_offset": args.request_offset,
        "num_requests": args.num_requests,
        "draft_tokens": args.draft_tokens,
        "visibility_fractions": list(fractions),
        "frozen_dense_latency_ms_mean": dense_latency,
        "cells": cells,
        "mechanism_trace": trace_summary,
        "pre_registered_screening_rule": {
            "minimum_acceptance_rate": 0.5,
            "minimum_speedup": 1.5,
            "passing_cells": [
                [cell["candidate_id"], cell["visible_fraction"]] for cell in viable
            ],
            "subset_path_viable": bool(viable),
        },
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "candidate_id",
        "active_layers",
        "visible_fraction",
        "logical_kv_fraction_of_dense",
        "latency_ms_mean",
        "speedup_vs_frozen_dense_control",
        "acceptance_rate",
        "accepted_prefix_mean",
        "zero_accept_fraction",
    ]
    with (output_dir / "paper_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["cells"])
    lines = [
        "# Non-contiguous layer-subset draft screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| candidate | layers | pages | mean ms | speedup | acceptance | prefix |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lines.append(
            f"| {cell['candidate_id']} | {cell['active_layers']} | "
            f"{cell['visible_fraction']:.0%} | {cell['latency_ms_mean']:.3f} | "
            f"{cell['speedup_vs_frozen_dense_control']:.3f}x | "
            f"{cell['acceptance_rate']:.3f} | "
            f"{cell['accepted_prefix_mean']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_worker(
    args: argparse.Namespace, *, pattern: dict[str, Any], gpu: str
) -> None:
    output_dir = Path(args.output_dir).resolve()
    single_pattern = output_dir / f"pattern_{pattern['id']}.json"
    single_pattern.write_text(
        json.dumps({"patterns": [pattern]}, indent=2) + "\n", encoding="utf-8"
    )
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
    identifier = str(pattern["id"])
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--candidate-id",
        identifier,
        "--layer-indices-json",
        json.dumps(pattern["layer_indices"], separators=(",", ":")),
        "--single-pattern-file",
        str(single_pattern),
        "--model",
        args.model,
        "--requests-jsonl",
        args.requests_jsonl,
        "--target-run-dir",
        args.target_run_dir,
        "--patterns-json",
        args.patterns_json,
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
        str(output_dir / f"candidate_{identifier}.jsonl"),
        "--stats-file",
        str(output_dir / f"attention_{identifier}.jsonl"),
        "--completion-file",
        str(output_dir / f"completion_{identifier}.json"),
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
    patterns = load_patterns(args.patterns_json, full_depth=config["num_hidden_layers"])
    gpus = tuple(item.strip() for item in args.cuda_visible_devices.split(","))
    if len(gpus) != 1 or not gpus[0]:
        raise ValueError("selection timing requires exactly one physical GPU")
    root = Path(__file__).resolve().parents[1]
    hashed_inputs = {
        "requests_jsonl": Path(args.requests_jsonl).resolve(),
        "model_config": Path(args.model).resolve() / "config.json",
        "target_summary": Path(args.target_run_dir).resolve() / "summary.json",
        "target_seeds": (
            Path(args.target_run_dir).resolve() / "target_reference.jsonl"
        ),
        "target_dense_rows": (Path(args.target_run_dir).resolve() / "depth_32.jsonl"),
        "patterns": Path(args.patterns_json).resolve(),
        "benchmark": Path(__file__).resolve(),
        "protocol": root / "docs/LAYER_SUBSET_DRAFT_SCREEN_PROTOCOL.md",
        "llama_model": root / "vllm/vllm/model_executor/models/llama.py",
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
    manifest_path = output_dir / "execution_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for pattern in patterns:
        launch_worker(args, pattern=pattern, gpu=gpus[0])
    summary = aggregate(args)
    write_summary(output_dir, summary)
    artifacts = (
        output_dir / "summary.json",
        output_dir / "summary.md",
        output_dir / "paper_table.csv",
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
    parser.add_argument("--target-run-dir", required=True)
    parser.add_argument("--patterns-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--visibility-fractions", default="0.05,1.0")
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--page-order", default="uniform")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--candidate-id", help=argparse.SUPPRESS)
    parser.add_argument("--layer-indices-json", help=argparse.SUPPRESS)
    parser.add_argument("--single-pattern-file", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--stats-file", help=argparse.SUPPRESS)
    parser.add_argument("--completion-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
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
    worker_fields = (
        args.candidate_id,
        args.layer_indices_json,
        args.single_pattern_file,
        args.result_file,
        args.stats_file,
        args.completion_file,
    )
    if args.candidate_id is not None and not all(worker_fields):
        parser.error("worker mode requires all hidden worker paths")
    return args


def main() -> None:
    args = parse_args()
    if args.candidate_id is None:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
