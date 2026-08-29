# SPDX-License-Identifier: Apache-2.0
"""Measure when a producer-side EAGLE3 proposal becomes transferable.

This is a source-side selection gate for a token-only P/D design.  The dense
worker measures cold target prefill through the first exact token.  The EAGLE3
worker performs the same cold target prefill, exports its first raw proposal
at the end of the proposer call, and lets native vLLM finish target verification
only so the proposal's accepted prefix can be audited.  The trace timestamp is
captured before that verification and therefore identifies when P could hand
the tiny token sequence to D while full target KV is still in flight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, perf_counter_ns, time_ns
from typing import Any

try:
    from experiments.benchmark_eagle3_draft import (
        bootstrap_mean_ci,
        describe,
        draft_weight_file,
        load_prompts,
        read_jsonl,
        sha256_file,
        spec_metric_snapshot,
        subtract_spec_metrics,
        write_jsonl,
    )
except ModuleNotFoundError:
    from benchmark_eagle3_draft import (  # type: ignore[no-redef]
        bootstrap_mean_ci,
        describe,
        draft_weight_file,
        load_prompts,
        read_jsonl,
        sha256_file,
        spec_metric_snapshot,
        subtract_spec_metrics,
        write_jsonl,
    )


ROOT = Path(__file__).resolve().parents[1]
TRACE_ENV = "VLLM_EAGLE_PROPOSAL_TRACE_PATH"


def common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def parse_bandwidths(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("bandwidths must be comma-separated numbers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("bandwidths must be positive")
    if len(values) != len(set(values)):
        raise ValueError("bandwidths must be unique")
    return tuple(sorted(values))


def target_kv_bytes_per_token(model_dir: Path, dtype_bytes: int = 2) -> int:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    head_dim = int(
        config.get(
            "head_dim",
            int(config["hidden_size"]) // int(config["num_attention_heads"]),
        )
    )
    return layers * 2 * kv_heads * head_dim * dtype_bytes


def transfer_ms(num_bytes: int, bandwidth_gbps: float) -> float:
    return num_bytes * 8.0 / (bandwidth_gbps * 1e9) * 1000.0


def _assert_exclusive_gpu() -> tuple[int, int]:
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    used_bytes = total_bytes - free_bytes
    if used_bytes > 1 * 2**30:
        raise RuntimeError(
            f"selected GPU is not exclusive: {used_bytes / 2**30:.2f} GiB used"
        )
    return used_bytes, total_bytes


def _load_worker_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    model_config = json.loads(
        (Path(args.model) / "config.json").read_text(encoding="utf-8")
    )
    return load_prompts(
        requests_jsonl=Path(args.requests_jsonl),
        request_offset=args.request_offset,
        num_requests=args.num_requests,
        max_context_tokens=args.max_context_tokens,
        block_size=args.block_size,
        output_tokens=args.horizon + 1,
        model_limit=int(model_config["max_position_embeddings"]),
    )


def _read_trace(path: Path, request_started_at_ns: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("EAGLE proposal trace was not published")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "ready" or payload.get("method") != "eagle3":
        raise RuntimeError("invalid EAGLE proposal trace payload")
    created_at_ns = int(payload["created_at_ns"])
    proposal_started_at_ns = int(payload["proposal_started_at_ns"])
    if not request_started_at_ns <= proposal_started_at_ns <= created_at_ns:
        raise RuntimeError("proposal timestamps are outside the request interval")
    return payload


def run_worker(args: argparse.Namespace) -> None:
    prompts = _load_worker_prompts(args)
    trace_path = Path(args.trace_path)
    if args.worker_mode == "eagle3":
        os.environ[TRACE_ENV] = str(trace_path.resolve())
    else:
        os.environ.pop(TRACE_ENV, None)

    import torch
    from vllm import LLM, SamplingParams

    initial_used = 0
    initial_total = 0
    if args.require_exclusive_gpu:
        initial_used, initial_total = _assert_exclusive_gpu()
    maximum_prompt_tokens = max(int(prompt["prompt_tokens"]) for prompt in prompts)
    engine_kwargs: dict[str, Any] = {}
    if args.worker_mode == "eagle3":
        engine_kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": args.horizon,
        }
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=maximum_prompt_tokens + args.horizon + 2,
        max_num_seqs=1,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        disable_log_stats=False,
        enforce_eager=args.enforce_eager,
        **engine_kwargs,
    )
    loaded_allocated_bytes = int(torch.cuda.memory_allocated())
    loaded_reserved_bytes = int(torch.cuda.memory_reserved())
    dense_sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    eagle_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.horizon + 1,
        ignore_eos=False,
    )
    sampling = eagle_sampling if args.worker_mode == "eagle3" else dense_sampling

    # Warm the chosen execution shape with a short, non-cacheable prefix.
    trace_path.unlink(missing_ok=True)
    warmup_prompt = list(prompts[0]["prompt_token_ids"][: args.block_size])
    engine.generate(
        [{"prompt_token_ids": warmup_prompt}], sampling, use_tqdm=False
    )
    torch.cuda.synchronize()
    trace_path.unlink(missing_ok=True)

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        trace_path.unlink(missing_ok=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        metrics_before = spec_metric_snapshot(engine)
        started_at_ns = perf_counter_ns()
        started = perf_counter()
        output = engine.generate(
            [{"prompt_token_ids": list(prompt["prompt_token_ids"])}],
            sampling,
            use_tqdm=False,
        )[0]
        torch.cuda.synchronize()
        completed_at_ns = perf_counter_ns()
        elapsed_ms = (perf_counter() - started) * 1000.0
        token_ids = [int(token) for token in output.outputs[0].token_ids]
        if not token_ids:
            raise RuntimeError("worker produced no token")
        row: dict[str, Any] = {
            "request_offset": prompt["request_offset"],
            "prompt_sha256": prompt["prompt_sha256"],
            "prompt_tokens": prompt["prompt_tokens"],
            "mode": args.worker_mode,
            "horizon": args.horizon,
            "token_ids": token_ids,
            "seed_token_id": token_ids[0],
            "request_started_at_ns": started_at_ns,
            "request_completed_at_ns": completed_at_ns,
            "elapsed_ms": elapsed_ms,
            "num_cached_tokens": int(output.num_cached_tokens or 0),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "loaded_allocated_bytes": loaded_allocated_bytes,
            "loaded_reserved_bytes": loaded_reserved_bytes,
            "initial_gpu_used_bytes": initial_used,
            "initial_gpu_total_bytes": initial_total,
        }
        if int(output.num_cached_tokens or 0) != 0:
            raise RuntimeError("source-side gate requires a cold prompt prefill")
        if args.worker_mode == "eagle3":
            trace = _read_trace(trace_path, started_at_ns)
            if int(trace["created_at_ns"]) > completed_at_ns:
                raise RuntimeError("proposal became ready after request completion")
            raw_drafts = [int(token) for token in trace["draft_token_ids"]]
            if int(trace["seed_token_id"]) != token_ids[0]:
                raise RuntimeError("published EAGLE seed differs from target output")
            row.update(
                {
                    "proposal": trace,
                    "proposal_ready_ms": (
                        int(trace["created_at_ns"]) - started_at_ns
                    )
                    / 1e6,
                    "proposal_observed_span_ms": int(trace["proposal_compute_ns"])
                    / 1e6,
                    "first_proposal_accepted_prefix": common_prefix_length(
                        raw_drafts, token_ids[1:]
                    ),
                    "spec_metrics": subtract_spec_metrics(
                        spec_metric_snapshot(engine), metrics_before
                    ),
                }
            )
        rows.append(row)
    write_jsonl(args.result_file, rows)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    dense = read_jsonl(output_dir / "dense.jsonl")
    eagle = read_jsonl(output_dir / "eagle3.jsonl")
    if len(dense) != args.num_requests or len(eagle) != args.num_requests:
        raise RuntimeError("worker result count does not match the frozen request count")
    dense_by_offset = {int(row["request_offset"]): row for row in dense}
    eagle_by_offset = {int(row["request_offset"]): row for row in eagle}
    if set(dense_by_offset) != set(eagle_by_offset):
        raise RuntimeError("dense and EAGLE request offsets differ")

    paired: list[dict[str, Any]] = []
    for offset in sorted(dense_by_offset):
        dense_row = dense_by_offset[offset]
        eagle_row = eagle_by_offset[offset]
        if dense_row["prompt_sha256"] != eagle_row["prompt_sha256"]:
            raise RuntimeError("paired prompt hashes differ")
        seed_match = int(dense_row["seed_token_id"]) == int(
            eagle_row["seed_token_id"]
        )
        paired.append(
            {
                "request_offset": offset,
                "prompt_tokens": int(eagle_row["prompt_tokens"]),
                "prompt_sha256": eagle_row["prompt_sha256"],
                "dense_seed_ready_ms": float(dense_row["elapsed_ms"]),
                "eagle_proposal_ready_ms": float(eagle_row["proposal_ready_ms"]),
                "source_extra_ms": float(eagle_row["proposal_ready_ms"])
                - float(dense_row["elapsed_ms"]),
                "eagle_proposal_observed_span_ms": float(
                    eagle_row["proposal_observed_span_ms"]
                ),
                "first_proposal_accepted_prefix": int(
                    eagle_row["first_proposal_accepted_prefix"]
                ),
                "seed_match": seed_match,
                "proposal_token_ids": eagle_row["proposal"]["draft_token_ids"],
            }
        )
    write_jsonl(output_dir / "paired.jsonl", paired)

    extra_ms = [float(row["source_extra_ms"]) for row in paired]
    accepted = [float(row["first_proposal_accepted_prefix"]) for row in paired]
    proposal_span = [
        float(row["eagle_proposal_observed_span_ms"]) for row in paired
    ]
    kv_bytes_per_token = target_kv_bytes_per_token(Path(args.model))
    bandwidth_results: dict[str, Any] = {}
    for bandwidth in parse_bandwidths(args.bandwidths_gbps):
        windows = [
            transfer_ms(
                int(row["prompt_tokens"]) * kv_bytes_per_token, bandwidth
            )
            for row in paired
        ]
        slack = [window - extra for window, extra in zip(windows, extra_ms, strict=True)]
        bandwidth_results[f"{bandwidth:g}"] = {
            "bandwidth_gbps": bandwidth,
            "full_target_kv_transfer_ms": describe(windows),
            "proposal_ready_before_full_kv_fraction": sum(value >= 0 for value in slack)
            / len(slack),
            "remaining_overlap_slack_ms": describe(slack),
        }

    ci_low, ci_high = bootstrap_mean_ci(extra_ms)
    summary = {
        "schema_version": 1,
        "status": "valid_source_side_selection",
        "scope": (
            "cold producer target prefill plus first EAGLE3 proposal; no LMCache "
            "transport or D-side replay is included"
        ),
        "model": args.model,
        "draft_model": args.draft_model,
        "requests_jsonl": args.requests_jsonl,
        "request_offset": args.request_offset,
        "samples": len(paired),
        "horizon": args.horizon,
        "target_kv_bytes_per_prompt_token": kv_bytes_per_token,
        "seed_match_fraction": sum(bool(row["seed_match"]) for row in paired)
        / len(paired),
        "source_extra_ms": {
            **describe(extra_ms),
            "bootstrap_95pct_ci": [ci_low, ci_high],
        },
        "eagle_proposal_observed_span_ms": {
            **describe(proposal_span),
            "interpretation": (
                "CPU wall span inside propose through token D2H readiness; because "
                "target CUDA work is asynchronous, this includes queued target "
                "prefill work and is not an isolated draft-kernel time"
            ),
        },
        "first_proposal_accepted_prefix": describe(accepted),
        "bandwidth_sweep": bandwidth_results,
        "decision": {
            "token_handoff_valid": all(bool(row["seed_match"]) for row in paired),
            "note": (
                "Passing only shows that proposal tokens become ready inside the "
                "modeled KV-transfer window. D-side verifier savings and producer "
                "throughput opportunity cost remain separate mandatory gates."
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _worker_command(args: argparse.Namespace, mode: str, result_file: Path) -> list[str]:
    return [
        args.python,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--model",
        args.model,
        "--draft-model",
        args.draft_model,
        "--requests-jsonl",
        args.requests_jsonl,
        "--output-dir",
        args.output_dir,
        "--result-file",
        str(result_file),
        "--trace-path",
        str(Path(args.output_dir) / "current_proposal.json"),
        "--request-offset",
        str(args.request_offset),
        "--num-requests",
        str(args.num_requests),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--horizon",
        str(args.horizon),
        "--block-size",
        str(args.block_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        *( ["--enforce-eager"] if args.enforce_eager else [] ),
        *( ["--require-exclusive-gpu"] if args.require_exclusive_gpu else [] ),
    ]


def orchestrate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    local_vllm = str(ROOT / "vllm")
    environment["PYTHONPATH"] = (
        local_vllm
        if not environment.get("PYTHONPATH")
        else f"{local_vllm}:{environment['PYTHONPATH']}"
    )
    commands: list[list[str]] = []
    for mode in ("dense", "eagle3"):
        result_file = output_dir / f"{mode}.jsonl"
        command = _worker_command(args, mode, result_file)
        commands.append(command)
        subprocess.run(command, check=True, cwd=ROOT, env=environment)
    summary = aggregate(args)
    manifest = {
        "schema_version": 1,
        "created_at_ns": time_ns(),
        "commands": commands,
        "gpu": args.gpu,
        "python": args.python,
        "hashes": {
            "requests": sha256_file(args.requests_jsonl),
            "target_config": sha256_file(Path(args.model) / "config.json"),
            "draft_weights": sha256_file(draft_weight_file(Path(args.draft_model))),
            "benchmark": sha256_file(Path(__file__)),
            "proposal_trace": sha256_file(
                ROOT
                / "vllm/vllm/v1/spec_decode/eagle_proposal_trace.py"
            ),
            "proposer": sha256_file(
                ROOT / "vllm/vllm/v1/spec_decode/llm_base_proposer.py"
            ),
        },
        "summary_status": summary["status"],
    }
    (output_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--max-context-tokens", type=int, default=131008)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--bandwidths-gbps", default="25,50,100")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--worker-mode", choices=("dense", "eagle3"))
    parser.add_argument("--result-file")
    parser.add_argument("--trace-path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_requests <= 0 or args.horizon <= 0:
        raise ValueError("request count and horizon must be positive")
    parse_bandwidths(args.bandwidths_gbps)
    if args.worker_mode:
        if not args.result_file or not args.trace_path:
            raise ValueError("worker mode requires --result-file and --trace-path")
        run_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
