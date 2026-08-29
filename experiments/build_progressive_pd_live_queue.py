"""Build the audited live SparseCache-PD execution queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_PACK_ROOT = Path("outputs/progressive_kv/pd_packs/llama31_8b")
DEFAULT_RUN_ROOT = Path("outputs/progressive_kv/live_matrix/llama31_8b")
DEFAULT_MODEL_PATH = Path("/data/models/llama/Llama-3.1-8B-Instruct")
DEFAULT_MODEL_NAME = "Llama-3.1-8B-Instruct"
DEFAULT_SERVED_MODEL_NAME = "sparsecache-llama31-8b"
PYTHON = "/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python"
# Five-point schedules make the second bundle 20% of a long prompt. At
# 25 Gbps that bundle arrives after the entire eight-token draft horizon, so
# the nominal continuous arm degenerates into fixed-S1. Equal 5% bundles let
# arrival events occur at the same cadence as real sparse draft steps while
# preserving exactly the same total bytes on the controlled link.
FRACTIONS = ",".join(f"{step / 20:.2f}" for step in range(1, 21))
ARMS = "baseline,fixed_s1,continuous"
PROTECTED_PREFIX_TOKENS = 256
PROTECTED_SUFFIX_TOKENS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bandwidths", default="25,50")
    parser.add_argument("--core-requests", type=int, default=100)
    parser.add_argument("--confirmation-requests", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument(
        "--protected-prefix-tokens", type=int, default=PROTECTED_PREFIX_TOKENS
    )
    parser.add_argument(
        "--protected-suffix-tokens", type=int, default=PROTECTED_SUFFIX_TOKENS
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _require_passed(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("status") != "passed":
        raise ValueError(f"dataset audit did not pass: {path}")
    return value


def load_model_spec(
    model_path: Path = DEFAULT_MODEL_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
) -> dict[str, Any]:
    """Load the exact model geometry used for controlled-link byte accounting."""
    model_config = _load_json(model_path / "config.json")
    num_attention_heads = int(model_config["num_attention_heads"])
    head_dim = int(
        model_config.get(
            "head_dim", int(model_config["hidden_size"]) // num_attention_heads
        )
    )
    eos_token_ids: list[int] = []
    generation_config_path = model_path / "generation_config.json"
    if generation_config_path.is_file():
        generation_config = _load_json(generation_config_path)
        configured_eos = generation_config.get("eos_token_id", [])
        if isinstance(configured_eos, int) and not isinstance(configured_eos, bool):
            eos_token_ids = [configured_eos]
        elif isinstance(configured_eos, list) and all(
            isinstance(token, int) and not isinstance(token, bool) and token >= 0
            for token in configured_eos
        ):
            eos_token_ids = sorted(set(configured_eos))
        else:
            raise ValueError("generation_config eos_token_id is invalid")
    return {
        "name": model_name,
        "path": str(model_path),
        "served_model_name": served_model_name,
        "max_position_embeddings": int(model_config["max_position_embeddings"]),
        "eos_token_ids": eos_token_ids,
        "logical_bf16_kv_bytes_per_token": (
            2
            * int(model_config["num_hidden_layers"])
            * int(model_config.get("num_key_value_heads", num_attention_heads))
            * head_dim
            * 2
        ),
    }


def build_live_job(
    *,
    job_id: str,
    phase: str,
    status: str,
    pack_dir: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    requests: int,
    request_offset: int,
    warmups: int,
    bandwidth: int | str,
    arms: str,
    fixed_output_tokens: int | None,
    max_draft_tokens: int,
    model: dict[str, Any],
    protected_prefix_tokens: int,
    protected_suffix_tokens: int,
    priority_jsonl: Path | None = None,
    priority_chunk_tokens: int = 256,
    priority_mode: str | None = None,
) -> dict[str, Any]:
    if (
        requests <= 0
        or request_offset < 0
        or request_offset + requests > int(manifest["requests"])
    ):
        raise ValueError(f"invalid request count for {job_id}")
    benchmark = [
        PYTHON,
        "experiments/benchmark_progressive_pd_live.py",
        "--endpoint",
        "/v1/completions",
        "--requests-jsonl",
        str(pack_dir / "requests.jsonl"),
        "--output-jsonl",
        str(run_dir / "results.jsonl"),
        "--num-requests",
        str(requests),
        "--request-offset",
        str(request_offset),
        "--warmup-requests",
        str(warmups),
        "--arms",
        arms,
    ]
    aggregate = [
        PYTHON,
        "experiments/aggregate_progressive_pd_live.py",
        "--results-jsonl",
        str(run_dir / "results.jsonl"),
        "--metadata-jsonl",
        str(pack_dir / "metadata.jsonl"),
        "--scheduler-stats-jsonl",
        str(run_dir / "scheduler.jsonl"),
        "--attention-stats-jsonl",
        str(run_dir / "attention.jsonl"),
        "--link-stats-jsonl",
        str(run_dir / "link.jsonl"),
        "--output-dir",
        str(run_dir / "aggregate"),
        "--expected-requests",
        str(requests),
        "--arms",
        arms,
    ]
    if fixed_output_tokens is not None:
        fixed = ["--fixed-output-tokens", str(fixed_output_tokens)]
        benchmark.extend(fixed)
        aggregate.extend(fixed)
    aggregate.extend(
        [
            "--protected-prefix-tokens",
            str(protected_prefix_tokens),
            "--protected-suffix-tokens",
            str(protected_suffix_tokens),
        ]
    )
    if model.get("eos_token_ids"):
        aggregate.extend(
            [
                "--stop-token-ids",
                ",".join(str(token) for token in model["eos_token_ids"]),
            ]
        )
    if priority_jsonl is not None:
        benchmark.extend(["--priority-jsonl", str(priority_jsonl)])
        aggregate.extend(
            [
                "--priority-jsonl",
                str(priority_jsonl),
                "--priority-chunk-tokens",
                str(priority_chunk_tokens),
            ]
        )
    return {
        "id": job_id,
        "phase": phase,
        "status": status,
        "pack_dir": str(pack_dir),
        "run_dir": str(run_dir),
        "dataset": manifest["dataset"],
        "requests": requests,
        "request_offset": request_offset,
        "warmup_requests": warmups,
        "bandwidth_gbps": bandwidth,
        "arms": arms.split(","),
        "priority_schedule": {
            "mode": priority_mode,
            "jsonl": str(priority_jsonl) if priority_jsonl is not None else None,
            "chunk_tokens": priority_chunk_tokens,
        },
        "pd_roles": {"prefiller": "kv_producer", "decoder": "kv_consumer"},
        "fixed_output_tokens": fixed_output_tokens,
        "model": model,
        "pack_hashes": {
            "requests_sha256": manifest["requests_sha256"],
            "metadata_sha256": manifest["metadata_sha256"],
        },
        "decoder_environment": {
            "VLLM_PROGRESSIVE_KV_DOC_START_TOKEN": "0",
            "VLLM_PROGRESSIVE_KV_DOC_END_TOKEN": str(
                model["max_position_embeddings"]
            ),
            "VLLM_PROGRESSIVE_KV_PROTECTED_PREFIX_TOKENS": str(
                protected_prefix_tokens
            ),
            "VLLM_PROGRESSIVE_KV_PROTECTED_SUFFIX_TOKENS": str(
                protected_suffix_tokens
            ),
            "VLLM_PROGRESSIVE_KV_COMPLETION_PATH": str(
                run_dir / "completion.json"
            ),
            "VLLM_PROGRESSIVE_KV_STATS_PATH": str(run_dir / "attention.jsonl"),
            "VLLM_PROGRESSIVE_PD_STATS_PATH": str(run_dir / "scheduler.jsonl"),
        },
        "decoder_connector_extra_config": {
            "lmcache.mp.progressive_retrieve_fractions": FRACTIONS,
            "lmcache.mp.progressive_retrieve_priority": "uniform",
            "lmcache.mp.progressive_protected_prefix_tokens": (
                protected_prefix_tokens
            ),
            "lmcache.mp.progressive_protected_suffix_tokens": (
                protected_suffix_tokens
            ),
            "lmcache.mp.progressive_completion_path": str(
                run_dir / "completion.json"
            ),
            "lmcache.mp.controlled_link_stats_path": str(run_dir / "link.jsonl"),
            "lmcache.mp.progressive_link_gbps": bandwidth,
            "lmcache.mp.progressive_bytes_per_token": model[
                "logical_bf16_kv_bytes_per_token"
            ],
        },
        "proxy_args": {
            "max_draft_tokens": max_draft_tokens,
            "start_fraction": 0.05,
        },
        "benchmark_command": benchmark,
        "aggregate_command": aggregate,
        "validity_requirements": {
            "exclusive_prefill_gpu": True,
            "exclusive_decode_gpu": True,
            "minimum_free_gpu_count": 2,
            "fresh_run_directory": True,
            "aggregate_gates_must_pass": True,
            "request_scoped_priority_gate": priority_jsonl is not None,
        },
    }


def build_queue(
    pack_root: Path,
    run_root: Path,
    *,
    bandwidths: tuple[int, ...],
    core_requests: int,
    confirmation_requests: int,
    warmup_requests: int,
    max_draft_tokens: int,
    model_path: Path = DEFAULT_MODEL_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
    protected_prefix_tokens: int = PROTECTED_PREFIX_TOKENS,
    protected_suffix_tokens: int = PROTECTED_SUFFIX_TOKENS,
) -> dict[str, Any]:
    if not bandwidths or len(bandwidths) != len(set(bandwidths)):
        raise ValueError("bandwidths must be a non-empty unique list")
    if any(value <= 0 for value in bandwidths):
        raise ValueError("bandwidths must be positive")
    if min(core_requests, confirmation_requests, max_draft_tokens) <= 0:
        raise ValueError("request and draft counts must be positive")
    if warmup_requests < 0:
        raise ValueError("warmup count cannot be negative")
    if protected_prefix_tokens < 0 or protected_suffix_tokens < 0:
        raise ValueError("protected token counts cannot be negative")

    model = load_model_spec(model_path, model_name, served_model_name)

    ruler_audit = _require_passed(pack_root / "controlled/ruler_qa2_audit.json")
    quality_audit = _require_passed(pack_root / "quality_audit.json")
    longbench_v2_audit = _require_passed(pack_root / "longbench_v2_audit.json")
    del quality_audit, longbench_v2_audit

    ruler_cells = {}
    for cell in ruler_audit["cells"]:
        key = (int(cell["prompt_tokens"]), str(cell["placement"]))
        source_dir = str(cell["source_length_directory"])
        pack_dir = pack_root / "controlled/ruler_qa2" / source_dir / key[1]
        manifest = _load_json(pack_dir / "manifest.json")
        ruler_cells[key] = (pack_dir, manifest)

    jobs = []
    scheduling_length = (
        65536 if 65536 in ruler_audit["lengths"] else max(ruler_audit["lengths"])
    )
    ready_keys = {
        (length, "evidence_first", bandwidth)
        for length in ruler_audit["lengths"]
        for bandwidth in bandwidths
    }
    ready_keys.update(
        (scheduling_length, placement, bandwidth)
        for placement in ruler_audit["placements"]
        for bandwidth in bandwidths
    )
    for length, placement, bandwidth in sorted(ready_keys):
        pack_dir, manifest = ruler_cells[(length, placement)]
        job_id = f"core_ruler_{length}_{placement}_bw{bandwidth}_n{core_requests}"
        jobs.append(
            build_live_job(
                job_id=job_id,
                phase="core_mechanism",
                status="ready_when_two_gpus_are_exclusive",
                pack_dir=pack_dir,
                run_dir=run_root / job_id,
                manifest=manifest,
                requests=core_requests,
                request_offset=0,
                warmups=warmup_requests,
                bandwidth=bandwidth,
                arms=ARMS,
                fixed_output_tokens=32,
                max_draft_tokens=max_draft_tokens,
                model=model,
                protected_prefix_tokens=protected_prefix_tokens,
                protected_suffix_tokens=protected_suffix_tokens,
            )
        )

    for (length, placement), (pack_dir, manifest) in sorted(ruler_cells.items()):
        for bandwidth in bandwidths:
            job_id = (
                f"confirm_ruler_{length}_{placement}_bw{bandwidth}_"
                f"n{confirmation_requests}_offset{core_requests}"
            )
            jobs.append(
                build_live_job(
                    job_id=job_id,
                    phase="controlled_confirmation",
                    status="blocked_on_core_gate_and_method_freeze",
                    pack_dir=pack_dir,
                    run_dir=run_root / job_id,
                    manifest=manifest,
                    requests=confirmation_requests,
                    request_offset=core_requests,
                    warmups=warmup_requests,
                    bandwidth=bandwidth,
                    arms=ARMS,
                    fixed_output_tokens=32,
                    max_draft_tokens=max_draft_tokens,
                    model=model,
                    protected_prefix_tokens=protected_prefix_tokens,
                    protected_suffix_tokens=protected_suffix_tokens,
                )
            )

    quality_root = pack_root / "quality"
    quality_jobs = []
    for manifest_path in sorted(quality_root.glob("*/manifest.json")):
        pack_dir = manifest_path.parent
        manifest = _load_json(manifest_path)
        job_id = f"quality_{manifest['dataset']}_bwFROZEN_n{manifest['requests']}"
        quality_jobs.append(
            build_live_job(
                job_id=job_id,
                phase="natural_quality",
                status="template_blocked_on_method_freeze",
                pack_dir=pack_dir,
                run_dir=run_root / job_id,
                manifest=manifest,
                requests=int(manifest["requests"]),
                request_offset=0,
                warmups=warmup_requests,
                bandwidth="{FROZEN_BANDWIDTH_GBPS}",
                arms="baseline,continuous",
                fixed_output_tokens=None,
                max_draft_tokens=max_draft_tokens,
                model=model,
                protected_prefix_tokens=protected_prefix_tokens,
                protected_suffix_tokens=protected_suffix_tokens,
            )
        )
    jobs.extend(quality_jobs)

    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise AssertionError("live queue contains duplicate job IDs")
    statuses: dict[str, int] = {}
    phases: dict[str, int] = {}
    for job in jobs:
        statuses[job["status"]] = statuses.get(job["status"], 0) + 1
        phases[job["phase"]] = phases.get(job["phase"], 0) + 1
    return {
        "schema_version": 1,
        "status": "execution queue; no experiment result",
        "pack_root": str(pack_root),
        "run_root": str(run_root),
        "protocol": {
            "model": model,
            "pd_roles": {"prefiller": "kv_producer", "decoder": "kv_consumer"},
            "page_order": "uniform",
            "protected_prompt_tokens": {
                "prefix": protected_prefix_tokens,
                "suffix": protected_suffix_tokens,
                "first_bundle_is_atomic": True,
            },
            "progressive_fractions": FRACTIONS,
            "max_draft_tokens": max_draft_tokens,
            "warmups_replay_measured_shapes": True,
            "core_bandwidths_gbps": list(bandwidths),
            "bootstrap_samples": 10_000,
        },
        "counts": {
            "jobs": len(jobs),
            "by_phase": phases,
            "by_status": statuses,
        },
        "jobs": jobs,
    }


def main() -> None:
    args = parse_args()
    bandwidths = tuple(
        int(item.strip()) for item in args.bandwidths.split(",") if item.strip()
    )
    payload = build_queue(
        args.pack_root,
        args.run_root,
        bandwidths=bandwidths,
        core_requests=args.core_requests,
        confirmation_requests=args.confirmation_requests,
        warmup_requests=args.warmup_requests,
        max_draft_tokens=args.max_draft_tokens,
        model_path=args.model_path,
        model_name=args.model_name,
        served_model_name=args.served_model_name,
        protected_prefix_tokens=args.protected_prefix_tokens,
        protected_suffix_tokens=args.protected_suffix_tokens,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
