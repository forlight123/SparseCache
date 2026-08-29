"""Build the paired semantic-scheduling live execution queue."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from experiments.build_progressive_pd_live_queue import (
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_PATH,
        DEFAULT_SERVED_MODEL_NAME,
        PROTECTED_PREFIX_TOKENS,
        PROTECTED_SUFFIX_TOKENS,
        PYTHON,
        build_live_job,
        load_model_spec,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_progressive_pd_live_queue import (
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_PATH,
        DEFAULT_SERVED_MODEL_NAME,
        PROTECTED_PREFIX_TOKENS,
        PROTECTED_SUFFIX_TOKENS,
        PYTHON,
        build_live_job,
        load_model_spec,
    )


DEFAULT_PACK_ROOT = Path("outputs/progressive_kv/pd_packs/llama31_8b")
DEFAULT_SCHEDULE_ROOT = DEFAULT_PACK_ROOT / "schedules/ruler_qa2/65536/original"
DEFAULT_RUN_ROOT = Path("outputs/progressive_kv/live_scheduling/llama31_8b")
MODES = ("sequential", "uniform", "random", "bm25", "oracle")
ARMS = "fixed_s1,continuous"
CHUNK_TOKENS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--schedule-root", type=Path, default=DEFAULT_SCHEDULE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bandwidths", default="25,50")
    parser.add_argument("--selection-requests", type=int, default=100)
    parser.add_argument("--confirmation-requests", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--start-fraction", type=float, default=0.05)
    parser.add_argument("--tranche-fraction", type=float, default=0.05)
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


def _fraction_schedule(tranche_fraction: float) -> str:
    steps = round(1.0 / tranche_fraction)
    if (
        not 0.0 < tranche_fraction < 1.0
        or steps < 2
        or not math.isclose(
            tranche_fraction * steps, 1.0, rel_tol=0.0, abs_tol=1e-9
        )
    ):
        raise ValueError("tranche fraction must evenly partition (0, 1]")
    return ",".join(f"{step / steps:.10g}" for step in range(1, steps + 1))


def _commands(
    *,
    pack_dir: Path,
    run_dir: Path,
    sidecars: dict[str, Path],
    requests: int,
    offset: int,
    warmups: int,
    stop_token_ids: list[int],
) -> tuple[list[str], list[str]]:
    benchmark = [
        PYTHON,
        "experiments/benchmark_progressive_pd_scheduling_live.py",
        "--endpoint",
        "/v1/completions",
        "--requests-jsonl",
        str(pack_dir / "requests.jsonl"),
        "--output-jsonl",
        str(run_dir / "results.jsonl"),
        "--num-requests",
        str(requests),
        "--request-offset",
        str(offset),
        "--warmup-requests",
        str(warmups),
        "--fixed-output-tokens",
        "32",
        "--arms",
        ARMS,
    ]
    aggregate = [
        PYTHON,
        "experiments/aggregate_progressive_pd_scheduling_live.py",
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
        "--fixed-output-tokens",
        "32",
        "--arms",
        ARMS,
        "--protected-prefix-tokens",
        str(PROTECTED_PREFIX_TOKENS),
        "--protected-suffix-tokens",
        str(PROTECTED_SUFFIX_TOKENS),
        "--priority-chunk-tokens",
        str(CHUNK_TOKENS),
    ]
    if stop_token_ids:
        aggregate.extend(
            ["--stop-token-ids", ",".join(str(token) for token in stop_token_ids)]
        )
    for mode, path in sidecars.items():
        specification = f"{mode}={path}"
        benchmark.extend(["--priority-sidecar", specification])
        aggregate.extend(["--priority-sidecar", specification])
    return benchmark, aggregate


def build_scheduling_queue(
    pack_root: Path,
    schedule_root: Path,
    run_root: Path,
    *,
    bandwidths: tuple[int, ...],
    selection_requests: int,
    confirmation_requests: int,
    warmup_requests: int,
    max_draft_tokens: int,
    start_fraction: float = 0.05,
    tranche_fraction: float = 0.05,
    model_path: Path = DEFAULT_MODEL_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
) -> dict[str, Any]:
    if (
        not bandwidths
        or len(bandwidths) != len(set(bandwidths))
        or any(value <= 0 for value in bandwidths)
    ):
        raise ValueError("bandwidths must be positive and unique")
    if min(selection_requests, confirmation_requests, max_draft_tokens) <= 0:
        raise ValueError("request and draft counts must be positive")
    if warmup_requests < 0:
        raise ValueError("warmup count cannot be negative")
    fractions = _fraction_schedule(tranche_fraction)
    if (
        not 0.0 < start_fraction < 1.0
        or not math.isclose(
            start_fraction / tranche_fraction,
            round(start_fraction / tranche_fraction),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("start fraction must align to the tranche fraction")

    audit = _load_json(schedule_root / "audit.json")
    if audit.get("status") != "passed" or tuple(audit.get("modes", ())) != MODES:
        raise ValueError("scheduling audit must pass with the frozen five modes")
    if selection_requests + confirmation_requests > int(audit["requests"]):
        raise ValueError("selection and confirmation rows overlap or exceed the pack")
    pack_dir = pack_root / "controlled/ruler_qa2/65536/original"
    manifest = _load_json(pack_dir / "manifest.json")
    if int(manifest["requests"]) != int(audit["requests"]):
        raise ValueError("schedule and RULER pack request counts differ")
    sidecars = {
        mode: schedule_root / f"priority_{mode}.jsonl" for mode in MODES
    }
    if any(not path.is_file() for path in sidecars.values()):
        raise FileNotFoundError("one or more frozen priority sidecars are missing")
    model = load_model_spec(model_path, model_name, served_model_name)

    jobs = []
    phase_specs = (
        (
            "scheduling_selection",
            "ready_when_two_gpus_are_exclusive",
            selection_requests,
            0,
        ),
        (
            "scheduling_confirmation",
            "blocked_on_scheduling_selection_freeze",
            confirmation_requests,
            selection_requests,
        ),
    )
    for phase, status, requests, offset in phase_specs:
        for bandwidth in bandwidths:
            tranche_suffix = ""
            if not math.isclose(tranche_fraction, 0.05):
                tranche_suffix = f"_tranche{round(tranche_fraction * 10_000)}bp"
            job_id = (
                f"{phase}_ruler_65536_original_bw{bandwidth}_n{requests}_"
                f"offset{offset}{tranche_suffix}"
            )
            run_dir = run_root / job_id
            job = build_live_job(
                job_id=job_id,
                phase=phase,
                status=status,
                pack_dir=pack_dir,
                run_dir=run_dir,
                manifest=manifest,
                requests=requests,
                request_offset=offset,
                warmups=warmup_requests,
                bandwidth=bandwidth,
                arms=ARMS,
                fixed_output_tokens=32,
                max_draft_tokens=max_draft_tokens,
                model=model,
                protected_prefix_tokens=PROTECTED_PREFIX_TOKENS,
                protected_suffix_tokens=PROTECTED_SUFFIX_TOKENS,
            )
            job["proxy_args"]["start_fraction"] = start_fraction
            job["decoder_connector_extra_config"][
                "lmcache.mp.progressive_retrieve_fractions"
            ] = fractions
            benchmark, aggregate = _commands(
                pack_dir=pack_dir,
                run_dir=run_dir,
                sidecars=sidecars,
                requests=requests,
                offset=offset,
                warmups=warmup_requests,
                stop_token_ids=list(model["eos_token_ids"]),
            )
            job["benchmark_command"] = benchmark
            job["aggregate_command"] = aggregate
            job["schedule_modes"] = list(MODES)
            job["priority_sidecars"] = {
                mode: {
                    "path": str(path),
                    "sha256": audit["priority_sha256"][mode],
                    "role": (
                        "analysis_upper_bound" if mode == "oracle" else "online"
                    ),
                }
                for mode, path in sidecars.items()
            }
            job["validity_requirements"].update(
                {
                    "complete_schedule_x_arm_pairing": True,
                    "single_shared_prefill_across_schedules": True,
                    "runtime_priority_matches_sidecar": True,
                    "cross_schedule_exact_output_equivalence": True,
                    "selection_confirmation_rows_disjoint": True,
                }
            )
            jobs.append(job)

    return {
        "schema_version": 1,
        "status": "execution queue; no experiment result",
        "pack_root": str(pack_root),
        "schedule_root": str(schedule_root),
        "run_root": str(run_root),
        "protocol": {
            "model": model,
            "dataset": "RULER qa_2",
            "prompt_tokens": 65536,
            "schedule_modes": list(MODES),
            "online_modes": [mode for mode in MODES if mode != "oracle"],
            "oracle_role": "analysis upper bound; never claimed as online policy",
            "arms": ARMS.split(","),
            "chunk_tokens": CHUNK_TOKENS,
            "start_fraction": start_fraction,
            "tranche_fraction": tranche_fraction,
            "retrieve_fractions": fractions,
            "protected_prefix_tokens": PROTECTED_PREFIX_TOKENS,
            "protected_suffix_tokens": PROTECTED_SUFFIX_TOKENS,
            "bandwidths_gbps": list(bandwidths),
            "paired_execution": (
                "request-interleaved schedule x arm rotation with one shared "
                "producer prefill per source request"
            ),
            "selection_rows": [0, selection_requests],
            "confirmation_rows": [
                selection_requests,
                selection_requests + confirmation_requests,
            ],
        },
        "counts": {
            "jobs": len(jobs),
            "selection": len(bandwidths),
            "confirmation": len(bandwidths),
            "measured_conditions_per_request": len(MODES) * len(ARMS.split(",")),
        },
        "jobs": jobs,
    }


def main() -> None:
    args = parse_args()
    bandwidths = tuple(
        int(item.strip()) for item in args.bandwidths.split(",") if item.strip()
    )
    queue = build_scheduling_queue(
        args.pack_root,
        args.schedule_root,
        args.run_root,
        bandwidths=bandwidths,
        selection_requests=args.selection_requests,
        confirmation_requests=args.confirmation_requests,
        warmup_requests=args.warmup_requests,
        max_draft_tokens=args.max_draft_tokens,
        start_fraction=args.start_fraction,
        tranche_fraction=args.tranche_fraction,
        model_path=args.model_path,
        model_name=args.model_name,
        served_model_name=args.served_model_name,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(queue["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
