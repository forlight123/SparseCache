"""Fail-closed aggregation for the paired progressive scheduling experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from experiments.aggregate_progressive_pd_live import (
        aggregate_live_run,
        describe,
        parse_token_ids,
        read_jsonl,
        write_live_artifacts,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from aggregate_progressive_pd_live import (
        aggregate_live_run,
        describe,
        parse_token_ids,
        read_jsonl,
        write_live_artifacts,
    )


def _parse_sidecars(values: list[str]) -> dict[str, Path]:
    sidecars = {}
    for value in values:
        mode, separator, raw_path = value.partition("=")
        mode = mode.strip()
        if not separator or not mode or not raw_path.strip() or mode in sidecars:
            raise ValueError(
                "priority sidecars must be unique non-empty MODE=PATH values"
            )
        sidecars[mode] = Path(raw_path.strip())
    if len(sidecars) < 2:
        raise ValueError("scheduling aggregation requires at least two modes")
    return sidecars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--scheduler-stats-jsonl", type=Path, required=True)
    parser.add_argument("--attention-stats-jsonl", type=Path, required=True)
    parser.add_argument("--link-stats-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--priority-sidecar",
        action="append",
        required=True,
        metavar="MODE=PATH",
    )
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--arms", default="fixed_s1,continuous")
    parser.add_argument("--fixed-output-tokens", type=int, default=32)
    parser.add_argument("--protected-prefix-tokens", type=int, default=256)
    parser.add_argument("--protected-suffix-tokens", type=int, default=512)
    parser.add_argument("--priority-chunk-tokens", type=int, default=256)
    parser.add_argument("--stop-token-ids", type=parse_token_ids, default=())
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_scheduling_run(
    *,
    results: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    scheduler_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
    link_rows: list[dict[str, Any]],
    priorities: dict[str, list[dict[str, Any]]],
    arms: tuple[str, ...],
    expected_requests: int,
    fixed_output_tokens: int,
    protected_prefix_tokens: int,
    protected_suffix_tokens: int,
    priority_chunk_tokens: int,
    bootstrap_samples: int,
    seed: int,
    stop_token_ids: tuple[int, ...] = (),
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    modes = tuple(priorities)
    expected_records = expected_requests * len(modes) * len(arms)
    matrix: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    raw_errors = []
    for row in results:
        mode = row.get("schedule_mode")
        arm = row.get("arm")
        request_index = row.get("request_index")
        if mode not in priorities or arm not in arms:
            raw_errors.append(f"unexpected scheduling condition {mode}/{arm}")
            continue
        if not isinstance(request_index, int) or request_index < 0:
            raw_errors.append("invalid scheduling request index")
            continue
        condition = (str(mode), str(arm))
        request = matrix.setdefault(request_index, {})
        if condition in request:
            raw_errors.append(f"duplicate scheduling condition {request_index}/{condition}")
        request[condition] = row
    expected_conditions = {(mode, arm) for mode in modes for arm in arms}
    complete = (
        len(results) == expected_records
        and set(matrix) == set(range(expected_requests))
        and all(set(request) == expected_conditions for request in matrix.values())
    )

    shared_prefill_errors = []
    observed_prefill_groups: set[str] = set()
    for request_index in sorted(matrix):
        request = matrix[request_index]
        prefill_groups = {row.get("prefill_group") for row in request.values()}
        seeds = {row.get("seed_token_id") for row in request.values()}
        fresh_prefills = sum(
            row.get("prefill_reused") is False for row in request.values()
        )
        if len(prefill_groups) != 1:
            shared_prefill_errors.append(
                f"request {request_index}: conditions do not share one prefill group"
            )
        else:
            prefill_group = next(iter(prefill_groups))
            if not isinstance(prefill_group, str) or not prefill_group:
                shared_prefill_errors.append(
                    f"request {request_index}: shared prefill group is invalid"
                )
            elif prefill_group in observed_prefill_groups:
                shared_prefill_errors.append(
                    f"request {request_index}: prefill group reused across requests"
                )
            else:
                observed_prefill_groups.add(prefill_group)
        if len(seeds) != 1:
            shared_prefill_errors.append(
                f"request {request_index}: conditions do not share one producer seed"
            )
        if fresh_prefills != 1:
            shared_prefill_errors.append(
                f"request {request_index}: expected one producer prefill, got "
                f"{fresh_prefills}"
            )

    mode_summaries = {}
    paired_by_mode = {}
    for mode_index, mode in enumerate(modes):
        mode_results = [
            row for row in results if row.get("schedule_mode") == mode
        ]
        summary, paired = aggregate_live_run(
            mode_results,
            metadata_rows,
            scheduler_rows,
            attention_rows,
            link_rows,
            arms=arms,
            expected_requests=expected_requests,
            fixed_output_tokens=fixed_output_tokens,
            bootstrap_samples=bootstrap_samples,
            seed=seed + mode_index * 1000,
            protected_prefix_tokens=protected_prefix_tokens,
            protected_suffix_tokens=protected_suffix_tokens,
            priority_rows=priorities[mode],
            priority_chunk_tokens=priority_chunk_tokens,
            enforce_paired_prefill=False,
            stop_token_ids=stop_token_ids,
        )
        mode_summaries[mode] = summary
        paired_by_mode[mode] = paired

    equality_errors = []
    if complete:
        for request_index, request in matrix.items():
            reference_condition = next(iter(expected_conditions))
            reference = request[reference_condition].get("token_ids")
            for condition in expected_conditions:
                if request[condition].get("token_ids") != reference:
                    equality_errors.append(
                        f"request {request_index}: output differs at {condition}"
                    )

    comparisons = {}
    primary_mode = "bm25" if "bm25" in modes else modes[0]
    primary_pairs = {
        int(row["request_index"]): row for row in paired_by_mode[primary_mode]
    }
    comparison_arm = "continuous" if "continuous" in arms else arms[-1]
    for mode_index, mode in enumerate(modes):
        if mode == primary_mode:
            continue
        other_pairs = {
            int(row["request_index"]): row for row in paired_by_mode[mode]
        }
        shared = sorted(set(primary_pairs) & set(other_pairs))
        completion_gain = [
            other_pairs[index]["arms"][comparison_arm]["decode_completion_ms"]
            - primary_pairs[index]["arms"][comparison_arm]["decode_completion_ms"]
            for index in shared
        ]
        accepted_gain = [
            primary_pairs[index]["arms"][comparison_arm]["scheduler"][
                "useful_accepted_tokens"
            ]
            - other_pairs[index]["arms"][comparison_arm]["scheduler"][
                "useful_accepted_tokens"
            ]
            for index in shared
        ]
        raw_accepted_gain = [
            primary_pairs[index]["arms"][comparison_arm]["scheduler"][
                "accepted_tokens"
            ]
            - other_pairs[index]["arms"][comparison_arm]["scheduler"][
                "accepted_tokens"
            ]
            for index in shared
        ]
        comparisons[f"{mode}_vs_{primary_mode}"] = {
            "requests": len(shared),
            "arm": comparison_arm,
            "completion_gain_definition": (
                f"{mode} minus {primary_mode}; positive favors {primary_mode}"
            ),
            "accepted_gain_definition": (
                f"useful {primary_mode} minus {mode} before first stop token; "
                f"positive favors {primary_mode}"
            ),
            "decode_completion_gain_ms": describe(
                completion_gain,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 10_000 + mode_index * 2,
            ),
            "accepted_token_gain": describe(
                accepted_gain,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 10_001 + mode_index * 2,
            ),
            "raw_fixed_horizon_accepted_token_gain": describe(
                raw_accepted_gain,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 20_001 + mode_index * 2,
            ),
        }

    per_mode_passed = all(
        summary["gates"]["overall"]["passed"]
        for summary in mode_summaries.values()
    )
    gates = {
        "complete_paired_schedule_matrix": {
            "passed": complete and not raw_errors,
            "expected_records": expected_records,
            "observed_records": len(results),
            "errors": raw_errors,
        },
        "all_per_mode_runtime_gates": {
            "passed": per_mode_passed,
            "failed_modes": [
                mode
                for mode, summary in mode_summaries.items()
                if not summary["gates"]["overall"]["passed"]
            ],
        },
        "single_shared_prefill_across_schedules": {
            "passed": complete and not shared_prefill_errors,
            "expected_fresh_prefills": expected_requests,
            "observed_prefill_groups": len(observed_prefill_groups),
            "errors": shared_prefill_errors,
        },
        "cross_schedule_exact_greedy_equivalence": {
            "passed": complete and not equality_errors,
            "errors": equality_errors,
        },
    }
    gates["overall"] = {
        "passed": all(gate["passed"] for gate in gates.values())
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if gates["overall"]["passed"] else "failed_gates",
        "modes": list(modes),
        "arms": list(arms),
        "requests": expected_requests,
        "stop_token_ids": list(stop_token_ids),
        "primary_online_schedule": primary_mode,
        "oracle_role": "analysis upper bound" if "oracle" in modes else None,
        "gates": gates,
        "mode_summaries": mode_summaries,
        "comparisons": comparisons,
    }
    return summary, paired_by_mode, mode_summaries


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Progressive KV scheduling result",
        "",
        f"Gate status: **{summary['status']}**; requests: {summary['requests']}.",
        "",
        "| Schedule | Continuous effective accepted rate | Continuous completion p50 (ms) | Fixed-S1 vs continuous gain (ms) |",
        "|---|---:|---:|---:|",
    ]
    for mode, mode_summary in summary["mode_summaries"].items():
        arm = mode_summary["arm_summaries"]["continuous"]
        comparison = mode_summary["comparisons"]["fixed_s1_vs_continuous"]
        lines.append(
            f"| {mode} | {arm['draft']['effective_acceptance_rate']:.4f} | "
            f"{arm['decode_completion_ms']['p50']:.3f} | "
            f"{comparison['decode_completion_gain_ms']['mean']:.3f} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, gate in summary["gates"].items():
        lines.append(f"- {name}: {'PASS' if gate['passed'] else 'FAIL'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "schedule",
        "requests",
        "continuous_effective_acceptance_rate",
        "continuous_raw_acceptance_rate",
        "continuous_decode_completion_p50_ms",
        "fixed_s1_vs_continuous_gain_mean_ms",
        "runtime_gates_passed",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for mode, mode_summary in summary["mode_summaries"].items():
            arm = mode_summary["arm_summaries"]["continuous"]
            comparison = mode_summary["comparisons"]["fixed_s1_vs_continuous"]
            writer.writerow(
                {
                    "schedule": mode,
                    "requests": mode_summary["requests"],
                    "continuous_effective_acceptance_rate": arm["draft"][
                        "effective_acceptance_rate"
                    ],
                    "continuous_raw_acceptance_rate": arm["draft"][
                        "acceptance_rate"
                    ],
                    "continuous_decode_completion_p50_ms": arm[
                        "decode_completion_ms"
                    ]["p50"],
                    "fixed_s1_vs_continuous_gain_mean_ms": comparison[
                        "decode_completion_gain_ms"
                    ]["mean"],
                    "runtime_gates_passed": mode_summary["gates"]["overall"][
                        "passed"
                    ],
                }
            )


def main() -> None:
    args = parse_args()
    sidecars = _parse_sidecars(args.priority_sidecar)
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    priorities = {mode: read_jsonl(path) for mode, path in sidecars.items()}
    summary, paired_by_mode, mode_summaries = aggregate_scheduling_run(
        results=read_jsonl(args.results_jsonl),
        metadata_rows=read_jsonl(args.metadata_jsonl),
        scheduler_rows=read_jsonl(args.scheduler_stats_jsonl),
        attention_rows=read_jsonl(args.attention_stats_jsonl),
        link_rows=read_jsonl(args.link_stats_jsonl),
        priorities=priorities,
        arms=arms,
        expected_requests=args.expected_requests,
        fixed_output_tokens=args.fixed_output_tokens,
        protected_prefix_tokens=args.protected_prefix_tokens,
        protected_suffix_tokens=args.protected_suffix_tokens,
        priority_chunk_tokens=args.priority_chunk_tokens,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        stop_token_ids=args.stop_token_ids,
    )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    for mode in sidecars:
        write_live_artifacts(
            args.output_dir / mode,
            mode_summaries[mode],
            paired_by_mode[mode],
        )
    summary["priority_sidecars"] = {
        mode: {"path": str(path), "sha256": _sha256(path)}
        for mode, path in sidecars.items()
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output_dir / "summary.md", summary)
    _write_csv(args.output_dir / "paper_table.csv", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["gates"]["overall"]["passed"]:
        raise SystemExit("one or more scheduling validity gates failed")


if __name__ == "__main__":
    main()
