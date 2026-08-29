# SPDX-License-Identifier: Apache-2.0
"""Aggregate matched-bandwidth runs of the sparse-draft P/D pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from aggregate_pd_progressive_kv_pipeline import describe, paired_delta


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def load_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize_run(rows):
    schedule = rows[0]["protocol"]["schedules"][0]
    window = str(rows[0]["protocol"]["commit_windows"][0])
    schedule_rows = [row["schedules"][schedule] for row in rows]
    targets = [item["full_target"] for item in schedule_rows]
    chains = [item["chains"][window] for item in schedule_rows]
    pipelines = [item["pipeline"] for item in chains]
    target_ms = [item["response_ms"] for item in targets]
    pipeline_ms = [item["response_ms"] for item in pipelines]
    first_committed = [
        item["first_committed_ms"]
        for item in pipelines
        if item["first_committed_ms"] is not None
    ]
    terminated = {}
    for item in pipelines:
        stage = str(item["terminated_stage"] + 1)
        terminated[stage] = terminated.get(stage, 0) + 1
    protocol = rows[0]["protocol"]
    bandwidth = protocol["transport_gbps"]
    configured_drafts = [
        item.get("configured_draft_tokens", protocol["draft_tokens"])
        for item in chains
    ]
    actual_drafts = [
        sum(stage["drafted"] for stage in item["trace"])
        for item in pipelines
    ]
    # Intermediate verifiers repeatedly inspect the same tentative prefix.
    # Summing their accepted lengths double-counts proposals, so acceptance is
    # always defined at the last (immutable full-KV) verifier.
    accepted_drafts = []
    intermediate_accepted_checks = []
    for item in pipelines:
        verified = [stage for stage in item["trace"] if stage.get("verified")]
        accepted_drafts.append(
            verified[-1]["accepted_pending"] if verified else 0
        )
        intermediate_accepted_checks.append(
            sum(stage["accepted_pending"] for stage in verified[:-1])
        )
    sparse_draft_ms = [
        item["draft_prefill_ms"] + item["draft_decode_ms"]
        for item in pipelines
    ]
    residual_wire_ms = [
        sum(item["transfer"]["stage_wire_ms"][1:])
        for item in pipelines
    ]
    target_unattributed_ms = [
        max(
            0.0,
            target["response_ms"]
            - target["seed_forward_ms"]
            - target["decode_ms"]
            - target["transfer"]["total_h2d_ms"]
            - sum(target["transfer"].get("stage_wire_wait_ms", [])),
        )
        for target in targets
    ]
    pipeline_unattributed_ms = [
        max(
            0.0,
            item["response_ms"]
            - item["draft_prefill_ms"]
            - item["draft_decode_ms"]
            - item["final_prefill_ms"]
            - item["final_correction_ms"]
            - item["final_decode_ms"]
            - item["verify_ms"]
            - item["transfer"]["total_h2d_ms"]
            - sum(item["transfer"].get("stage_wire_wait_ms", [])),
        )
        for item in pipelines
    ]
    fixed_controls = [item.get("paired_fixed_control") for item in chains]
    continuous_visibility = None
    if all("draft_visibility_stages" in item for item in pipelines):
        per_request_mean_visible_fraction = []
        per_request_last_visible_fraction = []
        per_request_after_s1_fraction = []
        distinct_visibility_stages = []
        completion_poll_ms = []
        for item in pipelines:
            stage_bytes = item["transfer"]["stage_bytes"]
            cumulative = []
            running = 0
            for value in stage_bytes:
                running += value
                cumulative.append(running / item["transfer"]["total_bytes"])
            stages = item["draft_visibility_stages"]
            fractions = [cumulative[stage] for stage in stages]
            per_request_mean_visible_fraction.append(mean(fractions) or 0.0)
            per_request_last_visible_fraction.append(
                fractions[-1] if fractions else cumulative[0]
            )
            per_request_after_s1_fraction.append(
                mean(stage > 0 for stage in stages) or 0.0
            )
            distinct_visibility_stages.append(len(set(stages)))
            completion_poll_ms.append(item.get("completion_poll_ms", 0.0))
        continuous_visibility = {
            "mean_visible_kv_fraction_per_request": describe(
                per_request_mean_visible_fraction
            ),
            "last_proposal_visible_kv_fraction": describe(
                per_request_last_visible_fraction
            ),
            "proposal_fraction_after_s1": describe(
                per_request_after_s1_fraction
            ),
            "distinct_visibility_stages": describe(
                distinct_visibility_stages
            ),
            "completion_poll_ms": describe(completion_poll_ms),
        }
    paired_fixed = None
    if all(item is not None for item in fixed_controls):
        fixed_verified = [
            [stage for stage in item["trace"] if stage.get("verified")][-1]
            for item in fixed_controls
        ]
        fixed_actual_drafts = [
            sum(stage["drafted"] for stage in item["trace"])
            for item in fixed_controls
        ]
        paired_fixed = {
            "latency_saved_by_method_ms": paired_delta(
                (item["response_ms"] for item in fixed_controls),
                pipeline_ms,
            ),
            "accepted_token_delta_method_minus_fixed": paired_delta(
                accepted_drafts,
                (stage["accepted_pending"] for stage in fixed_verified),
            ),
            "proposal_token_delta_method_minus_fixed": paired_delta(
                actual_drafts,
                fixed_actual_drafts,
            ),
            "accepted_fraction_delta_method_minus_fixed": paired_delta(
                (
                    accepted / max(1, proposed)
                    for accepted, proposed in zip(
                        accepted_drafts, actual_drafts
                    )
                ),
                (
                    stage["accepted_pending"] / max(1, proposed)
                    for stage, proposed in zip(
                        fixed_verified, fixed_actual_drafts
                    )
                ),
            ),
            "task_score_delta_method_minus_fixed": paired_delta(
                (chain["task_score"] for chain in chains),
                (item["task_score"] for item in fixed_controls),
            ),
            "fixed_response_ms": describe(
                item["response_ms"] for item in fixed_controls
            ),
            "fixed_sparse_draft_ms": describe(
                item["draft_prefill_ms"] + item["draft_decode_ms"]
                for item in fixed_controls
            ),
        }
    return {
        "transport_gbps": bandwidth,
        "cases": len(rows),
        "latency": {
            "full_target_ms": describe(target_ms),
            "pipeline_ms": describe(pipeline_ms),
            "saved_vs_full_target_ms": paired_delta(target_ms, pipeline_ms),
            "speedup_ratio_of_means": mean(target_ms) / mean(pipeline_ms),
            "full_target_first_decode_token_ms": describe(
                target["response_ms"] - target["decode_ms"]
                for target in targets
            ),
            "first_draft_batch_ms": describe(
                item["first_draft_batch_ms"] for item in pipelines
            ),
            "first_committed_ms": (
                describe(first_committed) if first_committed else None
            ),
            "measurement_diagnostics": {
                "full_target_unattributed_wall_ms_lower_bound": describe(
                    target_unattributed_ms
                ),
                "pipeline_unattributed_wall_ms_lower_bound": describe(
                    pipeline_unattributed_ms
                ),
                "pipeline_over_100ms_count": sum(
                    value > 100.0 for value in pipeline_unattributed_ms
                ),
                "measurement_order_counts": {
                    name: sum(
                        chain.get("measurement_order", "legacy") == name
                        for chain in chains
                    )
                    for name in sorted(
                        {
                            chain.get("measurement_order", "legacy")
                            for chain in chains
                        }
                    )
                },
            },
        },
        "quality": {
            "task_metric": chains[0].get("task_metric", "qa_f1"),
            "token_match_target": mean(
                item["token_match_target"] for item in chains
            ),
            "normalized_match_target": mean(
                item["normalized_match_target"] for item in chains
            ),
            "timing_token_match_target": mean(
                item.get("timing_token_match_target", False)
                for item in chains
            ),
            "quality_output_length_match": mean(
                len(item["token_ids"]) == len(target["token_ids"])
                for item, target in zip(chains, targets)
            ),
            "timing_output_length_match": mean(
                len(item.get("timing_token_ids", item["token_ids"]))
                == len(target.get("timing_token_ids", target["token_ids"]))
                for item, target in zip(chains, targets)
            ),
            "em": mean(item["em"] for item in chains),
            "f1": mean(item["f1"] for item in chains),
            "full_target_f1": mean(item["f1"] for item in targets),
            "task_score_delta_vs_full_target": paired_delta(
                (item["f1"] for item in chains),
                (item["f1"] for item in targets),
            ),
        },
        "pipeline": {
            "draft": {
                "configured_tokens": describe(configured_drafts),
                "actual_tokens": describe(actual_drafts),
                "accepted_tokens": describe(accepted_drafts),
                "accepted_fraction": describe(
                    accepted / max(1, drafted)
                    for accepted, drafted in zip(accepted_drafts, actual_drafts)
                ),
                "intermediate_accepted_checks": describe(
                    intermediate_accepted_checks
                ),
                "continuous_visibility": continuous_visibility,
            },
            "critical_path_model": {
                "residual_wire_ms": describe(residual_wire_ms),
                "sparse_draft_ms": describe(sparse_draft_ms),
                "residual_minus_draft_ms": describe(
                    residual - draft
                    for residual, draft in zip(residual_wire_ms, sparse_draft_ms)
                ),
                "exposed_draft_ms": describe(
                    max(0.0, draft - residual)
                    for residual, draft in zip(residual_wire_ms, sparse_draft_ms)
                ),
            },
            "termination_stage_counts_1_based": terminated,
            "early_termination_rate": mean(
                item["terminated_stage"] < len(protocol["stage_fractions"]) - 1
                for item in pipelines
            ),
            "final_verify_state_reuse_rate": mean(
                item["reused_final_verify"] for item in pipelines
            ),
            "compute_ms": {
                name: describe(item[name] for item in pipelines)
                for name in (
                    "draft_prefill_ms",
                    "draft_decode_ms",
                    "final_prefill_ms",
                    "final_correction_ms",
                    "final_decode_ms",
                    "verify_ms",
                )
            },
            "transfer": {
                "actual_s1_kv_fraction": describe(
                    item["transfer"]["stage_bytes"][0]
                    / row["logical_bf16_kv_bytes"]
                    for item, row in zip(pipelines, rows)
                ),
                "h2d_kv_fraction": describe(
                    item["transfer"]["total_bytes"]
                    / row["logical_bf16_kv_bytes"]
                    for item, row in zip(pipelines, rows)
                ),
                "wire_kv_fraction_at_finish": describe(
                    item["transfer"]["wire_bytes_at_finish"]
                    / row["logical_bf16_kv_bytes"]
                    for item, row in zip(pipelines, rows)
                ),
                "total_h2d_ms": describe(
                    item["transfer"]["total_h2d_ms"] for item in pipelines
                ),
            },
        },
        "paired_fixed_control": paired_fixed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--concat-shards",
        action="store_true",
        help="concatenate disjoint dataset-index shards into one run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    groups = [load_rows(path) for path in args.inputs]
    if any(not rows for rows in groups):
        raise ValueError("all input runs must contain rows")
    if args.concat_shards:
        rows = [row for group in groups for row in group]
        indices = [row["dataset_index"] for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError("concatenated shards contain duplicate dataset indices")
        groups = [rows]
    reference_indices = [row["dataset_index"] for row in groups[0]]
    for rows in groups[1:]:
        if [row["dataset_index"] for row in rows] != reference_indices:
            raise ValueError("bandwidth runs are not paired on dataset index/order")
    runs = [summarize_run(rows) for rows in groups]
    runs.sort(
        key=lambda item: (
            float("inf") if item["transport_gbps"] is None
            else item["transport_gbps"]
        )
    )
    payload = {
        "protocol": {
            **groups[0][0]["protocol"],
            "transport_gbps": [run["transport_gbps"] for run in runs],
            "transport_pacing": (
                "native control plus serial per-tranche wire-arrival deadlines "
                "followed by real H2D"
            ),
        },
        "source_files": [str(Path(path).resolve()) for path in args.inputs],
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
