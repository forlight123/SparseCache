# SPDX-License-Identifier: Apache-2.0
"""Aggregate the real pinned-memory progressive P/D pipeline experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_mean_ci(values, seed=20260825):
    values = list(values)
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(10_000)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def paired_delta(left, right):
    values = [float(a) - float(b) for a, b in zip(left, right)]
    return {
        **describe(values),
        "mean_95pct_bootstrap_ci": bootstrap_mean_ci(values),
        "positive_count": sum(value > 0 for value in values),
    }


def accepted_draft_tokens(chain):
    """Count proposals accepted by verifier stages in one pipeline trace."""

    return sum(
        int(stage.get("accepted_pending", 0))
        for stage in chain["pipeline"]["trace"]
        if stage.get("verified")
    )


def summarize(rows):
    result = {
        "cases": len(rows),
        "prompt_tokens": describe(row["prompt_tokens"] for row in rows),
        "anchor_tokens": describe(row["anchor_tokens"] for row in rows),
        "logical_kv_mib": describe(
            row["logical_bf16_kv_bytes"] / 2**20 for row in rows
        ),
        "producer_d2h_to_pinned_cpu_ms": describe(
            row["producer"]["d2h_to_pinned_cpu_ms"] for row in rows
        ),
        "schedules": {},
    }
    for schedule in rows[0]["schedules"]:
        schedule_rows = [row["schedules"][schedule] for row in rows]
        targets = [item["full_target"] for item in schedule_rows]
        schedule_result = {
            "full_transfer_target": {
                "response_ms": describe(item["response_ms"] for item in targets),
                "h2d_ms": describe(
                    item["transfer"]["total_h2d_ms"] for item in targets
                ),
                "em": statistics.fmean(item["em"] for item in targets),
                "f1": statistics.fmean(item["f1"] for item in targets),
            },
            "chains": {},
        }
        for window in schedule_rows[0]["chains"]:
            chains = [item["chains"][window] for item in schedule_rows]
            serial = [item["serial"] for item in chains]
            pipeline = [item["pipeline"] for item in chains]
            has_serial = all(item is not None for item in serial)
            serial_ms = [item["response_ms"] for item in serial] if has_serial else None
            pipeline_ms = [item["response_ms"] for item in pipeline]
            target_ms = [item["response_ms"] for item in targets]
            target_f1 = [item["f1"] for item in targets]
            chain_f1 = [item["f1"] for item in chains]
            configured_drafts = [item["configured_draft_tokens"] for item in chains]
            accepted_drafts = [accepted_draft_tokens(item) for item in chains]
            injectable_suffixes = [max(0, accepted - 1) for accepted in accepted_drafts]
            schedule_result["chains"][window] = {
                "quality": {
                    "em": statistics.fmean(item["em"] for item in chains),
                    "f1": statistics.fmean(chain_f1),
                    "f1_delta_vs_full_target": paired_delta(chain_f1, target_f1),
                    "token_match_target": statistics.fmean(
                        item["token_match_target"] for item in chains
                    ),
                    "normalized_match_target": statistics.fmean(
                        item["normalized_match_target"] for item in chains
                    ),
                    "serial_pipeline_token_match": (
                        statistics.fmean(
                            item["serial_pipeline_token_match"] for item in chains
                        )
                        if has_serial
                        else None
                    ),
                },
                "serial_response_ms": (describe(serial_ms) if has_serial else None),
                "pipeline_response_ms": describe(pipeline_ms),
                "pipeline_latency_saved_vs_serial_ms": (
                    paired_delta(serial_ms, pipeline_ms) if has_serial else None
                ),
                "pipeline_latency_saved_vs_full_target_ms": paired_delta(
                    target_ms, pipeline_ms
                ),
                "pipeline_speedup_vs_serial": (
                    describe(a / b for a, b in zip(serial_ms, pipeline_ms))
                    if has_serial
                    else None
                ),
                "pipeline_speedup_vs_full_target": describe(
                    a / b for a, b in zip(target_ms, pipeline_ms)
                ),
                "draft_acceptance": {
                    "configured_tokens": describe(configured_drafts),
                    "accepted_tokens": {
                        **describe(accepted_drafts),
                        "mean_95pct_bootstrap_ci": bootstrap_mean_ci(accepted_drafts),
                    },
                    "runtime_suffix_after_one_token_alignment": {
                        **describe(injectable_suffixes),
                        "mean_95pct_bootstrap_ci": bootstrap_mean_ci(
                            injectable_suffixes
                        ),
                    },
                    "accepted_fraction": describe(
                        accepted / configured
                        for accepted, configured in zip(
                            accepted_drafts, configured_drafts
                        )
                    ),
                    "zero_acceptance_rate": statistics.fmean(
                        accepted == 0 for accepted in accepted_drafts
                    ),
                    "full_acceptance_rate": statistics.fmean(
                        accepted == configured
                        for accepted, configured in zip(
                            accepted_drafts, configured_drafts
                        )
                    ),
                },
                "pipeline_transfer": {
                    "launched_stages": describe(
                        item["transfer"]["launched_stages"] for item in pipeline
                    ),
                    "total_h2d_ms": describe(
                        item["transfer"]["total_h2d_ms"] for item in pipeline
                    ),
                    "first_stage_h2d_ms": describe(
                        item["transfer"]["first_stage_h2d_ms"] for item in pipeline
                    ),
                    "transferred_kv_fraction": describe(
                        item["transfer"]["total_bytes"] / row["logical_bf16_kv_bytes"]
                        for item, row in zip(pipeline, rows)
                    ),
                },
                "pipeline_compute": {
                    "first_draft_batch_ms": describe(
                        item["first_draft_batch_ms"] for item in pipeline
                    ),
                    "draft_prefill_ms": describe(
                        item["draft_prefill_ms"] for item in pipeline
                    ),
                    "draft_decode_ms": describe(
                        item["draft_decode_ms"] for item in pipeline
                    ),
                    "final_correction_ms": describe(
                        item["final_correction_ms"] for item in pipeline
                    ),
                    "final_decode_ms": describe(
                        item["final_decode_ms"] for item in pipeline
                    ),
                    "draft_and_refresh_ms": describe(
                        item["draft_and_refresh_ms"] for item in pipeline
                    ),
                    "verify_ms": describe(item["verify_ms"] for item in pipeline),
                },
            }
        result["schedules"][schedule] = schedule_result
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for name in args.inputs:
        with Path(name).open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError("no input rows")
    indices = [row["dataset_index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate dataset indices")
    payload = {
        "protocol": rows[0]["protocol"],
        "source_files": [str(Path(name).resolve()) for name in args.inputs],
        "summary": summarize(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
