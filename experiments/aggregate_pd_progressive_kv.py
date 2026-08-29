# SPDX-License-Identifier: Apache-2.0
"""Aggregate P/D progressive-transfer feasibility shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from aggregate_progressive_kv import bootstrap_mean_ci, mean, paired_diagnostics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def idealized_timing(rows, schedule, window):
    result = {}
    for bandwidth in (1, 3, 6, 12, 25, 50):
        baseline_values = []
        progressive_values = []
        for row in rows:
            size_gib = row["logical_bf16_kv_bytes"] / 2**30
            target_compute = (
                row["pd_target"]["seed_forward_ms"]
                + row["pd_target"]["decode_ms"]
            )
            chain = row["schedules"][schedule]["chains"][window]
            levels = row["schedules"][schedule]["levels"]
            first_fraction = levels[0]["actual_total_kv_fraction"]
            required_fraction = levels[int(chain["terminated_stage"])][
                "actual_total_kv_fraction"
            ]
            chain_compute = chain["draft_and_refresh_ms"] + chain["verify_ms"]
            baseline_values.append(size_gib / bandwidth * 1000 + target_compute)
            first_transfer = size_gib * first_fraction / bandwidth * 1000
            remaining_transfer = (
                size_gib * (required_fraction - first_fraction) / bandwidth * 1000
            )
            progressive_values.append(
                first_transfer + max(remaining_transfer, chain_compute)
            )
        baseline = sum(baseline_values) / len(baseline_values)
        progressive = sum(progressive_values) / len(progressive_values)
        result[str(bandwidth)] = {
            "baseline_ms": baseline,
            "optimistic_progressive_ms": progressive,
            "speedup": baseline / progressive,
        }
    return result


def summarize(rows):
    base = {
        "prefill_side": {
            key: mean([row["prefill_side"] for row in rows], key)
            for key in ("em", "f1", "prefill_ms", "decode_ms", "cache_producer_ms")
        },
        "pd_target": {
            key: mean([row["pd_target"] for row in rows], key)
            for key in (
                "em",
                "f1",
                "seed_forward_ms",
                "decode_ms",
                "token_match_prefill_side",
                "normalized_match_prefill_side",
                "teacher_self_top1_agreement",
            )
        },
    }
    schedules = {}
    for schedule in rows[0]["schedules"]:
        levels = []
        for level_index in range(len(rows[0]["schedules"][schedule]["levels"])):
            values = [row["schedules"][schedule]["levels"][level_index] for row in rows]
            levels.append(
                {
                    key: mean(values, key)
                    for key in (
                        "configured_fraction",
                        "actual_document_fraction",
                        "actual_total_kv_fraction",
                        "supporting_token_coverage",
                        "adjacent_top1_agreement",
                        "final_top1_agreement",
                        "strict_final_survival",
                        "mean_target_logprob",
                        "mean_top1_margin",
                        "teacher_forward_ms",
                    )
                }
            )
        chains = {}
        for window in rows[0]["schedules"][schedule]["chains"]:
            values = [row["schedules"][schedule]["chains"][window] for row in rows]
            f1_values = [item["f1"] for item in values]
            target_f1 = [row["pd_target"]["f1"] for row in rows]
            chain = {
                key: mean(values, key)
                for key in (
                    "em",
                    "f1",
                    "token_match_pd_target",
                    "raw_match_pd_target",
                    "normalized_match_pd_target",
                    "draft_and_refresh_ms",
                    "verify_ms",
                    "ignored_committed_flip_rate",
                    "terminated_stage",
                )
            }
            if window == "inf":
                chain["pre_exact_replay_token_match"] = mean(
                    values, "pre_exact_replay_token_match"
                )
                chain["exact_replay_rate"] = mean(values, "exact_replay_required")
                chain["exact_replay_ms"] = mean(values, "exact_replay_ms")
                chain["lossless_gate_pass"] = mean(values, "lossless_gate_pass")
            chain["f1_vs_pd_target"] = paired_diagnostics(f1_values, target_f1)
            chain["mean_required_document_fraction"] = mean(
                [
                    {
                        "value": row["schedules"][schedule]["levels"][
                            int(item["terminated_stage"])
                        ]["actual_document_fraction"]
                    }
                    for row, item in zip(rows, values)
                ],
                "value",
            )
            chain["mean_required_total_kv_fraction"] = mean(
                [
                    {
                        "value": row["schedules"][schedule]["levels"][
                            int(item["terminated_stage"])
                        ]["actual_total_kv_fraction"]
                    }
                    for row, item in zip(rows, values)
                ],
                "value",
            )
            chain["optimistic_timing_by_bandwidth_gib_s"] = idealized_timing(
                rows, schedule, window
            )
            chains[window] = chain
        schedules[schedule] = {"levels": levels, "chains": chains}

    hop_groups = defaultdict(list)
    for row in rows:
        hop_groups[row["hop_count"]].append(row)
    by_hop = {
        str(hop): {
            "cases": len(group),
            "pd_target_f1": mean([row["pd_target"] for row in group], "f1"),
            "chains": {
                schedule: {
                    window: mean(
                        [row["schedules"][schedule]["chains"][window] for row in group],
                        "f1",
                    )
                    for window in group[0]["schedules"][schedule]["chains"]
                }
                for schedule in group[0]["schedules"]
            },
        }
        for hop, group in sorted(hop_groups.items())
    }
    return {
        "cases": len(rows),
        "hop_counts": {str(hop): len(group) for hop, group in sorted(hop_groups.items())},
        "mean_document_tokens": mean(rows, "document_tokens"),
        "mean_prompt_tokens": mean(rows, "prompt_tokens"),
        "mean_anchor_tokens": mean(rows, "anchor_tokens"),
        "mean_logical_bf16_kv_mib": mean(rows, "logical_bf16_kv_bytes") / 2**20,
        "base": base,
        "schedules": schedules,
        "by_hop": by_hop,
    }


def main():
    args = parse_args()
    rows = []
    for input_name in args.inputs:
        with Path(input_name).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("no rows")
    indices = [row["dataset_index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate dataset indices")
    payload = {
        "protocol": rows[0]["protocol"],
        "source_files": [str(Path(name).resolve()) for name in args.inputs],
        "summary": summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
