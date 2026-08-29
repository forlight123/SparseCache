# SPDX-License-Identifier: Apache-2.0
"""Aggregate progressive-KV feasibility JSONL shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def ci95(values):
    if len(values) < 2:
        return None
    return 1.96 * statistics.stdev(values) / len(values) ** 0.5


def bootstrap_mean_ci(values, *, samples=10000, seed=20260825):
    if not values:
        return None
    rng = random.Random(seed)
    estimates = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]]


def paired_diagnostics(left, right):
    deltas = [a - b for a, b in zip(left, right)]
    tolerance = 1e-12
    return {
        "mean_delta": sum(deltas) / len(deltas),
        "mean_delta_bootstrap_ci95": bootstrap_mean_ci(deltas),
        "wins": sum(delta > tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "losses": sum(delta < -tolerance for delta in deltas),
    }


def idealized_timing(rows, schedule, window):
    results = {}
    for bandwidth in (1, 3, 6, 12, 25, 50):
        baseline_latencies = []
        progressive_latencies = []
        for row in rows:
            size_gib = row["logical_bf16_kv_bytes"] / 2**30
            chain = row["schedules"][schedule]["chains"][window]
            levels = row["schedules"][schedule]["levels"]
            first_fraction = levels[0]["actual_document_fraction"]
            required_fraction = levels[int(chain["terminated_stage"])][
                "actual_document_fraction"
            ]
            chain_compute = chain["draft_and_refresh_ms"] + chain["verify_ms"]
            baseline_compute = (
                row["fullreuse"]["prefill_ms"] + row["fullreuse"]["decode_ms"]
            )
            baseline_latencies.append(size_gib / bandwidth * 1000 + baseline_compute)
            first_transfer = size_gib * first_fraction / bandwidth * 1000
            remaining_transfer = (
                size_gib * (required_fraction - first_fraction) / bandwidth * 1000
            )
            # This is intentionally optimistic: all post-anchor transfer and
            # GPU work are assumed to overlap perfectly.
            progressive_latencies.append(
                first_transfer + max(remaining_transfer, chain_compute)
            )
        baseline = sum(baseline_latencies) / len(baseline_latencies)
        progressive = sum(progressive_latencies) / len(progressive_latencies)
        results[str(bandwidth)] = {
            "baseline_ms": baseline,
            "optimistic_progressive_ms": progressive,
            "speedup": baseline / progressive,
        }
    return results


def summarize(rows):
    base = {}
    for name in ("fullprefill", "fullcontext_cache", "fullreuse"):
        base[name] = {
            metric: mean([row[name] for row in rows], metric)
            for metric in ("em", "f1", "prefill_ms", "decode_ms")
        }
    base["fullreuse"]["raw_match_fullprefill"] = mean(
        [row["fullreuse"] for row in rows], "raw_match_fullprefill"
    )
    base["fullreuse"]["normalized_match_fullprefill"] = mean(
        [row["fullreuse"] for row in rows], "normalized_match_fullprefill"
    )
    base["fullreuse"]["raw_match_fullcontext_cache"] = mean(
        [row["fullreuse"] for row in rows], "raw_match_fullcontext_cache"
    )
    base["fullreuse"]["normalized_match_fullcontext_cache"] = mean(
        [row["fullreuse"] for row in rows], "normalized_match_fullcontext_cache"
    )
    base["fullreuse"]["teacher_self_top1_agreement"] = mean(
        [row["fullreuse"] for row in rows], "teacher_self_top1_agreement"
    )
    base["fullcontext_cache"]["token_match_one_shot_fullprefill"] = mean(
        [row["fullcontext_cache"] for row in rows],
        "token_match_one_shot_fullprefill",
    )
    base["fullcontext_cache"]["normalized_match_one_shot_fullprefill"] = mean(
        [row["fullcontext_cache"] for row in rows],
        "normalized_match_one_shot_fullprefill",
    )
    base["fullcontext_cache"]["f1_vs_one_shot_fullprefill"] = paired_diagnostics(
        [row["fullcontext_cache"]["f1"] for row in rows],
        [row["fullprefill"]["f1"] for row in rows],
    )
    base["fullreuse"]["f1_vs_fullcontext_cache"] = paired_diagnostics(
        [row["fullreuse"]["f1"] for row in rows],
        [row["fullcontext_cache"]["f1"] for row in rows],
    )
    fullcontext_correct = [row for row in rows if row["fullcontext_cache"]["em"] == 1]
    base["fullreuse"]["fullcontext_correct_to_fullreuse_wrong"] = {
        "count": sum(row["fullreuse"]["em"] == 0 for row in fullcontext_correct),
        "denominator": len(fullcontext_correct),
        "rate": mean(
            [{"value": row["fullreuse"]["em"] == 0} for row in fullcontext_correct],
            "value",
        ),
    }

    schedule_names = rows[0]["schedules"].keys()
    schedules = {}
    for schedule in schedule_names:
        level_count = len(rows[0]["schedules"][schedule]["levels"])
        levels = []
        for level_index in range(level_count):
            values = [row["schedules"][schedule]["levels"][level_index] for row in rows]
            levels.append(
                {
                    key: mean(values, key)
                    for key in (
                        "configured_fraction",
                        "actual_document_fraction",
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
            chain_summary = {
                key: mean(values, key)
                for key in (
                    "em",
                    "f1",
                    "token_match_fullreuse",
                    "raw_match_fullreuse",
                    "normalized_match_fullreuse",
                    "draft_and_refresh_ms",
                    "verify_ms",
                    "ignored_committed_flip_rate",
                    "terminated_stage",
                )
            }
            f1_values = [item["f1"] for item in values]
            chain_summary["f1_ci95"] = ci95(f1_values)
            chain_summary["f1_vs_fullreuse"] = paired_diagnostics(
                f1_values, [row["fullreuse"]["f1"] for row in rows]
            )
            chain_summary["f1_vs_fullcontext_cache"] = paired_diagnostics(
                f1_values, [row["fullcontext_cache"]["f1"] for row in rows]
            )
            chain_summary["fullreuse_correct_to_chain_wrong"] = sum(
                row["fullreuse"]["em"] == 1 and item["em"] == 0
                for row, item in zip(rows, values)
            )
            chain_summary["fullreuse_wrong_to_chain_correct"] = sum(
                row["fullreuse"]["em"] == 0 and item["em"] == 1
                for row, item in zip(rows, values)
            )
            chain_summary["mean_required_document_fraction"] = mean(
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
            chain_summary["optimistic_timing_by_bandwidth_gib_s"] = idealized_timing(
                rows, schedule, window
            )
            chains[window] = chain_summary
        schedules[schedule] = {"levels": levels, "chains": chains}

    by_hop = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row["hop_count"]].append(row)
    for hop, hop_rows in sorted(groups.items()):
        by_hop[str(hop)] = {
            "cases": len(hop_rows),
            "fullprefill_f1": mean([row["fullprefill"] for row in hop_rows], "f1"),
            "fullreuse_f1": mean([row["fullreuse"] for row in hop_rows], "f1"),
            "chains": {
                schedule: {
                    window: mean(
                        [row["schedules"][schedule]["chains"][window] for row in hop_rows],
                        "f1",
                    )
                    for window in hop_rows[0]["schedules"][schedule]["chains"]
                }
                for schedule in hop_rows[0]["schedules"]
            },
        }
    return {
        "cases": len(rows),
        "hop_counts": dict(sorted((str(key), len(value)) for key, value in groups.items())),
        "mean_document_tokens": mean(rows, "document_tokens"),
        "mean_supporting_document_tokens": mean(rows, "supporting_document_tokens"),
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
        raise ValueError("no result rows found")
    dataset_indices = [row["dataset_index"] for row in rows]
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError("duplicate dataset indices across shards")
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
