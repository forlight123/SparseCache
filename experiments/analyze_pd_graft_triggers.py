# SPDX-License-Identifier: Apache-2.0
"""Analyze online trigger candidates from an in-process fixed/graft run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_pd_progressive_kv_pipeline import paired_delta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-threshold", type=float, default=6.0)
    return parser.parse_args()


def final_accepted(run):
    verified = [stage for stage in run["trace"] if stage.get("verified")]
    if not verified:
        raise ValueError("run has no verifier stage")
    return verified[-1]["accepted_pending"]


def proposal_ids(stage):
    if "draft_token_ids" not in stage:
        raise ValueError("input predates draft_token_ids instrumentation")
    return stage["draft_token_ids"]


def summarize_trigger(rows, key):
    triggered = [row for row in rows if row[key]]
    policy_deltas = [row["latency_saved_ms"] if row[key] else 0.0 for row in rows]
    return {
        "triggered": len(triggered),
        "true_positive": sum(row["acceptance_gain"] > 0 for row in triggered),
        "false_positive": sum(row["acceptance_gain"] == 0 for row in triggered),
        "missed_positive": sum(
            row["acceptance_gain"] > 0 and not row[key] for row in rows
        ),
        "latency_positive_when_triggered": sum(
            row["latency_saved_ms"] > 0 for row in triggered
        ),
        "idealized_policy_latency_saved_ms": paired_delta(
            policy_deltas, [0.0] * len(policy_deltas)
        ),
        "probe_cost_charged": False,
    }


def main():
    args = parse_args()
    source_rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for source in source_rows:
        schedule_name = source["protocol"]["schedules"][0]
        window_name = str(source["protocol"]["commit_windows"][0])
        chain = source["schedules"][schedule_name]["chains"][window_name]
        graft = chain["pipeline"]
        fixed = chain.get("paired_fixed_control")
        if fixed is None:
            raise ValueError("input does not contain paired_fixed_control")
        fixed_tokens = proposal_ids(fixed["trace"][0])
        graft_stages = [
            proposal_ids(stage)
            for stage in graft["trace"]
            if stage.get("drafted", 0) > 0
        ]
        if len(graft_stages) < 2 or len(fixed_tokens) < 4:
            raise ValueError("stability analysis requires two graft stages and four fixed drafts")
        first_margins = graft["trace"][0].get("draft_top1_margins")
        if not first_margins:
            raise ValueError("input predates draft margin instrumentation")
        prefix_changed = graft_stages[0] != fixed_tokens[:len(graft_stages[0])]
        s2_width = len(graft_stages[1])
        s2_start = len(graft_stages[0])
        rows.append(
            {
                "dataset_index": source["dataset_index"],
                "acceptance_gain": final_accepted(graft) - final_accepted(fixed),
                "latency_saved_ms": fixed["response_ms"] - graft["response_ms"],
                "margin_trigger": min(first_margins) >= args.margin_threshold,
                "s2_first_token_instability": (
                    prefix_changed
                    or graft_stages[1][0] != fixed_tokens[s2_start]
                ),
                "s2_batch_instability": (
                    prefix_changed
                    or graft_stages[1]
                    != fixed_tokens[s2_start:s2_start + s2_width]
                ),
            }
        )
    payload = {
        "source": str(Path(args.input).resolve()),
        "cases": len(rows),
        "margin_threshold": args.margin_threshold,
        "unconditional_graft": {
            "latency_saved_ms": paired_delta(
                (row["latency_saved_ms"] for row in rows),
                [0.0] * len(rows),
            ),
            "acceptance_gain": paired_delta(
                (row["acceptance_gain"] for row in rows),
                [0.0] * len(rows),
            ),
        },
        "triggers": {
            key: summarize_trigger(rows, key)
            for key in (
                "margin_trigger",
                "s2_first_token_instability",
                "s2_batch_instability",
            )
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
