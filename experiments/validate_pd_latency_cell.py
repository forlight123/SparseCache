# SPDX-License-Identifier: Apache-2.0
"""Fail closed when a P/D latency cell violates the paper protocol."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def pipeline_unattributed_lower_bound(item):
    accounted_upper_bound = (
        item["draft_prefill_ms"]
        + item["draft_decode_ms"]
        + item["final_prefill_ms"]
        + item["final_correction_ms"]
        + item["final_decode_ms"]
        + item["verify_ms"]
        + item["transfer"]["total_h2d_ms"]
        + sum(item["transfer"].get("stage_wire_wait_ms", []))
    )
    return max(0.0, item["response_ms"] - accounted_upper_bound)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--max-unattributed-ms", type=float, default=100.0)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.inputs)
    failures = []
    if len(rows) != args.expected_count:
        failures.append(
            f"expected {args.expected_count} rows, observed {len(rows)}"
        )
    indices = [row["dataset_index"] for row in rows]
    if len(indices) != len(set(indices)):
        failures.append("dataset indices are not unique")
    if not rows:
        raise ValueError("no rows")
    protocol = rows[0]["protocol"]
    for key in (
        "fixed_token_horizon",
        "paired_measurement",
        "require_exclusive_gpu",
        "eager_serial_wire",
    ):
        if not protocol.get(key, False):
            failures.append(f"protocol field {key} is not true")
    if protocol.get("initial_gpu_used_bytes", 2**63) > 1 * 2**30:
        failures.append("selected GPU was not exclusive at process start")
    if (
        protocol.get("dataset_format") in {"longbench", "longbench_v2"}
        and protocol.get("longbench_prompt_mode") != "official_chat"
    ):
        failures.append("paper LongBench cell did not use official_chat prompts")

    timing_mismatches = []
    noisy = []
    orders = Counter()
    for row in rows:
        for schedule_name, schedule in row["schedules"].items():
            target = schedule["full_target"]
            for window, chain in schedule["chains"].items():
                key = (row["dataset_index"], schedule_name, window)
                pipeline = chain["pipeline"]
                orders[chain.get("measurement_order", "missing")] += 1
                if len(target["timing_token_ids"]) != len(
                    chain["timing_token_ids"]
                ):
                    timing_mismatches.append(key)
                unattributed = pipeline_unattributed_lower_bound(pipeline)
                if unattributed > args.max_unattributed_ms:
                    noisy.append((*key, unattributed))
    if timing_mismatches:
        failures.append(
            f"{len(timing_mismatches)} baseline/method timing-length mismatches"
        )
    if noisy:
        failures.append(
            f"{len(noisy)} requests exceed the unattributed-wall threshold"
        )
    first = orders["full_target_then_pipeline"]
    second = orders["pipeline_then_full_target"]
    if abs(first - second) > 1:
        failures.append(f"measurement order is unbalanced: {dict(orders)}")

    payload = {
        "passed": not failures,
        "rows": len(rows),
        "failures": failures,
        "measurement_order_counts": dict(orders),
        "timing_length_mismatches": timing_mismatches,
        "unattributed_over_threshold": noisy,
        "max_unattributed_ms": args.max_unattributed_ms,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
