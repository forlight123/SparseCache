# SPDX-License-Identifier: Apache-2.0
"""Aggregate the independent-document CacheBlend/progressive reuse experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from aggregate_pd_progressive_kv_pipeline import describe, paired_delta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError("no experiment rows found")
    indices = [row["dataset_index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate dataset indices across input shards")
    return rows


def fraction(rows, predicate):
    return sum(bool(predicate(row)) for row in rows) / len(rows)


def quality(rows, key):
    return {
        "em": statistics.fmean(row[key]["em"] for row in rows),
        "f1": statistics.fmean(row[key]["f1"] for row in rows),
    }


def summarize(rows):
    stores = {}
    for row in rows:
        protocol = row["protocol"]
        shard = (protocol["shard_offset"], protocol["shard_count"])
        stores.setdefault(shard, row["producer"])
    store_rows = list(stores.values())
    full_response = [row["full_prefill"]["response_ms"] for row in rows]
    full_ttft = [row["full_prefill"]["prefill_ms"] for row in rows]
    cacheblend_response = [row["cacheblend_15"]["response_ms"] for row in rows]
    cacheblend_ttft = [
        row["cacheblend_15"].get(
            "first_token_ready_ms",
            row["cacheblend_15"]["response_ms"]
            - row["cacheblend_15"]["decode_ms"],
        )
        for row in rows
    ]
    progressive_response = [
        row["sparsecache_progressive"]["response_ms"] for row in rows
    ]
    serial_response = [
        row["sparsecache_progressive"]["serial_response_ms"] for row in rows
    ]
    serial_matched_rows = [
        row
        for row in rows
        if row["sparsecache_progressive"].get(
            "serial_pipeline_token_match", False
        )
    ]

    stages = defaultdict(list)
    for row in rows:
        for item in row["sparsecache_progressive"]["trace"]:
            stages[item["stage"]].append(item)
    stage_summary = {}
    for stage_index, items in sorted(stages.items()):
        pending = sum(item["pending_before"] for item in items)
        accepted = sum(item["accepted_pending"] for item in items)
        stage_summary[str(stage_index)] = {
            "requests": len(items),
            "document_fraction": describe(
                item["document_fraction"] for item in items
            ),
            "repair_ms": describe(item["repair_ms"] for item in items),
            "pending_tokens": pending,
            "accepted_prefix_tokens": accepted,
            "prefix_acceptance_rate": accepted / pending if pending else None,
            "requests_with_any_acceptance": sum(
                item["accepted_pending"] > 0 for item in items
            ),
        }

    result = {
        "cases": len(rows),
        "prompt_tokens": describe(row["prompt_tokens"] for row in rows),
        "document_tokens": describe(row["document_tokens"] for row in rows),
        "offline_document_store": {
            "shards": len(store_rows),
            "producer_calls_sum": sum(
                item["producer_calls"] for item in store_rows
            ),
            "logical_bytes_sum": sum(item["logical_bytes"] for item in store_rows),
            "producer_wall_ms_sum": sum(item["wall_ms"] for item in store_rows),
            "note": (
                "Each document is encoded once within its experiment shard; "
                "offline producer work is excluded from online latency."
            ),
        },
        "full_prefill": {
            "quality": quality(rows, "full_prefill"),
            "ttft_ms": describe(full_ttft),
            "response_ms": describe(full_response),
        },
        "cacheblend_15": {
            "quality": quality(rows, "cacheblend_15"),
            "token_match_full_prefill": fraction(
                rows, lambda row: row["cacheblend_15"]["token_match_full_prefill"]
            ),
            "normalized_match_full_prefill": fraction(
                rows,
                lambda row: row["cacheblend_15"][
                    "normalized_match_full_prefill"
                ],
            ),
            "ttft_ms": describe(cacheblend_ttft),
            "response_ms": describe(cacheblend_response),
            "repair_ms": describe(
                row["cacheblend_15"]["repair_ms"] for row in rows
            ),
            "h2d_ms": describe(
                row["cacheblend_15"]["transfer"]["total_h2d_ms"]
                for row in rows
            ),
            "ttft_saved_vs_full_prefill_ms": paired_delta(
                full_ttft, cacheblend_ttft
            ),
            "response_saved_vs_full_prefill_ms": paired_delta(
                full_response, cacheblend_response
            ),
        },
        "sparsecache_progressive": {
            "quality": quality(rows, "sparsecache_progressive"),
            "token_match_full_prefill": fraction(
                rows,
                lambda row: row["sparsecache_progressive"][
                    "token_match_full_prefill"
                ],
            ),
            "token_match_cacheblend": fraction(
                rows,
                lambda row: row["sparsecache_progressive"][
                    "token_match_cacheblend"
                ],
            ),
            "serial_pipeline_token_match": fraction(
                rows,
                lambda row: row["sparsecache_progressive"].get(
                    "serial_pipeline_token_match", False
                ),
            ),
            "response_ms": describe(progressive_response),
            "serial_response_ms": describe(serial_response),
            "first_draft_batch_ms": describe(
                row["sparsecache_progressive"]["first_draft_batch_ms"]
                for row in rows
                if row["sparsecache_progressive"]["first_draft_batch_ms"]
                is not None
            ),
            "first_committed_ms": describe(
                row["sparsecache_progressive"]["first_committed_ms"]
                for row in rows
                if row["sparsecache_progressive"]["first_committed_ms"] is not None
            ),
            "repair_ms": describe(
                row["sparsecache_progressive"]["repair_ms"] for row in rows
            ),
            "verify_ms": describe(
                row["sparsecache_progressive"]["verify_ms"] for row in rows
            ),
            "draft_and_replay_ms": describe(
                row["sparsecache_progressive"]["draft_decode_ms"]
                + row["sparsecache_progressive"]["draft_replay_ms"]
                for row in rows
            ),
            "h2d_ms": describe(
                row["sparsecache_progressive"]["transfer"]["total_h2d_ms"]
                for row in rows
            ),
            "pipeline_saved_vs_serial_ms": paired_delta(
                serial_response, progressive_response
            ),
            "pipeline_saved_vs_serial_on_token_matched_rows_ms": paired_delta(
                [
                    row["sparsecache_progressive"]["serial_response_ms"]
                    for row in serial_matched_rows
                ],
                [
                    row["sparsecache_progressive"]["response_ms"]
                    for row in serial_matched_rows
                ],
            ),
            "response_saved_vs_cacheblend_ms": paired_delta(
                cacheblend_response, progressive_response
            ),
            "response_saved_vs_full_prefill_ms": paired_delta(
                full_response, progressive_response
            ),
            "stages": stage_summary,
        },
    }
    result["cacheblend_15"]["f1_delta_vs_full_prefill"] = paired_delta(
        [row["cacheblend_15"]["f1"] for row in rows],
        [row["full_prefill"]["f1"] for row in rows],
    )
    result["sparsecache_progressive"]["f1_delta_vs_full_prefill"] = paired_delta(
        [row["sparsecache_progressive"]["f1"] for row in rows],
        [row["full_prefill"]["f1"] for row in rows],
    )
    return result


def main():
    args = parse_args()
    summary = summarize(load_rows(args.inputs))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
