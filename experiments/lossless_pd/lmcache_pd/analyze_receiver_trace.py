"""Summarize decoder-side AnchorReady mailbox traces."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.lossless_pd.lmcache_pd.analyze_trace import bootstrap_mean_ci


def metric(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "bootstrap_95ci": bootstrap_mean_ci(values),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no receiver_anchor_ready events in trace")
    complete = [row for row in rows if row.get("complete")]
    result = {
        "requests": len(rows),
        "complete_requests": len(complete),
        "complete_rate": len(complete) / len(rows),
        "control_plane_ms": metric([row["control_plane_ms"] for row in rows]),
        "lookup_ms": metric([row["lookup_ms"] for row in rows]),
        "expected_keys": sum(row["expected_keys"] for row in rows),
        "found_keys": sum(row["found_keys"] for row in rows),
        "expected_resident_bytes": sum(
            row["expected_resident_bytes"] for row in rows
        ),
        "resolved_bytes": sum(row["resolved_bytes"] for row in rows),
    }
    layer_rows = [row["layer_views"] for row in rows if "layer_views" in row]
    if layer_rows:
        result["layer_views"] = {
            "requests": len(layer_rows),
            "all_storage_aliases": all(
                row["all_storage_aliases"] for row in layer_rows
            ),
            "logical_bytes": sum(row["logical_bytes"] for row in layer_rows),
            "views": sum(row["views"] for row in layer_rows),
            "inspection_ms": metric(
                [row["inspection_ms"] for row in layer_rows]
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--last", type=int)
    args = parser.parse_args()
    source = Path(args.trace)
    events = [json.loads(line) for line in source.read_text().splitlines() if line]
    rows = [row for row in events if row.get("event") == "receiver_anchor_ready"]
    if args.last is not None:
        if args.last <= 0:
            parser.error("last must be positive")
        rows = rows[-args.last :]
    result = {
        "contract": (
            "NIXL remote completion to decoder mailbox resolution of every "
            "advertised resident CUDA KV object"
        ),
        "trace": str(source.resolve()),
        "filter_last": args.last,
        "summary": summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
