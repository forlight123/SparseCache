"""Reconstruct AnchorReady and FullReady from gather-first LMCache traces."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Iterable


def bootstrap_mean_ci(
    values: Iterable[float], *, samples: int = 10_000, seed: int = 20260909
) -> list[float]:
    values = list(values)
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return [values[0], values[0]]
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def _match_write(
    gather: dict, phase: dict, writes: list[dict], used: set[int]
) -> tuple[int, dict]:
    """Match old numeric-id traces by time and new traces by request id."""

    request_id = gather.get("request_id", "")
    exact = [
        (index, row)
        for index, row in enumerate(writes)
        if index not in used
        and row.get("phase") == phase["phase"]
        and request_id
        and row.get("request_id") == request_id
    ]
    candidates = exact or [
        (index, row)
        for index, row in enumerate(writes)
        if index not in used
        and row.get("phase") == phase["phase"]
        and phase["gather_done_ns"] <= row["started_ns"]
        and row["started_ns"] <= gather["store_returned_ns"] + 50_000_000
    ]
    if not candidates:
        raise ValueError(
            f"no {phase['phase']} write for request {request_id or '<unknown>'}"
        )
    return min(
        candidates,
        key=lambda item: abs(item[1]["started_ns"] - phase["submitted_ns"]),
    )


def reconstruct(rows: list[dict], *, last: int | None = None) -> list[dict]:
    gathers = [row for row in rows if row.get("event") == "gather_submit"]
    writes = [row for row in rows if row.get("event") == "nixl_write"]
    if last is not None:
        if last <= 0:
            raise ValueError("last must be positive")
        gathers = gathers[-last:]
    used: set[int] = set()
    result = []
    for gather in gathers:
        phases = {row["phase"]: row for row in gather["phases"]}
        if set(phases) != {"anchor", "residual"}:
            continue
        anchor_index, anchor_write = _match_write(
            gather, phases["anchor"], writes, used
        )
        used.add(anchor_index)
        residual_index, residual_write = _match_write(
            gather, phases["residual"], writes, used
        )
        used.add(residual_index)
        anchor_ready = (
            anchor_write["finished_ns"] - gather["store_started_ns"]
        ) / 1e6
        full_ready = (
            residual_write["finished_ns"] - gather["store_started_ns"]
        ) / 1e6
        row = {
            "request_id": gather.get("request_id", ""),
            "total_chunks": gather["total_chunks"],
            "total_bytes": gather["total_bytes"],
            "anchor_chunks": phases["anchor"]["chunks"],
            "anchor_bytes": phases["anchor"]["bytes"],
            "anchor_fraction_bytes": (
                phases["anchor"]["bytes"] / gather["total_bytes"]
            ),
            "anchor_gather_ms": phases["anchor"]["gather_ms"],
            "residual_gather_ms": phases["residual"]["gather_ms"],
            "anchor_write_ms": anchor_write["write_ms"],
            "residual_write_ms": residual_write["write_ms"],
            "anchor_ready_ms": anchor_ready,
            "full_ready_ms": full_ready,
            "draft_window_ms": full_ready - anchor_ready,
        }
        seed_record = anchor_write.get("seed_record") or gather.get("seed_record")
        if seed_record is not None:
            sampled_ns = seed_record["seed_sampled_ns"]
            row.update(
                seed_token_id=seed_record["seed_token_id"],
                seed_to_anchor_ready_ms=(
                    anchor_write["finished_ns"] - sampled_ns
                )
                / 1e6,
                seed_lead_before_store_ms=(
                    gather["store_started_ns"] - sampled_ns
                )
                / 1e6,
            )
        result.append(row)
    return result


def metric(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "bootstrap_95ci": bootstrap_mean_ci(values),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no complete two-phase requests in trace")
    names = [
        "anchor_fraction_bytes",
        "anchor_gather_ms",
        "residual_gather_ms",
        "anchor_write_ms",
        "residual_write_ms",
        "anchor_ready_ms",
        "full_ready_ms",
        "draft_window_ms",
    ]
    if all("seed_to_anchor_ready_ms" in row for row in rows):
        names.extend(("seed_to_anchor_ready_ms", "seed_lead_before_store_ms"))
    return {
        "requests": len(rows),
        **{name: metric([row[name] for row in rows]) for name in names},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--last", type=int)
    parser.add_argument("--chunks", type=int)
    args = parser.parse_args()
    trace_path = Path(args.trace)
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    rows = reconstruct(events, last=args.last)
    if args.chunks is not None:
        rows = [row for row in rows if row["total_chunks"] == args.chunks]
    result = {
        "contract": "monotonic store-start to actual asynchronous NIXL completion",
        "trace": str(trace_path.resolve()),
        "filter_last": args.last,
        "filter_chunks": args.chunks,
        "summary": summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
