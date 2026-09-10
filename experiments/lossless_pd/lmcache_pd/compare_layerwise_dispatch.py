"""Compare early layer-ready dispatch with the wait-for-FullReady control.

The endpoint benchmark and physical LMCache traces are deliberately analyzed
separately.  Endpoint rows establish lossless output equality and paired
latency.  The optional traces establish that the sparse draft really ran
inside the measured Anchor-to-FullReady transfer window and that no KV object
was retransmitted.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.lossless_pd.lmcache_pd.benchmark import bootstrap_mean_ci


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise TypeError(f"benchmark has no rows: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _delta(values: list[float], *, early_label: str = "early") -> dict[str, Any]:
    return {
        "mean_ms": statistics.fmean(values),
        "bootstrap_95ci_ms": bootstrap_mean_ci(values),
        f"{early_label}_faster": sum(value < 0 for value in values),
        "same": sum(value == 0 for value in values),
        f"{early_label}_slower": sum(value > 0 for value in values),
    }


def compare_benchmarks(early: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    early_rows = early["rows"]
    full_rows = full["rows"]
    early_ids = [row["record_id"] for row in early_rows]
    full_ids = [row["record_id"] for row in full_rows]
    if early_ids != full_ids:
        raise ValueError("early and FullReady benchmark record order differs")
    if not early_rows:
        raise ValueError("cannot compare empty benchmarks")

    early_full_equal = [
        left["pd"]["text"] == right["pd"]["text"]
        for left, right in zip(early_rows, full_rows, strict=True)
    ]
    summary: dict[str, Any] = {
        "requests": len(early_rows),
        "early_equals_full_pd": sum(early_full_equal),
        "early_full_mismatch_ordinals": [
            index for index, equal in enumerate(early_full_equal) if not equal
        ],
        "early_equals_monolithic": sum(row["outputs_equal"] for row in early_rows),
        "full_equals_monolithic": sum(row["outputs_equal"] for row in full_rows),
    }
    for metric in ("ttft_ms", "total_ms"):
        raw = [
            left["pd"][metric] - right["pd"][metric]
            for left, right in zip(early_rows, full_rows, strict=True)
        ]
        adjusted = [
            (left["pd"][metric] - left["monolithic"][metric])
            - (right["pd"][metric] - right["monolithic"][metric])
            for left, right in zip(early_rows, full_rows, strict=True)
        ]
        summary[f"early_minus_full_{metric}"] = _delta(raw)
        summary[f"monolithic_adjusted_{metric}"] = {
            "mean_ms": statistics.fmean(adjusted),
            "bootstrap_95ci_ms": bootstrap_mean_ci(adjusted),
        }
    return summary


def summarize_physical_traces(
    *,
    proxy_rows: list[dict[str, Any]],
    sender_rows: list[dict[str, Any]],
    receiver_rows: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    dispatch = [row for row in proxy_rows if row.get("event") == "decoder_dispatch"]
    request_ids = [str(row["pd_request_id"]) for row in dispatch]
    if not request_ids:
        raise ValueError("proxy trace contains no decoder dispatch")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("proxy trace repeats a P/D request ID")

    schedules = {
        str(row["pd_request_id"]): row
        for row in sender_rows
        if row.get("event") == "layerwise_schedule"
    }
    receiver_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receiver_rows:
        receiver_by_id[str(row.get("pd_request_id", ""))].append(row)
    draft_by_id = {
        str(row["pd_request_id"]): row
        for row in draft_rows
        if row.get("event") == "live_sparse_draft"
    }
    handoff_by_id = {
        str(row["pd_request_id"]): row
        for row in draft_rows
        if row.get("event") == "online_draft_handoff"
    }
    feedback_by_id = {
        str(row["pd_request_id"]): row
        for row in draft_rows
        if row.get("event") == "online_verify_feedback"
    }

    metrics = []
    missing: dict[str, list[str]] = defaultdict(list)
    for pd_request_id in request_ids:
        schedule = schedules.get(pd_request_id)
        draft = draft_by_id.get(pd_request_id)
        events = receiver_by_id[pd_request_id]
        anchor = next(
            (row for row in events if row.get("event") == "receiver_anchor_ready"),
            None,
        )
        layers = [row for row in events if row.get("event") == "receiver_layer_ready"]
        for label, value in (
            ("schedule", schedule),
            ("anchor", anchor),
            ("draft", draft),
            ("layers", layers),
        ):
            if not value:
                missing[pd_request_id].append(label)
        if missing[pd_request_id]:
            continue
        assert schedule is not None and anchor is not None and draft is not None
        full_ns = max(int(row["receiver_received_ns"]) for row in layers)
        anchor_ns = int(anchor["receiver_received_ns"])
        draft_finished_ns = int(draft["draft_finished_ns"])
        feedback = feedback_by_id.get(pd_request_id)
        metrics.append(
            {
                "pd_request_id": pd_request_id,
                "anchor_to_full_ms": (full_ns - anchor_ns) / 1e6,
                "draft_finish_minus_full_ms": (draft_finished_ns - full_ns) / 1e6,
                "draft_finished_before_full": draft_finished_ns <= full_ns,
                "accepted_injected_suffix": int(
                    feedback.get("accepted_injected_suffix", 0) if feedback else 0
                ),
                "handoff": pd_request_id in handoff_by_id,
                "actual_anchor_fraction": float(draft["actual_fraction"]),
                "draft_gpu_ms": float(draft["total_gpu_ms"]),
                "draft_wall_ms": float(draft["wall_ms"]),
                "wire_bytes": int(schedule["wire_bytes"]),
                "authoritative_bytes": int(schedule["authoritative_bytes"]),
                "retransmitted_bytes": int(schedule["retransmitted_bytes"]),
            }
        )

    incomplete = {key: value for key, value in missing.items() if value}
    if incomplete:
        raise ValueError(f"incomplete physical trace requests: {incomplete}")
    if len(metrics) != len(request_ids):
        raise ValueError("physical trace join lost requests")
    return {
        "requests": len(metrics),
        "mean_anchor_to_full_ms": statistics.fmean(
            row["anchor_to_full_ms"] for row in metrics
        ),
        "mean_draft_finish_minus_full_ms": statistics.fmean(
            row["draft_finish_minus_full_ms"] for row in metrics
        ),
        "draft_finished_before_full": sum(
            row["draft_finished_before_full"] for row in metrics
        ),
        "handoffs": sum(row["handoff"] for row in metrics),
        "mean_accepted_injected_suffix": statistics.fmean(
            row["accepted_injected_suffix"] for row in metrics
        ),
        "positive_acceptance_requests": sum(
            row["accepted_injected_suffix"] > 0 for row in metrics
        ),
        "mean_actual_anchor_fraction": statistics.fmean(
            row["actual_anchor_fraction"] for row in metrics
        ),
        "mean_draft_gpu_ms": statistics.fmean(row["draft_gpu_ms"] for row in metrics),
        "mean_draft_wall_ms": statistics.fmean(row["draft_wall_ms"] for row in metrics),
        "wire_bytes": sum(row["wire_bytes"] for row in metrics),
        "authoritative_bytes": sum(row["authoritative_bytes"] for row in metrics),
        "retransmitted_bytes": sum(row["retransmitted_bytes"] for row in metrics),
        "rows": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy-trace", type=Path)
    parser.add_argument("--sender-trace", type=Path)
    parser.add_argument("--receiver-trace", type=Path)
    parser.add_argument("--draft-trace", type=Path)
    args = parser.parse_args()

    trace_paths = (
        args.proxy_trace,
        args.sender_trace,
        args.receiver_trace,
        args.draft_trace,
    )
    if any(trace_paths) and not all(trace_paths):
        parser.error("all four physical trace paths must be supplied together")
    result: dict[str, Any] = {
        "contract": (
            "separate fresh-server runs over identical greedy requests; paired "
            "early-layer dispatch minus wait-for-FullReady control"
        ),
        "early": str(args.early.resolve()),
        "full": str(args.full.resolve()),
        "summary": compare_benchmarks(
            _read_json(args.early.resolve()), _read_json(args.full.resolve())
        ),
    }
    if all(trace_paths):
        result["physical"] = summarize_physical_traces(
            proxy_rows=_read_jsonl(args.proxy_trace.resolve()),
            sender_rows=_read_jsonl(args.sender_trace.resolve()),
            receiver_rows=_read_jsonl(args.receiver_trace.resolve()),
            draft_rows=_read_jsonl(args.draft_trace.resolve()),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
