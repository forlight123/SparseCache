"""Summarize online sparse-KV proposal and full-verifier feedback traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from experiments.lossless_pd.lmcache_pd.benchmark import bootstrap_mean_ci


def describe(values: Iterable[float]) -> dict | None:
    values = list(values)
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "bootstrap_95ci": bootstrap_mean_ci(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row.get("event", "") for row in rows)
    drafts = [row for row in rows if row.get("event") == "live_sparse_draft"]
    handoffs = [row for row in rows if row.get("event") == "online_draft_handoff"]
    feedback = [row for row in rows if row.get("event") == "online_verify_feedback"]
    injections = [row for row in handoffs if row.get("returned_proposals")]
    accepted = [int(row["accepted_prefix"]) for row in feedback]
    accepted_suffix = [max(0, value - 1) for value in accepted]
    requests = len(handoffs)
    if len(feedback) > len(injections):
        raise ValueError("verifier feedback exceeds injected proposal blocks")
    if any(value <= 0 for value in accepted):
        raise ValueError("injected blocks must include the reconciled first token")
    return {
        "events": dict(sorted(counts.items())),
        "requests_with_draft": len(drafts),
        "handoffs": requests,
        "first_token_matches": sum(
            bool(row.get("first_proposal_matches")) for row in handoffs
        ),
        "first_token_match_rate": (
            sum(bool(row.get("first_proposal_matches")) for row in handoffs) / requests
            if requests
            else None
        ),
        "injected_blocks": len(injections),
        "feedback_blocks": len(feedback),
        "feedback_missing_at_trace_end": len(injections) - len(feedback),
        "accepted_prefix_distribution_injected": dict(
            sorted(Counter(str(value) for value in accepted).items())
        ),
        "mean_accepted_prefix_per_request": (
            sum(accepted) / requests if requests else None
        ),
        "mean_accepted_prefix_per_injected_block": (
            statistics.fmean(accepted) if accepted else None
        ),
        "mean_accepted_injected_suffix_per_request": (
            sum(accepted_suffix) / requests if requests else None
        ),
        "draft_total_gpu_ms": describe(float(row["total_gpu_ms"]) for row in drafts),
        "draft_total_gpu_ms_excluding_first": describe(
            float(row["total_gpu_ms"]) for row in drafts[1:]
        ),
        "draft_wall_ms_excluding_first": describe(
            float(row["wall_ms"]) for row in drafts[1:]
        ),
        "draft_ready_lead_ms": describe(
            float(row["draft_ready_lead_ms"]) for row in handoffs
        ),
        "target_conditioned_repairs": sum(
            bool(row.get("target_conditioned_repair")) for row in handoffs
        ),
        "repair_gpu_ms": describe(
            float(row["repair_gpu_ms"])
            for row in handoffs
            if row.get("repair_gpu_ms") is not None
        ),
        "repair_wall_ms": describe(
            float(row["repair_wall_ms"])
            for row in handoffs
            if row.get("repair_wall_ms") is not None
        ),
        "visible_token_fraction": describe(
            float(row["actual_fraction"]) for row in drafts
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trace = Path(args.trace).resolve()
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line]
    result = {
        "contract": (
            "one online direct sparse-KV draft per real LMCache Anchor; first token "
            "reconciled with Target; suffix accepted only by the full-KV verifier"
        ),
        "trace": str(trace),
        "summary": summarize(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
