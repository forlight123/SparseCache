"""Evaluate the frozen lossless external-drafter stop-loss gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from experiments.lossless_pd.lmcache_pd.compare_online_modes import sandwich_summary


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise TypeError(f"benchmark has no rows: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze(
    observe_before: dict[str, Any],
    inject: dict[str, Any],
    observe_after: dict[str, Any],
    *,
    proxy_rows: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
    min_accepted_suffix: float = 3.5,
    min_speedup: float = 1.10,
) -> dict[str, Any]:
    before_rows = observe_before["rows"]
    inject_rows = inject["rows"]
    after_rows = observe_after["rows"]
    paired = sandwich_summary(before_rows, inject_rows, after_rows)
    requests = len(inject_rows)
    if requests == 0:
        raise ValueError("external draft gate requires at least one request")

    feedback = [
        row for row in draft_rows if row.get("event") == "online_verify_feedback"
    ]
    handoffs = [
        row for row in draft_rows if row.get("event") == "online_draft_handoff"
    ]
    submissions = [
        row for row in proxy_rows if row.get("event") == "external_draft_submitted"
    ]
    dispatches = [
        row for row in proxy_rows if row.get("event") == "decoder_dispatch"
    ]
    if min(len(handoffs), len(submissions), len(dispatches)) != requests:
        raise ValueError("external draft physical trace is incomplete")

    accepted = sum(int(row["accepted_injected_suffix"]) for row in feedback)
    mean_accepted = accepted / requests
    baseline_total_ms = statistics.fmean(
        [
            statistics.fmean(row["pd"]["total_ms"] for row in before_rows),
            statistics.fmean(row["pd"]["total_ms"] for row in after_rows),
        ]
    )
    inject_total_ms = statistics.fmean(
        row["pd"]["total_ms"] for row in inject_rows
    )
    speedup = baseline_total_ms / inject_total_ms
    output_equal = paired["all_three_pd_outputs_equal"] == requests
    draft_before_full = sum(
        not bool(row["full_ready_at_dispatch"]) for row in dispatches
    )
    positive_ci = paired["inject_minus_sandwich_observe_total_ms"][
        "bootstrap_95ci_ms"
    ][1] < 0
    gates = {
        "output_id_equality": output_equal,
        "mean_accepted_suffix_at_least_threshold": (
            mean_accepted >= min_accepted_suffix
        ),
        "all_drafts_before_fullready": draft_before_full == requests,
        "total_saving_ci_strictly_positive": positive_ci,
        "speedup_at_least_threshold": speedup >= min_speedup,
    }
    return {
        "contract": (
            "fresh-server observe-inject-observe; greedy token-ID equality; "
            "third drafter GPU charged; thresholds fixed before live run"
        ),
        "requests": requests,
        "thresholds": {
            "mean_accepted_suffix": min_accepted_suffix,
            "total_speedup": min_speedup,
        },
        "metrics": {
            "all_three_output_ids_equal": paired["all_three_pd_outputs_equal"],
            "mean_accepted_injected_suffix": mean_accepted,
            "positive_acceptance_requests": sum(
                int(row["accepted_injected_suffix"]) > 0 for row in feedback
            ),
            "seed_branch_hits": sum(
                bool(row["seed_branch_hit"]) for row in submissions
            ),
            "drafts_before_fullready": draft_before_full,
            "observe_sandwich_total_ms": baseline_total_ms,
            "inject_total_ms": inject_total_ms,
            "total_speedup": speedup,
            "inject_minus_observe_total_ms": paired[
                "inject_minus_sandwich_observe_total_ms"
            ],
            "inject_minus_observe_ttft_ms": paired[
                "inject_minus_sandwich_observe_ttft_ms"
            ],
        },
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "continue_external_portfolio"
            if all(gates.values())
            else "stop_external_portfolio_as_iclr_mainline"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe-before", type=Path, required=True)
    parser.add_argument("--inject", type=Path, required=True)
    parser.add_argument("--observe-after", type=Path, required=True)
    parser.add_argument("--proxy-trace", type=Path, required=True)
    parser.add_argument("--draft-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-accepted-suffix", type=float, default=3.5)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    args = parser.parse_args()
    result = analyze(
        read_json(args.observe_before.resolve()),
        read_json(args.inject.resolve()),
        read_json(args.observe_after.resolve()),
        proxy_rows=read_jsonl(args.proxy_trace.resolve()),
        draft_rows=read_jsonl(args.draft_trace.resolve()),
        min_accepted_suffix=args.min_accepted_suffix,
        min_speedup=args.min_speedup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
