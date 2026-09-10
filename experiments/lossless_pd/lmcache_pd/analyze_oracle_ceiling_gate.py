"""Evaluate an optimistic same-stack Target oracle ceiling cell."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.lossless_pd.lmcache_pd.compare_online_modes import (
    sandwich_summary,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise TypeError(f"benchmark has no rows: {path}")
    if value.get("status") not in (None, "completed"):
        raise ValueError(f"benchmark is incomplete: {path}")
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
    draft_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    min_speedup: float = 1.10,
) -> dict[str, Any]:
    before_rows = observe_before["rows"]
    inject_rows = inject["rows"]
    after_rows = observe_after["rows"]
    paired = sandwich_summary(before_rows, inject_rows, after_rows)
    requests = len(inject_rows)
    if requests == 0:
        raise ValueError("oracle ceiling gate requires at least one request")
    feedback = [
        row for row in draft_rows if row.get("event") == "online_verify_feedback"
    ]
    if len(feedback) != requests:
        raise ValueError("oracle ceiling requires one verify feedback per request")
    accepted = [int(row["accepted_injected_suffix"]) for row in feedback]
    replay = [row for row in proxy_rows if row.get("event") == "canonical_replay"]
    baseline_ms = statistics.fmean(
        [
            statistics.fmean(row["pd"]["total_ms"] for row in before_rows),
            statistics.fmean(row["pd"]["total_ms"] for row in after_rows),
        ]
    )
    inject_ms = statistics.fmean(row["pd"]["total_ms"] for row in inject_rows)
    exact = paired["all_three_pd_outputs_equal"] == requests
    speedup = baseline_ms / inject_ms
    gates = {
        "same_stack_token_id_equality": exact,
        "total_speedup_at_least_threshold": speedup >= min_speedup,
        "total_saving_ci_strictly_positive": paired[
            "inject_minus_sandwich_observe_total_ms"
        ]["bootstrap_95ci_ms"][1]
        < 0,
    }
    return {
        "contract": (
            "fresh-server observe-oracle-inject-observe; exact same-stack Target "
            "proposal at zero draft cost; full-KV verifier; greedy token-ID gate"
        ),
        "requests": requests,
        "thresholds": {"total_speedup": min_speedup},
        "metrics": {
            "all_three_output_ids_equal": paired["all_three_pd_outputs_equal"],
            "mismatch_ordinals": paired["pd_mismatch_ordinals"],
            "accepted_suffix_mean": statistics.fmean(accepted),
            "accepted_suffix_distribution": {
                str(key): value for key, value in sorted(Counter(accepted).items())
            },
            "observe_sandwich_total_ms": baseline_ms,
            "inject_total_ms": inject_ms,
            "total_speedup": speedup,
            "inject_minus_observe_total_ms": paired[
                "inject_minus_sandwich_observe_total_ms"
            ],
            "inject_minus_observe_ttft_ms": paired[
                "inject_minus_sandwich_observe_ttft_ms"
            ],
            "canonical_replays": len(replay),
            "canonical_replay_mean_ms": (
                statistics.fmean(float(row["replay_ms"]) for row in replay)
                if replay
                else None
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "oracle_ceiling_passes"
            if all(gates.values())
            else "reject_current_verifier_cell"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe-before", type=Path, required=True)
    parser.add_argument("--inject", type=Path, required=True)
    parser.add_argument("--observe-after", type=Path, required=True)
    parser.add_argument("--draft-trace", type=Path, required=True)
    parser.add_argument("--proxy-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    args = parser.parse_args()
    result = analyze(
        read_json(args.observe_before.resolve()),
        read_json(args.inject.resolve()),
        read_json(args.observe_after.resolve()),
        draft_rows=read_jsonl(args.draft_trace.resolve()),
        proxy_rows=read_jsonl(args.proxy_trace.resolve()),
        min_speedup=args.min_speedup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
