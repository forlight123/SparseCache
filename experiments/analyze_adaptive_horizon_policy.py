"""Train and freeze a causal sparse-draft horizon stopping policy.

The controller starts at the shortest measured horizon and can only inspect
the sparse proposal margins that have already been computed.  It either stops
at the current horizon or continues to the next measured horizon.  Thresholds
are selected on a training split and evaluated unchanged on disjoint requests.
This is intentionally a small, interpretable policy class rather than an
in-sample per-request oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

if __package__:
    from .analyze_joint_pd_policy import (
        _describe,
        _extract,
        apply_adjudications,
        write_json,
    )
else:
    from analyze_joint_pd_policy import (  # type: ignore[no-redef]
        _describe,
        _extract,
        apply_adjudications,
        write_json,
    )

DEFAULT_THRESHOLDS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, math.inf)


def _ordered_labels(actions: dict[str, dict[str, dict]]) -> list[str]:
    configured = {
        label: next(iter(rows.values()))["configured"] for label, rows in actions.items()
    }
    if len(set(configured.values())) != len(configured):
        raise ValueError("candidate horizons must be unique")
    return sorted(actions, key=configured.get)


def validate_actions(actions: dict[str, dict[str, dict]]) -> list[str]:
    labels = _ordered_labels(actions)
    ids = list(actions[labels[0]])
    if any(list(actions[label]) != ids for label in labels[1:]):
        raise ValueError("candidate request order differs")
    for request_id in ids:
        longest = actions[labels[-1]][request_id]["draft_token_ids"]
        for label in labels:
            row = actions[label][request_id]
            meaningful = row.get("meaningful_draft_count", row["configured"])
            if row["draft_token_ids"] != longest[:meaningful]:
                raise ValueError(
                    f"proposal prefix differs for {request_id} at {label}"
                )
            if len(row["draft_margins"]) != meaningful:
                raise ValueError(
                    f"meaningful margin count differs from proposal at {label}"
                )
    return labels


def choose(
    actions: dict[str, dict[str, dict]],
    labels: list[str],
    thresholds: dict[str, float],
    request_id: str,
) -> str:
    for label in labels[:-1]:
        if actions[label][request_id].get("draft_reached_termination", False):
            return label
        if actions[label][request_id]["draft_min_margin"] < thresholds[label]:
            return label
    return labels[-1]


def fit_thresholds(
    actions: dict[str, dict[str, dict]],
    labels: list[str],
    train_ids: list[str],
    *,
    candidates: tuple[float, ...] = DEFAULT_THRESHOLDS,
    min_leaf: int = 10,
) -> dict[str, float]:
    if min_leaf <= 0:
        raise ValueError("min_leaf must be positive")
    thresholds: dict[str, float] = {}
    # Backward induction: at each gate, continuing follows the already-frozen
    # policy at later gates. Earlier gates are irrelevant to this local fit.
    for position in range(len(labels) - 2, -1, -1):
        label = labels[position]
        later = labels[position + 1 :]
        scored = []
        for threshold in candidates:
            stop = [
                request_id
                for request_id in train_ids
                if actions[label][request_id].get(
                    "draft_reached_termination", False
                )
                or actions[label][request_id]["draft_min_margin"] < threshold
            ]
            continued = [request_id for request_id in train_ids if request_id not in stop]
            if stop and continued and min(len(stop), len(continued)) < min_leaf:
                continue
            values = []
            for request_id in train_ids:
                if request_id in stop:
                    selected = label
                else:
                    selected = choose(actions, later, thresholds, request_id)
                values.append(actions[selected][request_id]["paired_saving_ms"])
            # Prefer the higher threshold (less extra drafting) on exact ties.
            scored.append((statistics.fmean(values), threshold))
        if not scored:
            raise ValueError(f"no admissible threshold for {label}")
        thresholds[label] = max(scored)[1]
    return thresholds


def evaluate(
    actions: dict[str, dict[str, dict]],
    labels: list[str],
    thresholds: dict[str, float],
    ids: list[str],
    *,
    fixed_label: str,
) -> dict:
    selected = [choose(actions, labels, thresholds, request_id) for request_id in ids]
    policy = [
        actions[label][request_id]["paired_saving_ms"]
        for request_id, label in zip(ids, selected, strict=True)
    ]
    fixed = [actions[fixed_label][request_id]["paired_saving_ms"] for request_id in ids]
    return {
        "requests": len(ids),
        "selected_counts": {label: selected.count(label) for label in labels},
        "selected_outputs_equal": sum(
            actions[label][request_id]["token_match_target"]
            for request_id, label in zip(ids, selected, strict=True)
        ),
        "paired_saving_ms": _describe(policy),
        "advantage_vs_frozen_fixed_ms": _describe(
            [left - right for left, right in zip(policy, fixed, strict=True)]
        ),
    }


def analyze(
    actions: dict[str, dict[str, dict]],
    *,
    train_count: int,
    min_leaf: int = 10,
) -> dict:
    labels = validate_actions(actions)
    ids = list(actions[labels[0]])
    if not 0 < train_count < len(ids):
        raise ValueError("train_count must leave non-empty train and evaluation splits")
    train_ids = ids[:train_count]
    evaluation_ids = ids[train_count:]
    fixed_label = max(
        labels,
        key=lambda label: statistics.fmean(
            actions[label][request_id]["paired_saving_ms"] for request_id in train_ids
        ),
    )
    thresholds = fit_thresholds(
        actions, labels, train_ids, min_leaf=min_leaf
    )
    return {
        "contract": (
            "causal minimum-margin stopping thresholds and fixed-horizon comparator "
            "selected only on the leading training requests, then frozen on disjoint "
            "evaluation requests"
        ),
        "horizon_order": labels,
        "thresholds_continue_when_margin_at_least": {
            label: (None if math.isinf(value) else value)
            for label, value in thresholds.items()
        },
        "frozen_fixed_label": fixed_label,
        "train": evaluate(
            actions, labels, thresholds, train_ids, fixed_label=fixed_label
        ),
        "evaluation": evaluate(
            actions, labels, thresholds, evaluation_ids, fixed_label=fixed_label
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--min-leaf", type=int, default=10)
    parser.add_argument(
        "--adjudication",
        action="append",
        default=[],
        help="LABEL:single-request-remeasurement.jsonl; raw files remain unchanged",
    )
    parser.add_argument(
        "--diagnostic-allow-output-mismatch",
        action="store_true",
        help="report numerical mismatches instead of failing; never use as exact evidence",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    actions = {}
    for item in args.candidate:
        label, path = item.split(":", 1)
        if label in actions:
            raise ValueError(f"duplicate candidate label: {label}")
        actions[label] = _extract(
            label,
            Path(path),
            require_exposed_equality=not args.diagnostic_allow_output_mismatch,
        )
    adjudications = apply_adjudications(
        actions,
        args.adjudication,
        require_exposed_equality=not args.diagnostic_allow_output_mismatch,
    )
    result = analyze(actions, train_count=args.train_count, min_leaf=args.min_leaf)
    result["adjudications"] = adjudications
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
