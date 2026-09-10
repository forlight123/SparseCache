"""Analyze whether online logit margins can gate canonical verifier repair."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed" or not isinstance(value.get("rows"), list):
        raise ValueError(f"benchmark is incomplete: {path}")
    return value


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for position, (lhs, rhs) in enumerate(zip(left, right, strict=False)):
        if lhs != rhs:
            return position
    return None if len(left) == len(right) else min(len(left), len(right))


def score_map(output: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["position"]): row
        for row in output.get("token_scores", [])
        if isinstance(row, dict) and "position" in row
    }


def min_margin_after(output: dict[str, Any], position: int) -> float | None:
    margins = [
        float(row["top1_top2_margin"])
        for row in output.get("token_scores", [])
        if int(row["position"]) > position
        and row.get("top1_top2_margin") is not None
    ]
    return min(margins) if margins else None


def analyze(
    baseline: dict[str, Any],
    inject: dict[str, Any],
    *,
    block_end_position: int,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    baseline_by_record = {row["record_id"]: row for row in baseline["rows"]}
    if len(baseline_by_record) != len(baseline["rows"]):
        raise ValueError("baseline record IDs must be unique")
    rows = []
    for injected in inject["rows"]:
        control = baseline_by_record.get(injected["record_id"])
        if control is None:
            raise ValueError(f"baseline row is missing: {injected['record_id']}")
        left = control["pd"]["token_ids"]
        right = injected["pd"]["token_ids"]
        divergence = first_divergence(left, right)
        left_scores = score_map(control["pd"])
        right_scores = score_map(injected["pd"])
        common_deltas = []
        prefix_end = divergence if divergence is not None else min(len(left), len(right))
        for position in range(1, prefix_end):
            lhs = left_scores.get(position, {}).get("top_logprobs") or {}
            rhs = right_scores.get(position, {}).get("top_logprobs") or {}
            common_deltas.extend(
                abs(float(lhs[token]) - float(rhs[token]))
                for token in lhs.keys() & rhs.keys()
            )
        rows.append(
            {
                "ordinal": int(injected["ordinal"]),
                "packet_index": int(injected["packet_index"]),
                "record_id": injected["record_id"],
                "outputs_equal": divergence is None,
                "first_divergence": divergence,
                "baseline_margin_at_divergence": (
                    left_scores.get(divergence, {}).get("top1_top2_margin")
                    if divergence is not None
                    else None
                ),
                "inject_margin_at_divergence": (
                    right_scores.get(divergence, {}).get("top1_top2_margin")
                    if divergence is not None
                    else None
                ),
                "min_inject_margin_after_block": min_margin_after(
                    injected["pd"], block_end_position
                ),
                "max_common_topk_logprob_delta_before_divergence": (
                    max(common_deltas) if common_deltas else None
                ),
            }
        )

    drift = sum(not row["outputs_equal"] for row in rows)
    gates = {}
    for threshold in thresholds:
        predicted = [
            row["min_inject_margin_after_block"] is not None
            and row["min_inject_margin_after_block"] <= threshold
            for row in rows
        ]
        actual = [not row["outputs_equal"] for row in rows]
        tp = sum(guess and label for guess, label in zip(predicted, actual, strict=True))
        fp = sum(guess and not label for guess, label in zip(predicted, actual, strict=True))
        fn = sum(not guess and label for guess, label in zip(predicted, actual, strict=True))
        tn = len(rows) - tp - fp - fn
        gates[str(threshold)] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "recall": tp / drift if drift else 1.0,
            "repair_fraction": (tp + fp) / len(rows) if rows else 0.0,
            "empirically_lossless_after_repair": fn == 0,
        }
    deltas = [
        float(row["max_common_topk_logprob_delta_before_divergence"])
        for row in rows
        if row["max_common_topk_logprob_delta_before_divergence"] is not None
    ]
    return {
        "contract": (
            "diagnostic only: streamed top-k logprob margins after one verified "
            "block; thresholding is not a numerical proof"
        ),
        "requests": len(rows),
        "block_end_position": block_end_position,
        "drift_requests": drift,
        "max_observed_common_topk_logprob_delta": max(deltas) if deltas else None,
        "mean_observed_common_topk_logprob_delta": (
            statistics.fmean(deltas) if deltas else None
        ),
        "thresholds": gates,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--inject", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-end-position", type=int, default=8)
    parser.add_argument("--thresholds", default="0.25,0.5,1.0")
    args = parser.parse_args()
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    result = analyze(
        read_result(args.baseline.resolve()),
        read_result(args.inject.resolve()),
        block_end_position=args.block_end_position,
        thresholds=thresholds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
