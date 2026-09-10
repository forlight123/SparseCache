"""Paired comparison of online sparse-KV observe and injection runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.lossless_pd.lmcache_pd.benchmark import bootstrap_mean_ci


def _same_output(left: dict, right: dict) -> bool:
    left_tokens = left["pd"].get("token_ids")
    right_tokens = right["pd"].get("token_ids")
    if left_tokens is not None and right_tokens is not None:
        return left_tokens == right_tokens
    return left["pd"]["text"] == right["pd"]["text"]


def _has_monolithic(*groups: list[dict]) -> bool:
    return all("monolithic" in row for group in groups for row in group)


def paired_summary(observe_rows: list[dict], inject_rows: list[dict]) -> dict:
    if [row["record_id"] for row in observe_rows] != [
        row["record_id"] for row in inject_rows
    ]:
        raise ValueError("benchmark record order differs")
    if len(observe_rows) != len(inject_rows):
        raise ValueError("benchmark request counts differ")

    output_matches = [
        _same_output(left, right)
        for left, right in zip(observe_rows, inject_rows, strict=True)
    ]
    result = {
        "requests": len(observe_rows),
        "inject_equals_observe_pd": sum(output_matches),
        "pd_mismatch_ordinals": [
            index for index, matches in enumerate(output_matches) if not matches
        ],
    }
    for metric in ("ttft_ms", "total_ms"):
        raw = [
            right["pd"][metric] - left["pd"][metric]
            for left, right in zip(observe_rows, inject_rows, strict=True)
        ]
        # The monolithic endpoint is replayed in both sequential runs.  This
        # difference-in-differences removes host/run drift shared with that
        # endpoint, while retaining the same request as the cluster unit.
        result[f"inject_minus_observe_{metric}"] = {
            "mean_ms": statistics.fmean(raw),
            "bootstrap_95ci_ms": bootstrap_mean_ci(raw),
            "inject_faster": sum(value < 0 for value in raw),
            "same": sum(value == 0 for value in raw),
            "inject_slower": sum(value > 0 for value in raw),
        }
        if _has_monolithic(observe_rows, inject_rows):
            adjusted = [
                (right["pd"][metric] - right["monolithic"][metric])
                - (left["pd"][metric] - left["monolithic"][metric])
                for left, right in zip(observe_rows, inject_rows, strict=True)
            ]
            result[f"difference_in_differences_{metric}"] = {
                "mean_ms": statistics.fmean(adjusted),
                "bootstrap_95ci_ms": bootstrap_mean_ci(adjusted),
            }
    observe_mono = [index for index, row in enumerate(observe_rows) if not row["outputs_equal"]]
    inject_mono = [index for index, row in enumerate(inject_rows) if not row["outputs_equal"]]
    if _has_monolithic(observe_rows, inject_rows):
        result.update(
            observe_monolithic_mismatch_ordinals=observe_mono,
            inject_monolithic_mismatch_ordinals=inject_mono,
            same_monolithic_mismatch_set=observe_mono == inject_mono,
        )
    else:
        result.update(
            observe_reference_mismatch_ordinals=observe_mono,
            inject_reference_mismatch_ordinals=inject_mono,
            same_reference_mismatch_set=observe_mono == inject_mono,
        )
    return result


def sandwich_summary(
    observe_before: list[dict],
    inject_rows: list[dict],
    observe_after: list[dict],
) -> dict:
    record_ids = [row["record_id"] for row in inject_rows]
    if [row["record_id"] for row in observe_before] != record_ids or [
        row["record_id"] for row in observe_after
    ] != record_ids:
        raise ValueError("sandwich benchmark record order differs")
    exact = [
        _same_output(before, inject) and _same_output(inject, after)
        for before, inject, after in zip(
            observe_before, inject_rows, observe_after, strict=True
        )
    ]
    monolithic_exact = [
        before["outputs_equal"] and inject["outputs_equal"] and after["outputs_equal"]
        for before, inject, after in zip(
            observe_before, inject_rows, observe_after, strict=True
        )
    ]
    result = {
        "requests": len(record_ids),
        "all_three_pd_outputs_equal": sum(exact),
        "pd_mismatch_ordinals": [
            index for index, matches in enumerate(exact) if not matches
        ],
        "all_three_equal_reference": sum(monolithic_exact),
        "reference_mismatch_ordinals": [
            index for index, matches in enumerate(monolithic_exact) if not matches
        ],
    }
    if _has_monolithic(observe_before, inject_rows, observe_after):
        result["all_three_equal_monolithic"] = sum(monolithic_exact)
        result["monolithic_mismatch_ordinals"] = result[
            "reference_mismatch_ordinals"
        ]
    for metric in ("ttft_ms", "total_ms"):
        delta = []
        adjusted = []
        drift = []
        for before, inject, after in zip(
            observe_before, inject_rows, observe_after, strict=True
        ):
            baseline = (before["pd"][metric] + after["pd"][metric]) / 2
            delta.append(inject["pd"][metric] - baseline)
            if _has_monolithic(observe_before, inject_rows, observe_after):
                before_gap = before["pd"][metric] - before["monolithic"][metric]
                inject_gap = inject["pd"][metric] - inject["monolithic"][metric]
                after_gap = after["pd"][metric] - after["monolithic"][metric]
                adjusted.append(inject_gap - (before_gap + after_gap) / 2)
            drift.append(after["pd"][metric] - before["pd"][metric])
        result[f"inject_minus_sandwich_observe_{metric}"] = {
            "mean_ms": statistics.fmean(delta),
            "bootstrap_95ci_ms": bootstrap_mean_ci(delta),
            "inject_faster": sum(value < 0 for value in delta),
            "same": sum(value == 0 for value in delta),
            "inject_slower": sum(value > 0 for value in delta),
        }
        if adjusted:
            result[f"sandwich_difference_in_differences_{metric}"] = {
                "mean_ms": statistics.fmean(adjusted),
                "bootstrap_95ci_ms": bootstrap_mean_ci(adjusted),
            }
        result[f"observe_after_minus_before_{metric}"] = {
            "mean_ms": statistics.fmean(drift),
            "bootstrap_95ci_ms": bootstrap_mean_ci(drift),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe", required=True)
    parser.add_argument("--inject", required=True)
    parser.add_argument(
        "--observe-after",
        help="optional second observe run for an observe-inject-observe sandwich",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    observe_path = Path(args.observe).resolve()
    inject_path = Path(args.inject).resolve()
    observe = json.loads(observe_path.read_text())
    inject = json.loads(inject_path.read_text())
    summary = paired_summary(observe["rows"], inject["rows"])
    contract = (
        "separate fresh-server runs over identical greedy requests; observe and "
        "inject use the same custom proposer stack; request-bootstrap is "
        "exploratory because mode order is not randomized"
    )
    observe_after_path = None
    if args.observe_after:
        observe_after_path = Path(args.observe_after).resolve()
        observe_after = json.loads(observe_after_path.read_text())
        summary = sandwich_summary(
            observe["rows"], inject["rows"], observe_after["rows"]
        )
        contract = (
            "fresh-server observe-inject-observe sandwich over identical greedy "
            "requests; treatment is compared with the per-request mean of its "
            "two surrounding baselines"
        )
    result = {
        "contract": contract,
        "observe": str(observe_path),
        "inject": str(inject_path),
        "observe_after": str(observe_after_path) if observe_after_path else None,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
