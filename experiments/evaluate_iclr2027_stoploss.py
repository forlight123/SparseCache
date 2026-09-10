"""Evaluate pre-registered SparseCache-PD continuation and stop-loss gates.

The evaluator distinguishes a failed mature experiment from missing evidence.
Only the former can stop the research line.  This prevents a 64-request proxy
or a task-contaminated model screen from being promoted into a paper claim or
misread as a definitive negative result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
MISSING = "MISSING"


def read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def gate(status: str, reason: str, **metrics: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, "metrics": metrics}


def method_contract_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    contract = evidence.get("method_contract", {})
    required = (
        "token_sparse_anchor",
        "draft_hidden_until_target_verify",
        "full_precision_target_kv_eventually_transferred",
        "single_immutable_target_verifier",
        "layer_ready_exact_target",
    )
    missing = [name for name in required if name not in contract]
    false = [name for name in required if contract.get(name) is False]
    if missing:
        return gate(MISSING, "method contract is incomplete", missing=missing)
    if false:
        return gate(FAIL, "lossless structural contract is violated", false=false)
    return gate(PASS, "lossless structural contract is frozen", required=list(required))


def layer_ready_gate(
    config: dict[str, Any], evidence: dict[str, Any], base: Path
) -> dict[str, Any]:
    item = evidence.get("layer_ready_verifier")
    if not item:
        return gate(MISSING, "no all-layer verifier result")
    result = read(resolve(base, item["result"]))
    summary = result.get("summary", result)
    threshold = config["thresholds"]["layer_ready_verifier"]
    requests = int(summary.get("requests", 0))
    serial = float(summary.get("mean_serial_wall_ms", math.nan))
    streamed = float(summary.get("mean_streamed_wall_ms", math.nan))
    speedup = serial / streamed
    ci = summary.get("paired_ci95_ms", [math.nan, math.nan])
    equal = bool(summary.get("serial_streamed_bitwise_equal", False))
    enough = requests >= threshold["min_requests"]
    passed = (
        enough
        and speedup >= threshold["min_copy_verify_speedup"]
        and float(ci[0]) > threshold["min_saving_ci_lower_ms"]
        and (equal or not threshold["require_bitwise_same_work"])
    )
    metrics = {
        "requests": requests,
        "copy_verify_speedup": speedup,
        "saving_ci95_ms": ci,
        "same_work_bitwise_equal": equal,
        "threshold": threshold,
    }
    if not enough:
        return gate(MISSING, "all-layer verifier sample is too small", **metrics)
    if not passed:
        return gate(
            FAIL, "layer-ready verification misses its mechanism gate", **metrics
        )
    return gate(
        PASS, "layer-ready exact verification clears its mechanism gate", **metrics
    )


def proposer_gate(
    config: dict[str, Any], evidence: dict[str, Any], base: Path
) -> dict[str, Any]:
    threshold = config["thresholds"]["task_unseen_proposer"]
    eligible = []
    excluded = []
    for item in evidence.get("proposer_evaluations", []):
        reasons = []
        if not item.get("task_unseen", False):
            reasons.append("task is not unseen")
        if (
            item.get("natural_output_min_tokens", 0)
            < threshold["min_natural_output_tokens"]
        ):
            reasons.append("natural output horizon is too short")
        if item.get("proposal_horizon", 0) < threshold["min_proposal_horizon"]:
            reasons.append("proposal horizon is too short")
        if reasons:
            excluded.append({"task": item.get("task"), "reasons": reasons})
            continue
        document = read(resolve(base, item["result"]))
        cell = document["cells"][item["cell"]]
        eligible.append(
            {
                "task": item["task"],
                "requests": int(cell["requests"]),
                "mean_accepted": float(cell["mean_accepted"]),
            }
        )

    tasks = {item["task"] for item in eligible}
    requests = sum(item["requests"] for item in eligible)
    weighted = (
        sum(item["requests"] * item["mean_accepted"] for item in eligible) / requests
        if requests
        else None
    )
    enough = (
        len(tasks) >= threshold["min_tasks"]
        and requests >= threshold["min_total_requests"]
    )
    budget = evidence.get("budget", {})
    exhausted = (
        budget.get("proposer_architecture_revisions", 0)
        >= config["caps"]["max_proposer_architecture_revisions"]
        or budget.get("proposer_gpu_hours", 0)
        >= config["caps"]["max_proposer_gpu_hours"]
    )
    metrics = {
        "eligible": eligible,
        "excluded": excluded,
        "eligible_tasks": len(tasks),
        "eligible_requests": requests,
        "weighted_mean_accepted": weighted,
        "budget_exhausted": exhausted,
        "threshold": threshold,
    }
    if not enough:
        return gate(MISSING, "task-unseen proposer evidence is incomplete", **metrics)
    if weighted is not None and weighted >= threshold["min_mean_accepted"]:
        return gate(PASS, "task-unseen proposer clears the acceptance gate", **metrics)
    if exhausted:
        return gate(
            FAIL, "proposer misses acceptance after its frozen budget", **metrics
        )
    return gate(
        MISSING, "proposer is below target but its frozen budget remains", **metrics
    )


def integrated_cell(
    summary: dict[str, Any], threshold: dict[str, Any]
) -> dict[str, Any]:
    baseline = float(summary["mean_baseline_same_progress_ms"])
    speculative = float(summary["mean_speculative_same_progress_ms"])
    speedup = baseline / speculative
    saving_ci = summary["same_progress_saving_ci95_request_bootstrap_ms"]
    ttft_ci = summary["first_commit_delta_ci95_request_bootstrap_ms"]
    baseline_bytes = float(summary["mean_baseline_bytes"])
    speculative_bytes = float(summary["mean_speculative_bytes"])
    byte_error = abs(baseline_bytes - speculative_bytes) / baseline_bytes
    metrics = {
        "requests": int(summary["requests"]),
        "speedup": speedup,
        "saving_ci95_ms": saving_ci,
        "first_commit_delta_ci95_ms": ttft_ci,
        "output_mismatch_requests": int(summary["committed_output_mismatch_requests"]),
        "byte_relative_error": byte_error,
        "mean_draft_overrun_ms": summary.get("mean_draft_overrun_ms"),
        "target_start_mode": summary.get("target_start_mode"),
    }
    metrics["mature"] = metrics["requests"] >= threshold["min_requests_per_cell"]
    metrics["passed"] = (
        metrics["mature"]
        and metrics["target_start_mode"] == "layer_ready"
        and speedup >= threshold["min_equal_progress_speedup"]
        and float(saving_ci[0]) > threshold["min_saving_ci_lower_ms"]
        and float(ttft_ci[1]) <= threshold["max_first_commit_ci_upper_ms"]
        and metrics["output_mismatch_requests"]
        <= threshold["max_output_mismatch_requests"]
        and byte_error <= threshold["max_byte_relative_error"]
        and (
            metrics["mean_draft_overrun_ms"] is not None
            and float(metrics["mean_draft_overrun_ms"])
            <= threshold["max_mean_draft_overrun_ms"]
        )
    )
    return metrics


def integrated_gate(
    config: dict[str, Any], evidence: dict[str, Any], base: Path
) -> dict[str, Any]:
    threshold = config["thresholds"]["integrated_proxy"]
    cells = {}
    for item in evidence.get("integrated_proxy_runs", []):
        document = read(resolve(base, item["result"]))
        summary = document.get("summary", document)
        bandwidth = float(summary["gbps_decimal_bits_per_second"])
        cells[bandwidth] = integrated_cell(summary, threshold)

    required = [float(value) for value in threshold["required_bandwidth_gbps"]]
    missing = [
        value for value in required if value not in cells or not cells[value]["mature"]
    ]
    metrics = {
        "cells": cells,
        "missing_or_immature_bandwidths": missing,
        "threshold": threshold,
    }
    if missing:
        return gate(MISSING, "integrated proxy matrix is incomplete", **metrics)
    failed = [value for value in required if not cells[value]["passed"]]
    if failed:
        return gate(
            FAIL,
            "mature integrated proxy cells miss hard gates",
            failed=failed,
            **metrics,
        )
    return gate(PASS, "integrated proxy clears all required link regimes", **metrics)


def live_gate(
    config: dict[str, Any], evidence: dict[str, Any], base: Path
) -> dict[str, Any]:
    del base
    threshold = config["thresholds"]["live_two_node"]
    cells = evidence.get("live_two_node_runs", [])
    eligible = [
        item
        for item in cells
        if item.get("physical_two_node")
        and item.get("requests", 0) >= threshold["min_requests"]
    ]
    if not eligible:
        return gate(
            MISSING,
            "no mature physical two-node run",
            supplied_runs=len(cells),
            threshold=threshold,
        )
    passed = [
        item
        for item in eligible
        if item["speedup"] >= threshold["min_end_to_end_speedup"]
        and item["saving_ci95_ms"][0] > threshold["min_saving_ci_lower_ms"]
        and item["output_mismatch_requests"]
        <= threshold["max_output_mismatch_requests"]
    ]
    if passed:
        return gate(
            PASS,
            "physical two-node deployment clears the final systems gate",
            runs=passed,
        )
    budget = evidence.get("budget", {})
    exhausted = (
        budget.get("live_optimization_rounds", 0)
        >= config["caps"]["max_live_optimization_rounds"]
    )
    if exhausted:
        return gate(
            FAIL,
            "two-node deployment misses speedup after its frozen budget",
            runs=eligible,
        )
    return gate(
        MISSING,
        "two-node result is below target but one optimization round remains",
        runs=eligible,
    )


def breadth_gate(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    threshold = config["thresholds"]["paper_breadth"]
    breadth = evidence.get("paper_breadth", {})
    models = int(breadth.get("target_models", 0))
    tasks = int(breadth.get("task_unseen_tasks", 0))
    if (
        models < threshold["min_target_models"]
        or tasks < threshold["min_task_unseen_tasks"]
    ):
        return gate(
            MISSING,
            "paper breadth has not been reached",
            target_models=models,
            task_unseen_tasks=tasks,
            threshold=threshold,
        )
    return gate(
        PASS,
        "paper breadth requirement is met",
        target_models=models,
        task_unseen_tasks=tasks,
    )


def evaluate(
    config: dict[str, Any], evidence: dict[str, Any], evidence_base: Path
) -> dict[str, Any]:
    gates = {
        "method_contract": method_contract_gate(evidence),
        "layer_ready_verifier": layer_ready_gate(config, evidence, evidence_base),
        "task_unseen_proposer": proposer_gate(config, evidence, evidence_base),
        "integrated_proxy": integrated_gate(config, evidence, evidence_base),
        "live_two_node": live_gate(config, evidence, evidence_base),
        "paper_breadth": breadth_gate(config, evidence),
    }
    failures = [name for name, item in gates.items() if item["status"] == FAIL]
    missing = [name for name, item in gates.items() if item["status"] == MISSING]
    if failures:
        decision = "STOP_OR_REDESIGN"
    elif missing:
        decision = "ITERATE"
    else:
        decision = "GO_PAPER_SCALE"
    return {
        "policy_version": config["version"],
        "round_id": evidence.get("round_id"),
        "decision": decision,
        "failed_gates": failures,
        "missing_gates": missing,
        "gates": gates,
        "decision_rule": (
            "STOP_OR_REDESIGN only follows a failed mature gate; incomplete or "
            "ineligible evidence yields ITERATE; GO requires every gate to pass"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output")
    parser.add_argument("--fail-on-stop", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    evidence_path = Path(args.evidence).resolve()
    result = evaluate(read(config_path), read(evidence_path), evidence_path.parent)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise ValueError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.fail_on_stop and result["decision"] == "STOP_OR_REDESIGN":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
