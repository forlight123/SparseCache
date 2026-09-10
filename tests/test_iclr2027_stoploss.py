import json

from experiments.evaluate_iclr2027_stoploss import FAIL, MISSING, PASS, evaluate


def config():
    return {
        "version": "test",
        "thresholds": {
            "layer_ready_verifier": {
                "min_requests": 2,
                "min_copy_verify_speedup": 1.25,
                "min_saving_ci_lower_ms": 0,
                "require_bitwise_same_work": True,
            },
            "task_unseen_proposer": {
                "min_tasks": 2,
                "min_total_requests": 4,
                "min_natural_output_tokens": 32,
                "min_proposal_horizon": 8,
                "min_mean_accepted": 3.5,
            },
            "integrated_proxy": {
                "required_bandwidth_gbps": [25, 50],
                "min_requests_per_cell": 2,
                "min_equal_progress_speedup": 1.1,
                "min_saving_ci_lower_ms": 0,
                "max_first_commit_ci_upper_ms": 5,
                "max_output_mismatch_requests": 0,
                "max_byte_relative_error": 0.001,
                "max_mean_draft_overrun_ms": 0,
            },
            "live_two_node": {
                "min_requests": 2,
                "min_end_to_end_speedup": 1.1,
                "min_saving_ci_lower_ms": 0,
                "max_output_mismatch_requests": 0,
            },
            "paper_breadth": {
                "min_target_models": 2,
                "min_task_unseen_tasks": 2,
            },
        },
        "caps": {
            "max_proposer_architecture_revisions": 2,
            "max_proposer_gpu_hours": 200,
            "max_live_optimization_rounds": 2,
        },
    }


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.name


def base_evidence(tmp_path):
    verifier = {
        "requests": 2,
        "mean_serial_wall_ms": 50,
        "mean_streamed_wall_ms": 30,
        "paired_ci95_ms": [10, 30],
        "serial_streamed_bitwise_equal": True,
    }
    eval_result = {"cells": {"exact": {"requests": 2, "mean_accepted": 4.0}}}
    integrated = {
        "requests": 2,
        "mean_baseline_same_progress_ms": 120,
        "mean_speculative_same_progress_ms": 100,
        "same_progress_saving_ci95_request_bootstrap_ms": [10, 30],
        "first_commit_delta_ci95_request_bootstrap_ms": [-1, 2],
        "committed_output_mismatch_requests": 0,
        "mean_baseline_bytes": 1000,
        "mean_speculative_bytes": 1000,
        "mean_draft_overrun_ms": 0,
        "target_start_mode": "layer_ready",
    }
    write(tmp_path / "verifier.json", verifier)
    write(tmp_path / "task_a.json", eval_result)
    write(tmp_path / "task_b.json", eval_result)
    write(tmp_path / "run25.json", {**integrated, "gbps_decimal_bits_per_second": 25})
    write(tmp_path / "run50.json", {**integrated, "gbps_decimal_bits_per_second": 50})
    return {
        "round_id": "test",
        "method_contract": {
            "token_sparse_anchor": True,
            "draft_hidden_until_target_verify": True,
            "full_precision_target_kv_eventually_transferred": True,
            "single_immutable_target_verifier": True,
            "layer_ready_exact_target": True,
        },
        "layer_ready_verifier": {"result": "verifier.json"},
        "proposer_evaluations": [
            {
                "task": task,
                "result": f"task_{task.lower()}.json",
                "cell": "exact",
                "proposal_horizon": 8,
                "natural_output_min_tokens": 32,
                "task_unseen": True,
            }
            for task in ("A", "B")
        ],
        "integrated_proxy_runs": [
            {"result": "run25.json"},
            {"result": "run50.json"},
        ],
        "live_two_node_runs": [
            {
                "physical_two_node": True,
                "requests": 2,
                "speedup": 1.2,
                "saving_ci95_ms": [1, 5],
                "output_mismatch_requests": 0,
            }
        ],
        "paper_breadth": {"target_models": 2, "task_unseen_tasks": 2},
        "budget": {
            "proposer_architecture_revisions": 0,
            "proposer_gpu_hours": 0,
            "live_optimization_rounds": 0,
        },
    }


def test_all_mature_gates_can_pass(tmp_path):
    evidence = base_evidence(tmp_path)
    result = evaluate(config(), evidence, tmp_path)

    assert result["decision"] == "GO_PAPER_SCALE"
    assert all(item["status"] == PASS for item in result["gates"].values())


def test_missing_evidence_does_not_trigger_stop(tmp_path):
    evidence = base_evidence(tmp_path)
    evidence["proposer_evaluations"] = []
    evidence["integrated_proxy_runs"] = []

    result = evaluate(config(), evidence, tmp_path)

    assert result["decision"] == "ITERATE"
    assert result["gates"]["task_unseen_proposer"]["status"] == MISSING
    assert result["gates"]["integrated_proxy"]["status"] == MISSING


def test_exhausted_bad_proposer_triggers_stop(tmp_path):
    evidence = base_evidence(tmp_path)
    for name in ("task_a.json", "task_b.json"):
        write(
            tmp_path / name,
            {"cells": {"exact": {"requests": 2, "mean_accepted": 2.0}}},
        )
    evidence["budget"]["proposer_architecture_revisions"] = 2

    result = evaluate(config(), evidence, tmp_path)

    assert result["decision"] == "STOP_OR_REDESIGN"
    assert result["gates"]["task_unseen_proposer"]["status"] == FAIL


def test_mature_proxy_output_mismatch_triggers_stop(tmp_path):
    evidence = base_evidence(tmp_path)
    bad = json.loads((tmp_path / "run50.json").read_text(encoding="utf-8"))
    bad["committed_output_mismatch_requests"] = 1
    write(tmp_path / "run50.json", bad)

    result = evaluate(config(), evidence, tmp_path)

    assert result["decision"] == "STOP_OR_REDESIGN"
    assert result["gates"]["integrated_proxy"]["status"] == FAIL
