from copy import deepcopy

import pytest

from experiments.compare_layer_ready_ablation import compare


def document(mode, baseline_ms, speculative_ms, mismatched=False):
    summary = {
        "target_start_mode": mode,
        "gbps_decimal_bits_per_second": 50.0,
        "fraction": 0.1,
        "order": "priority",
        "proposal_tokens": 8,
        "verifier_semantics": "eager",
    }
    rows = []
    for record_id in ("a", "b"):
        for repeat in (0, 1):
            rows.append(
                {
                    "record_id": record_id,
                    "repeat": repeat,
                    "context": 8192,
                    "actual_fraction": 0.1,
                    "proposal_tokens": 8,
                    "progress_tokens": 4,
                    "committed_output_equal": not (mismatched and record_id == "b"),
                    "baseline": {
                        "same_progress_ms": baseline_ms,
                        "bytes_sent": 1024,
                    },
                    "speculative": {
                        "same_progress_ms": speculative_ms,
                        "bytes_sent": 1024,
                    },
                }
            )
    return {"summary": summary, "rows": rows}


def test_compare_attributes_layer_ready_effect_to_each_arm():
    result = compare(
        document("layer_ready", baseline_ms=90, speculative_ms=70),
        document("full_ready", baseline_ms=120, speculative_ms=100),
    )

    assert result["requests"] == 2
    assert result["layer_ready_hidden_ms_no_draft"]["mean_ms"] == 30
    assert result["layer_ready_hidden_ms_sparse_draft"]["mean_ms"] == 30
    assert result["difference_in_differences_ms"]["mean_ms"] == 0
    assert result[
        "mean_sparse_path_speedup_layer_ready_over_wait_full"
    ] == pytest.approx(100 / 70)


def test_compare_rejects_unpaired_protocols():
    left = document("layer_ready", baseline_ms=90, speculative_ms=70)
    right = document("full_ready", baseline_ms=120, speculative_ms=100)
    right = deepcopy(right)
    right["summary"]["proposal_tokens"] = 7

    with pytest.raises(ValueError, match="protocol differs"):
        compare(left, right)


def test_compare_reports_either_arm_output_mismatch():
    result = compare(
        document("layer_ready", baseline_ms=90, speculative_ms=70),
        document("full_ready", baseline_ms=120, speculative_ms=100, mismatched=True),
    )

    assert result["strict_output_mismatch_requests"] == 1
    assert result["strict_output_mismatch_record_ids"] == ["b"]
