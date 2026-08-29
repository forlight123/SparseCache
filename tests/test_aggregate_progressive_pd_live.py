from pathlib import Path
from typing import Any

from experiments.aggregate_progressive_pd_live import (
    aggregate_live_run,
    write_live_artifacts,
)


def _fixture() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    results = []
    metadata = []
    schedulers = []
    attention = []
    link = []
    for request_index in range(2):
        metadata.append(
            {
                "request_index": request_index,
                "dataset": "ruler_qa2",
                "answers": ["answer"],
                "all_classes": None,
                "fixed_output_horizon": True,
                "output_tokens": 3,
            }
        )
        for arm, latency in (
            ("baseline", 12.0),
            ("fixed_s1", 10.0),
            ("continuous", 8.0),
        ):
            proxy_id = f"proxy{request_index}{arm}"
            arm_index = ("baseline", "fixed_s1", "continuous").index(arm)
            results.append(
                {
                    "request_index": request_index,
                    "source_request_index": request_index,
                    "arm": arm,
                    "runtime_mode": "baseline" if arm == "baseline" else "progressive",
                    "warmup": False,
                    "header_ms": 1.0,
                    "ttft_ms": latency / 2 + 1,
                    "decode_ttft_ms": latency / 2,
                    "completion_ms": latency + 1,
                    "decode_completion_ms": latency,
                    "token_ids": [1, 2, 9],
                    "token_arrival_ms": [4.0, 6.0, 8.0],
                    "text": "answer",
                    "proxy_request_id": proxy_id,
                    "prefill_group": f"paired-prefill-{request_index}",
                    "prefill_reused": arm_index != 0,
                    "seed_token_id": 1,
                }
            )
            request_id = f"cmpl-{proxy_id}-0"
            if arm == "baseline":
                link.append(
                    {
                        "request_id": request_id,
                        "mode": "monolithic",
                        "bundle_index": 0,
                        "completed_fraction": 1.0,
                        "logical_tokens": 2,
                        "logical_payload_bytes": 2_000_000.0,
                        "token_ranges": [[0, 2]],
                        "link_gbps": 1.0,
                        "bytes_per_token": 1_000_000.0,
                        "submitted_at_s": 0.0,
                        "not_before_s": 0.016,
                        "reported_at_s": 0.016,
                        "modeled_wire_ms": 16.0,
                        "succeeded": True,
                    }
                )
                continue
            for bundle_index in range(2):
                submitted = bundle_index * 0.008
                link.append(
                    {
                        "request_id": request_id,
                        "mode": "progressive_bundle",
                        "bundle_index": bundle_index,
                        "completed_fraction": (bundle_index + 1) / 2,
                        "logical_tokens": 1,
                        "logical_payload_bytes": 1_000_000.0,
                        "token_ranges": [[bundle_index, bundle_index + 1]],
                        "link_gbps": 1.0,
                        "bytes_per_token": 1_000_000.0,
                        "submitted_at_s": submitted,
                        "not_before_s": submitted + 0.008,
                        "reported_at_s": submitted + 0.008,
                        "modeled_wire_ms": 8.0,
                        "succeeded": True,
                    }
                )
            schedulers.append(
                {
                    "request_id": request_id,
                    "seed_token_id": 1,
                    "start_fraction": 0.25,
                    "visibility_mode": arm,
                    "max_draft_tokens": 1,
                    "draft_token_ids": [2],
                    "draft_completed_at_ns": [20],
                    "draft_tokens": 1,
                    "accepted_tokens": 1,
                    "verified_output_token_ids": [2, 9],
                    "started_at_ns": 10,
                    "last_draft_at_ns": 20,
                    "full_arrival_at_ns": 30,
                    "verify_started_at_ns": 40,
                    "verify_completed_at_ns": 50,
                    "transition_ns": 40,
                    "verify_ns": 10,
                }
            )
            for layer in ("layer.0", "layer.1"):
                attention.append(
                    {
                        "request_id": request_id,
                        "layer": layer,
                        "decode_step": 0,
                        "visible_fraction": 0.25,
                        "visibility_mode": arm,
                        "visible_token_ranges": [[0, 256]],
                        "candidate_pages": 4,
                        "visible_pages": 1,
                    }
                )
    return results, metadata, schedulers, attention, link


def test_live_aggregation_passes_complete_exact_three_arm_run() -> None:
    results, metadata, schedulers, attention, link = _fixture()

    summary, paired = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=200,
        seed=7,
    )

    assert summary["status"] == "passed"
    assert summary["gates"]["overall"]["passed"]
    assert len(paired) == 2
    comparison = summary["comparisons"]["baseline_vs_continuous"]
    assert comparison["token_equality_rate"] == 1.0
    assert comparison["decode_completion_gain_ms"]["mean"] == 4.0
    assert comparison["latency_model"]["predicted_gain_ms"]["mean"] > 1.9
    assert summary["gates"]["latency_model_telemetry"]["passed"]
    assert summary["arm_summaries"]["continuous"]["draft"]["acceptance_rate"] == 1


def test_live_aggregation_excludes_acceptance_after_natural_stop() -> None:
    results, metadata, schedulers, attention, link = _fixture()

    summary, _ = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=100,
        seed=7,
        stop_token_ids=(1,),
    )

    draft = summary["arm_summaries"]["continuous"]["draft"]
    assert summary["stop_token_ids"] == [1]
    assert draft["acceptance_rate"] == 1
    assert draft["effective_acceptance_rate"] == 0
    assert draft["useful_draft_tokens"]["mean"] == 0
    assert draft["accepted_tokens_after_first_stop"]["mean"] == 1


def test_live_aggregation_fails_exact_output_gate_on_drift() -> None:
    results, metadata, schedulers, attention, link = _fixture()
    results[0]["token_ids"] = [1, 2, 8]

    summary, _ = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=100,
        seed=7,
    )

    assert summary["status"] == "failed_gates"
    assert not summary["gates"]["exact_greedy_output_equivalence"]["passed"]


def test_live_aggregation_rejects_unfair_baseline_link() -> None:
    results, metadata, schedulers, attention, link = _fixture()
    baseline = next(row for row in link if row["mode"] == "monolithic")
    baseline["link_gbps"] = 50.0
    baseline["modeled_wire_ms"] = 0.32

    summary, _ = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=100,
        seed=7,
    )

    assert summary["status"] == "failed_gates"
    assert not summary["gates"]["fair_controlled_link"]["passed"]


def test_live_aggregation_rejects_unprotected_question_suffix() -> None:
    results, metadata, schedulers, attention, link = _fixture()

    summary, _ = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=100,
        seed=7,
        protected_prefix_tokens=1,
        protected_suffix_tokens=1,
    )

    assert summary["status"] == "failed_gates"
    assert not summary["gates"]["protected_anchor_arrival"]["passed"]
    assert any(
        "protected suffix not in S1"
        for item in summary["gates"]["protected_anchor_arrival"]["errors"]
    )


def test_live_aggregation_rejects_missing_break_even_telemetry() -> None:
    results, metadata, schedulers, attention, link = _fixture()
    results[0].pop("token_arrival_ms")

    summary, _ = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=100,
        seed=7,
    )

    assert summary["status"] == "failed_gates"
    assert not summary["gates"]["request_protocol"]["passed"]
    assert not summary["gates"]["latency_model_telemetry"]["passed"]


def test_live_aggregation_proves_request_scoped_priority_schedule() -> None:
    results, metadata, schedulers, attention, link = _fixture()
    priorities = [
        {"request_index": request_index, "priority_chunks": [0, 1]}
        for request_index in range(2)
    ]

    summary, _ = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=100,
        seed=7,
        priority_rows=priorities,
        priority_chunk_tokens=1,
    )

    assert summary["status"] == "passed"
    gate = summary["gates"]["request_scoped_priority_schedule"]
    assert gate["passed"]
    assert gate["applicable"]


def test_live_aggregation_rejects_link_order_that_ignores_priority() -> None:
    results, metadata, schedulers, attention, link = _fixture()
    priorities = [
        {"request_index": request_index, "priority_chunks": [1, 0]}
        for request_index in range(2)
    ]

    summary, _ = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=100,
        seed=7,
        priority_rows=priorities,
        priority_chunk_tokens=1,
    )

    assert summary["status"] == "failed_gates"
    gate = summary["gates"]["request_scoped_priority_schedule"]
    assert not gate["passed"]
    assert any("sidecar slice" in error for error in gate["errors"])


def test_live_artifacts_are_paper_ready(tmp_path: Path) -> None:
    results, metadata, schedulers, attention, link = _fixture()
    summary, paired = aggregate_live_run(
        results,
        metadata,
        schedulers,
        attention,
        link,
        arms=("baseline", "fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        bootstrap_samples=100,
        seed=7,
    )

    output = tmp_path / "aggregate"
    write_live_artifacts(output, summary, paired)

    assert {path.name for path in output.iterdir()} == {
        "summary.json",
        "summary.md",
        "paper_table.csv",
        "paired_rows.jsonl",
    }
    assert "baseline_vs_continuous" in (output / "summary.md").read_text()
