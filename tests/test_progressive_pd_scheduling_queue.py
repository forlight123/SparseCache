from pathlib import Path

from experiments.build_progressive_pd_scheduling_queue import (
    DEFAULT_PACK_ROOT,
    DEFAULT_SCHEDULE_ROOT,
    build_scheduling_queue,
)


def test_scheduling_queue_pairs_modes_and_holds_out_confirmation_rows() -> None:
    queue = build_scheduling_queue(
        DEFAULT_PACK_ROOT,
        DEFAULT_SCHEDULE_ROOT,
        Path("outputs/test-live-scheduling"),
        bandwidths=(25, 50),
        selection_requests=100,
        confirmation_requests=100,
        warmup_requests=4,
        max_draft_tokens=8,
    )

    assert queue["counts"] == {
        "jobs": 4,
        "selection": 2,
        "confirmation": 2,
        "measured_conditions_per_request": 10,
    }
    assert queue["protocol"]["selection_rows"] == [0, 100]
    assert queue["protocol"]["confirmation_rows"] == [100, 200]
    assert queue["protocol"]["start_fraction"] == 0.05
    assert queue["protocol"]["tranche_fraction"] == 0.05
    selection = next(
        job for job in queue["jobs"] if job["phase"] == "scheduling_selection"
    )
    confirmation = next(
        job for job in queue["jobs"] if job["phase"] == "scheduling_confirmation"
    )
    assert selection["request_offset"] == 0
    assert confirmation["request_offset"] == 100
    assert selection["arms"] == ["fixed_s1", "continuous"]
    assert selection["proxy_args"]["start_fraction"] == 0.05
    assert selection["schedule_modes"] == [
        "sequential",
        "uniform",
        "random",
        "bm25",
        "oracle",
    ]
    assert selection["priority_sidecars"]["oracle"]["role"] == (
        "analysis_upper_bound"
    )
    assert selection["validity_requirements"]["runtime_priority_matches_sidecar"]
    assert selection["validity_requirements"][
        "single_shared_prefill_across_schedules"
    ]
    assert "one shared producer prefill" in queue["protocol"]["paired_execution"]
    assert selection["benchmark_command"].count("--priority-sidecar") == 5
    assert selection["aggregate_command"].count("--priority-sidecar") == 5
    assert selection["aggregate_command"][
        selection["aggregate_command"].index("--stop-token-ids") + 1
    ] == "128001,128008,128009"


def test_scheduling_queue_accepts_aligned_ten_percent_seed() -> None:
    queue = build_scheduling_queue(
        DEFAULT_PACK_ROOT,
        DEFAULT_SCHEDULE_ROOT,
        Path("outputs/test-live-scheduling-s10"),
        bandwidths=(25,),
        selection_requests=2,
        confirmation_requests=2,
        warmup_requests=0,
        max_draft_tokens=8,
        start_fraction=0.10,
    )

    assert queue["protocol"]["start_fraction"] == 0.10
    assert all(job["proxy_args"]["start_fraction"] == 0.10 for job in queue["jobs"])


def test_scheduling_queue_supports_one_percent_tranches() -> None:
    queue = build_scheduling_queue(
        DEFAULT_PACK_ROOT,
        DEFAULT_SCHEDULE_ROOT,
        Path("outputs/test-live-scheduling-t1"),
        bandwidths=(25,),
        selection_requests=2,
        confirmation_requests=2,
        warmup_requests=0,
        max_draft_tokens=16,
        start_fraction=0.05,
        tranche_fraction=0.01,
    )

    assert queue["protocol"]["tranche_fraction"] == 0.01
    selection = queue["jobs"][0]
    assert selection["id"].endswith("_tranche100bp")
    fractions = selection["decoder_connector_extra_config"][
        "lmcache.mp.progressive_retrieve_fractions"
    ].split(",")
    assert fractions[:5] == ["0.01", "0.02", "0.03", "0.04", "0.05"]
    assert fractions[-1] == "1"
    assert len(fractions) == 100
