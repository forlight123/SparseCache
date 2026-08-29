from copy import deepcopy

from experiments.aggregate_progressive_pd_scheduling_live import (
    aggregate_scheduling_run,
)
from tests.test_aggregate_progressive_pd_live import _fixture


def _scheduling_fixture():
    results, metadata, schedulers, attention, link = _fixture()
    source_results = [row for row in results if row["arm"] != "baseline"]
    source_link = [row for row in link if row["mode"] == "progressive_bundle"]
    scheduled_results = []
    scheduled_schedulers = []
    scheduled_attention = []
    scheduled_link = []
    for mode in ("sequential", "bm25"):
        request_ids = {}
        for row in source_results:
            copied = deepcopy(row)
            old_proxy = copied["proxy_request_id"]
            new_proxy = f"{old_proxy}-{mode}"
            copied["proxy_request_id"] = new_proxy
            copied["schedule_mode"] = mode
            copied["prefill_group"] = f"scheduled-prefill-{copied['request_index']}"
            copied["prefill_reused"] = not (
                mode == "sequential" and copied["arm"] == "fixed_s1"
            )
            request_ids[old_proxy] = new_proxy
            scheduled_results.append(copied)
        for rows, output in (
            (schedulers, scheduled_schedulers),
            (attention, scheduled_attention),
            (source_link, scheduled_link),
        ):
            for row in rows:
                copied = deepcopy(row)
                for old_proxy, new_proxy in request_ids.items():
                    if old_proxy in copied["request_id"]:
                        copied["request_id"] = copied["request_id"].replace(
                            old_proxy, new_proxy
                        )
                        output.append(copied)
                        break
    priorities = {
        mode: [
            {"request_index": request_index, "priority_chunks": [0, 1]}
            for request_index in range(2)
        ]
        for mode in ("sequential", "bm25")
    }
    return (
        scheduled_results,
        metadata,
        scheduled_schedulers,
        scheduled_attention,
        scheduled_link,
        priorities,
    )


def _aggregate():
    results, metadata, schedulers, attention, link, priorities = (
        _scheduling_fixture()
    )
    return aggregate_scheduling_run(
        results=results,
        metadata_rows=metadata,
        scheduler_rows=schedulers,
        attention_rows=attention,
        link_rows=link,
        priorities=priorities,
        arms=("fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        protected_prefix_tokens=0,
        protected_suffix_tokens=0,
        priority_chunk_tokens=1,
        bootstrap_samples=100,
        seed=7,
    )


def test_scheduling_aggregation_passes_complete_paired_matrix() -> None:
    summary, paired, _ = _aggregate()

    assert summary["status"] == "passed"
    assert summary["gates"]["overall"]["passed"]
    assert summary["gates"]["single_shared_prefill_across_schedules"][
        "passed"
    ]
    assert set(paired) == {"sequential", "bm25"}
    assert summary["comparisons"]["sequential_vs_bm25"]["requests"] == 2
    for mode_summary in summary["mode_summaries"].values():
        assert mode_summary["gates"]["request_scoped_priority_schedule"]["passed"]


def test_scheduling_aggregation_rejects_cross_schedule_output_drift() -> None:
    results, metadata, schedulers, attention, link, priorities = (
        _scheduling_fixture()
    )
    drift = next(
        row
        for row in results
        if row["schedule_mode"] == "bm25" and row["request_index"] == 0
    )
    drift["token_ids"][-1] = 8

    summary, _, _ = aggregate_scheduling_run(
        results=results,
        metadata_rows=metadata,
        scheduler_rows=schedulers,
        attention_rows=attention,
        link_rows=link,
        priorities=priorities,
        arms=("fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        protected_prefix_tokens=0,
        protected_suffix_tokens=0,
        priority_chunk_tokens=1,
        bootstrap_samples=100,
        seed=7,
    )

    assert summary["status"] == "failed_gates"
    assert not summary["gates"]["cross_schedule_exact_greedy_equivalence"][
        "passed"
    ]


def test_scheduling_aggregation_rejects_independent_schedule_prefills() -> None:
    results, metadata, schedulers, attention, link, priorities = (
        _scheduling_fixture()
    )
    for row in results:
        if row["schedule_mode"] == "bm25" and row["request_index"] == 0:
            row["prefill_group"] += "-bm25"
            row["prefill_reused"] = row["arm"] == "continuous"
            row["seed_token_id"] += 1

    summary, _, _ = aggregate_scheduling_run(
        results=results,
        metadata_rows=metadata,
        scheduler_rows=schedulers,
        attention_rows=attention,
        link_rows=link,
        priorities=priorities,
        arms=("fixed_s1", "continuous"),
        expected_requests=2,
        fixed_output_tokens=3,
        protected_prefix_tokens=0,
        protected_suffix_tokens=0,
        priority_chunk_tokens=1,
        bootstrap_samples=100,
        seed=7,
    )

    gate = summary["gates"]["single_shared_prefill_across_schedules"]
    assert summary["status"] == "failed_gates"
    assert not gate["passed"]
    assert any("prefill group" in error for error in gate["errors"])
    assert any("producer seed" in error for error in gate["errors"])
