from experiments.lossless_pd.lmcache_pd.compare_layerwise_dispatch import (
    compare_benchmarks,
    summarize_physical_traces,
)


def _benchmark(latencies, texts=("a", "b")):
    rows = []
    for index, (pd_ms, mono_ms) in enumerate(latencies):
        rows.append(
            {
                "record_id": f"r{index}",
                "pd": {"text": texts[index], "ttft_ms": pd_ms, "total_ms": pd_ms},
                "monolithic": {
                    "text": texts[index],
                    "ttft_ms": mono_ms,
                    "total_ms": mono_ms,
                },
                "outputs_equal": True,
            }
        )
    return {"rows": rows}


def test_compare_benchmarks_is_paired_and_monolithic_adjusted():
    early = _benchmark([(90.0, 50.0), (110.0, 60.0)])
    full = _benchmark([(100.0, 55.0), (100.0, 55.0)])
    result = compare_benchmarks(early, full)
    assert result["early_equals_full_pd"] == 2
    assert result["early_minus_full_total_ms"]["mean_ms"] == 0
    assert result["monolithic_adjusted_total_ms"]["mean_ms"] == 0


def test_compare_benchmarks_rejects_order_drift():
    early = _benchmark([(90.0, 50.0), (110.0, 60.0)])
    full = _benchmark([(100.0, 55.0), (100.0, 55.0)])
    full["rows"].reverse()
    try:
        compare_benchmarks(early, full)
    except ValueError as error:
        assert "record order" in str(error)
    else:
        raise AssertionError("record-order drift was accepted")


def test_physical_trace_summary_counts_zero_acceptance_and_wire_bytes():
    proxy = [
        {"event": "decoder_dispatch", "pd_request_id": "p0"},
        {"event": "decoder_dispatch", "pd_request_id": "p1"},
    ]
    sender = [
        {
            "event": "layerwise_schedule",
            "pd_request_id": request_id,
            "wire_bytes": 100,
            "authoritative_bytes": 100,
            "retransmitted_bytes": 0,
        }
        for request_id in ("p0", "p1")
    ]
    receiver = []
    drafts = []
    for index, request_id in enumerate(("p0", "p1")):
        base = index * 1_000_000
        receiver.extend(
            [
                {
                    "event": "receiver_anchor_ready",
                    "pd_request_id": request_id,
                    "receiver_received_ns": base,
                },
                {
                    "event": "receiver_layer_ready",
                    "pd_request_id": request_id,
                    "receiver_received_ns": base + 10_000_000,
                },
            ]
        )
        drafts.append(
            {
                "event": "live_sparse_draft",
                "pd_request_id": request_id,
                "draft_finished_ns": base + 5_000_000,
                "actual_fraction": 0.1,
                "total_gpu_ms": 4.0,
                "wall_ms": 5.0,
            }
        )
    drafts.extend(
        [
            {"event": "online_draft_handoff", "pd_request_id": "p0"},
            {
                "event": "online_verify_feedback",
                "pd_request_id": "p0",
                "accepted_injected_suffix": 3,
            },
        ]
    )
    result = summarize_physical_traces(
        proxy_rows=proxy,
        sender_rows=sender,
        receiver_rows=receiver,
        draft_rows=drafts,
    )
    assert result["requests"] == 2
    assert result["draft_finished_before_full"] == 2
    assert result["mean_accepted_injected_suffix"] == 1.5
    assert result["positive_acceptance_requests"] == 1
    assert result["mean_draft_gpu_ms"] == 4.0
    assert result["mean_draft_wall_ms"] == 5.0
    assert result["retransmitted_bytes"] == 0
