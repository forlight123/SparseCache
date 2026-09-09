from experiments.lossless_pd.lmcache_pd.analyze_trace import reconstruct, summarize


def _trace(request_id="request-1"):
    return [
        {
            "event": "nixl_write",
            "request_id": request_id,
            "phase": "anchor",
            "started_ns": 13_000_000,
            "finished_ns": 15_000_000,
            "write_ms": 2.0,
            "seed_record": {
                "seed_token_id": 42,
                "seed_sampled_ns": 1_000_000,
            },
        },
        {
            "event": "gather_submit",
            "request_id": request_id,
            "total_chunks": 10,
            "total_bytes": 1000,
            "store_started_ns": 0,
            "store_returned_ns": 40_000_000,
            "phases": [
                {
                    "phase": "anchor",
                    "chunks": 1,
                    "bytes": 100,
                    "gather_done_ns": 10_000_000,
                    "submitted_ns": 12_000_000,
                    "gather_ms": 10.0,
                },
                {
                    "phase": "residual",
                    "chunks": 9,
                    "bytes": 900,
                    "gather_done_ns": 35_000_000,
                    "submitted_ns": 38_000_000,
                    "gather_ms": 20.0,
                },
            ],
        },
        {
            "event": "nixl_write",
            "request_id": request_id,
            "phase": "residual",
            "started_ns": 39_000_000,
            "finished_ns": 44_000_000,
            "write_ms": 5.0,
        },
    ]


def test_reconstructs_lossless_anchor_window():
    row = reconstruct(_trace())[0]
    assert row["anchor_fraction_bytes"] == 0.1
    assert row["anchor_ready_ms"] == 15.0
    assert row["full_ready_ms"] == 44.0
    assert row["draft_window_ms"] == 29.0
    assert row["seed_token_id"] == 42
    assert row["seed_to_anchor_ready_ms"] == 14.0


def test_summarizes_reconstructed_requests():
    result = summarize(reconstruct(_trace()))
    assert result["requests"] == 1
    assert result["draft_window_ms"]["mean"] == 29.0
