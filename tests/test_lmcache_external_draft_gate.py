from experiments.lossless_pd.lmcache_pd.analyze_external_draft_gate import analyze


def _benchmark(total: float, token_ids: list[int]):
    return {
        "rows": [
            {
                "record_id": "r0",
                "outputs_equal": True,
                "pd": {
                    "token_ids": token_ids,
                    "text": "x",
                    "ttft_ms": 5.0,
                    "total_ms": total,
                },
            }
        ]
    }


def test_external_gate_stops_when_speedup_misses_fixed_threshold():
    result = analyze(
        _benchmark(100.0, [1, 2]),
        _benchmark(95.0, [1, 2]),
        _benchmark(100.0, [1, 2]),
        proxy_rows=[
            {"event": "external_draft_submitted", "seed_branch_hit": True},
            {"event": "decoder_dispatch", "full_ready_at_dispatch": False},
        ],
        draft_rows=[
            {"event": "online_draft_handoff"},
            {"event": "online_verify_feedback", "accepted_injected_suffix": 4},
        ],
    )
    assert result["metrics"]["mean_accepted_injected_suffix"] == 4
    assert result["gates"]["speedup_at_least_threshold"] is False
    assert result["passed"] is False
    assert result["decision"] == "stop_external_portfolio_as_iclr_mainline"
