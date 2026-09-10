from experiments.lossless_pd.lmcache_pd.analyze_oracle_ceiling_gate import analyze


def _benchmark(total: float, token_ids: list[int]):
    return {
        "status": "completed",
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
        ],
    }


def test_oracle_ceiling_requires_exactness_and_speed():
    result = analyze(
        _benchmark(100.0, [1, 2]),
        _benchmark(80.0, [1, 3]),
        _benchmark(100.0, [1, 2]),
        draft_rows=[{"event": "online_verify_feedback", "accepted_injected_suffix": 4}],
        proxy_rows=[],
    )
    assert result["metrics"]["total_speedup"] == 1.25
    assert result["gates"]["same_stack_token_id_equality"] is False
    assert result["passed"] is False


def test_oracle_ceiling_counts_canonical_replay():
    result = analyze(
        _benchmark(100.0, [1, 2]),
        _benchmark(90.0, [1, 2]),
        _benchmark(100.0, [1, 2]),
        draft_rows=[{"event": "online_verify_feedback", "accepted_injected_suffix": 2}],
        proxy_rows=[{"event": "canonical_replay", "replay_ms": 12.5}],
    )
    assert result["metrics"]["canonical_replays"] == 1
    assert result["metrics"]["canonical_replay_mean_ms"] == 12.5
    assert result["passed"] is True
