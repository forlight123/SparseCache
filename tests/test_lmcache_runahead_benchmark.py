from experiments.lossless_pd.lmcache_pd.benchmark_runahead import summarize


def test_summary_uses_paired_request_deltas():
    rows = [
        {
            "baseline": {"total_ms": 100.0, "ttft_ms": 40.0},
            "runahead": {"total_ms": 80.0, "ttft_ms": 50.0},
            "outputs_equal": True,
        },
        {
            "baseline": {"total_ms": 120.0, "ttft_ms": 60.0},
            "runahead": {"total_ms": 110.0, "ttft_ms": 55.0},
            "outputs_equal": True,
        },
    ]
    result = summarize(rows)
    assert result["outputs_equal"] == 2
    assert result["runahead_minus_baseline_total_ms"]["mean"] == -15.0
    assert result["runahead_minus_baseline_ttft_ms"]["mean"] == 2.5
