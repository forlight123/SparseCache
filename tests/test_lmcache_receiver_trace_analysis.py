from experiments.lossless_pd.lmcache_pd.analyze_receiver_trace import summarize


def test_receiver_summary_counts_complete_objects_and_latency():
    rows = [
        {
            "complete": True,
            "control_plane_ms": 0.5,
            "lookup_ms": 0.1,
            "expected_keys": 4,
            "found_keys": 4,
            "expected_resident_bytes": 40,
            "resolved_bytes": 40,
            "layer_views": {
                "all_storage_aliases": True,
                "logical_bytes": 5,
                "views": 5,
                "inspection_ms": 0.1,
            },
        },
        {
            "complete": True,
            "control_plane_ms": 0.7,
            "lookup_ms": 0.2,
            "expected_keys": 2,
            "found_keys": 2,
            "expected_resident_bytes": 20,
            "resolved_bytes": 20,
            "layer_views": {
                "all_storage_aliases": True,
                "logical_bytes": 5,
                "views": 5,
                "inspection_ms": 0.2,
            },
        },
    ]
    result = summarize(rows)
    assert result["requests"] == result["complete_requests"] == 2
    assert result["complete_rate"] == 1
    assert result["control_plane_ms"]["mean"] == 0.6
    assert result["expected_keys"] == result["found_keys"] == 6
    assert result["expected_resident_bytes"] == result["resolved_bytes"] == 60
    assert result["layer_views"]["all_storage_aliases"] is True
    assert result["layer_views"]["views"] == 10
