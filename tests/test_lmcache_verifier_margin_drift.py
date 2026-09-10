from experiments.lossless_pd.lmcache_pd.analyze_verifier_margin_drift import (
    analyze,
    first_divergence,
)


def _row(record_id, tokens, margins):
    return {
        "ordinal": 0,
        "packet_index": 1,
        "record_id": record_id,
        "pd": {
            "token_ids": tokens,
            "token_scores": [
                {
                    "position": position,
                    "top1_top2_margin": margin,
                    "top_logprobs": {str(token): -0.1, "other": -0.1 - margin},
                }
                for position, (token, margin) in enumerate(zip(tokens, margins))
            ],
        },
    }


def test_first_divergence_includes_length_mismatch():
    assert first_divergence([1, 2], [1, 3]) == 1
    assert first_divergence([1], [1, 2]) == 1
    assert first_divergence([1], [1]) is None


def test_margin_threshold_reports_false_positives_and_recall():
    baseline = {
        "rows": [
            _row("drift", [1, 2, 3], [2.0, 1.0, 0.1]),
            _row("stable", [4, 5, 6], [2.0, 1.0, 0.2]),
        ]
    }
    injected = {
        "rows": [
            _row("drift", [1, 2, 9], [2.0, 1.0, 0.1]),
            _row("stable", [4, 5, 6], [2.0, 1.0, 0.2]),
        ]
    }
    injected["rows"][1]["ordinal"] = 1
    result = analyze(
        baseline,
        injected,
        block_end_position=1,
        thresholds=(0.15, 0.25),
    )
    assert result["drift_requests"] == 1
    assert result["thresholds"]["0.15"]["recall"] == 1.0
    assert result["thresholds"]["0.15"]["false_positive"] == 0
    assert result["thresholds"]["0.25"]["false_positive"] == 1
