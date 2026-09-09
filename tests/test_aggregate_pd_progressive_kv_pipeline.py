from experiments.aggregate_pd_progressive_kv_pipeline import accepted_draft_tokens


def test_accepted_draft_tokens_sums_only_verifier_stages():
    chain = {
        "pipeline": {
            "trace": [
                {"verified": False, "accepted_pending": 9},
                {"verified": True, "accepted_pending": 2},
                {"verified": True, "accepted_pending": 3},
            ]
        }
    }

    assert accepted_draft_tokens(chain) == 5
