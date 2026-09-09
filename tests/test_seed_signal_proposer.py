from types import SimpleNamespace

from experiments.lossless_pd.lmcache_pd.anchor_runtime import _claim_seed
from experiments.lossless_pd.lmcache_pd.seed_signal_proposer import SeedSignalProposer


def test_seed_signal_proposer_records_target_seed_and_returns_no_draft():
    while _claim_seed() is not None:
        pass
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=1)
    )
    proposer = SeedSignalProposer(config)
    output = proposer.propose([[13]], [100], [[1, 2]], slot_mappings=None)
    assert output == [[]]
    assert _claim_seed()["seed_token_id"] == 13
