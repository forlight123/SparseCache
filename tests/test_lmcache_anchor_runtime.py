import pytest

from dataclasses import dataclass

from experiments.lossless_pd.lmcache_pd.anchor_runtime import (
    _claim_seed,
    _phase_spec,
    anchor_indices,
    partition_indices,
    publish_seed_batch,
)


def test_uniform_anchor_indices_cover_requested_fraction_without_duplicates():
    selected = anchor_indices(31, 0.1, "uniform")
    assert selected == [3, 11, 19, 27]
    assert len(selected) == len(set(selected))


def test_prefix_anchor_indices_are_contiguous():
    assert anchor_indices(10, 0.2, "prefix") == [0, 1]


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_anchor_fraction_is_bounded(fraction):
    with pytest.raises(ValueError):
        anchor_indices(10, fraction, "uniform")


def test_anchor_mode_is_declared():
    with pytest.raises(ValueError):
        anchor_indices(10, 0.1, "unknown")


def test_partition_is_stable_exact_and_byte_conserving():
    anchors, residual = partition_indices(31, 0.1, "uniform")
    assert anchors == [3, 11, 19, 27]
    assert sorted(anchors + residual) == list(range(31))
    assert set(anchors).isdisjoint(residual)


@dataclass
class FakeDisaggSpec:
    req_id: str
    is_last_prefill: bool = True
    total_chunks: int = 0


def test_phase_specs_are_cloned_and_preserve_total_request_chunks():
    original = FakeDisaggSpec("request-1")
    anchor = _phase_spec(
        original,
        phase="anchor",
        is_last=False,
        total=31,
        request_id="external-1",
        seed_record={"seed_token_id": 42},
        indices=[3, 11, 19, 27],
    )
    residual = _phase_spec(
        original,
        phase="residual",
        is_last=True,
        total=31,
        request_id="external-1",
    )
    assert original.is_last_prefill is True
    assert original.total_chunks == 0
    assert (anchor.is_last_prefill, anchor.total_chunks) == (False, 31)
    assert (residual.is_last_prefill, residual.total_chunks) == (True, 31)
    assert anchor._sparsecache_phase == "anchor"
    assert residual._sparsecache_phase == "residual"
    assert anchor._sparsecache_request_id == "external-1"
    assert anchor._sparsecache_seed_record == {"seed_token_id": 42}
    assert anchor._sparsecache_indices == (3, 11, 19, 27)


def test_seed_queue_preserves_batch_order():
    while _claim_seed() is not None:
        pass
    publish_seed_batch([[7], [11]])
    assert _claim_seed()["seed_token_id"] == 7
    assert _claim_seed()["seed_token_id"] == 11
    assert _claim_seed() is None
