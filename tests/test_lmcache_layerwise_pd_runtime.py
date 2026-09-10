from types import SimpleNamespace

import pytest

from experiments.lossless_pd.lmcache_pd.layerwise_pd_runtime import (
    assert_schedule_exact,
    build_transfer_schedule,
    empty_layerwise_retrieve,
    layerwise_retrieve_with_empty_fallback,
    parse_progressive_request,
    schedule_byte_accounting,
)


def test_schedule_reuses_anchor_objects_without_retransmission():
    phases = build_transfer_schedule(4, 4, (1, 3), (0, 3))

    assert [(phase.kind, phase.layer) for phase in phases[:2]] == [
        ("anchor", 1),
        ("anchor", 3),
    ]
    assert phases[-1].is_last is True
    assert all(phase.layer_complete is False for phase in phases[:2])
    assert all(phase.layer_complete is True for phase in phases[2:])
    assert sum(len(phase.chunks) for phase in phases) == 16
    assert_schedule_exact(phases, num_layers=4, num_chunks=4)

    accounting = schedule_byte_accounting(phases, [10, 10, 10, 5])
    assert accounting == {
        "wire_bytes": 140,
        "authoritative_bytes": 140,
        "anchor_bytes": 30,
        "retransmitted_bytes": 0,
        "wire_ratio": 1.0,
    }


def test_full_anchor_marks_layer_complete_without_residual_phase():
    phases = build_transfer_schedule(2, 2, (1,), (0, 1))

    assert [(phase.kind, phase.layer) for phase in phases] == [
        ("anchor", 1),
        ("target", 0),
    ]
    assert all(phase.layer_complete for phase in phases)


def test_schedule_rejects_invalid_or_duplicate_coordinates():
    with pytest.raises(ValueError):
        build_transfer_schedule(4, 4, (1, 1), (0,))
    with pytest.raises(ValueError):
        build_transfer_schedule(4, 4, (4,), (0,))
    with pytest.raises(ValueError):
        build_transfer_schedule(4, 4, (1,), (0, 0))


def test_progressive_request_is_explicit_and_validated():
    request = SimpleNamespace(
        kv_transfer_params={
            "sparsecache_progressive": {
                "request_id": "pd-7",
                "prompt_tokens": 8192,
            }
        }
    )
    assert parse_progressive_request(request) == ("pd-7", 8192)
    assert parse_progressive_request(SimpleNamespace(kv_transfer_params=None)) is None

    request.kv_transfer_params["sparsecache_progressive"]["prompt_tokens"] = True
    with pytest.raises(ValueError):
        parse_progressive_request(request)


def test_empty_layerwise_retrieve_preserves_adapter_protocol():
    import torch

    values = list(empty_layerwise_retrieve(torch.tensor([1, 2, 3]), 3))
    assert len(values) == 5
    assert values[0].item() == 0
    assert values[1:4] == [None, None, None]
    assert values[4].tolist() == [False, False, False]


def test_empty_storage_lookup_supplies_missing_final_mask():
    import torch

    def broken_upstream(engine, tokens, mask=None, **kwargs):
        del engine, tokens, mask, kwargs
        yield None
        raise UnboundLocalError("mem_obj_consumer is unbound")

    values = list(
        layerwise_retrieve_with_empty_fallback(
            broken_upstream,
            SimpleNamespace(num_layers=1),
            torch.tensor([4, 5]),
            mask=torch.ones(2, dtype=torch.bool),
        )
    )
    assert values[0] is None
    assert values[1].tolist() == [False, False]
