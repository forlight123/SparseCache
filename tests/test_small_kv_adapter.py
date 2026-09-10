from types import SimpleNamespace

import pytest
import torch

from experiments.small_kv_adapter.model import (
    SmallKVAdapterConfig,
    SparseTargetKVAdapter,
)
from experiments.small_kv_adapter.train import validate_model_contract


def fake_cache(layers: int, tokens: int):
    return SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.randn(1, 2, tokens, 4),
                values=torch.randn(1, 2, tokens, 4),
            )
            for _ in range(layers)
        ]
    )


def test_zero_initialized_adapter_is_an_exact_cache_identity():
    config = SmallKVAdapterConfig(
        target_layer_ids=(1, 3),
        draft_layer_ids=(0, 2),
        num_key_value_heads=2,
        head_dim=4,
        bottleneck_size=4,
        rope_theta=10_000.0,
    )
    adapter = SparseTargetKVAdapter(config)
    cache = fake_cache(3, 8)
    before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    positions = torch.tensor([1, 6], dtype=torch.long)
    target_keys = torch.randn(1, 2, 2, 2, 4)
    target_values = torch.randn_like(target_keys)

    adapter(cache, target_keys, target_values, positions)

    for layer, (keys, values) in zip(cache.layers, before, strict=True):
        torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, values, rtol=0, atol=0)


def test_adapter_rejects_position_or_layer_mismatch():
    adapter = SparseTargetKVAdapter(
        SmallKVAdapterConfig(
            target_layer_ids=(1,),
            draft_layer_ids=(2,),
            num_key_value_heads=2,
            head_dim=4,
            bottleneck_size=4,
        )
    )
    target = torch.randn(1, 1, 2, 2, 4)
    with pytest.raises(ValueError):
        adapter(fake_cache(2, 8), target, target, torch.tensor([1, 2]))
    with pytest.raises(ValueError):
        adapter(fake_cache(3, 8), target, target, torch.tensor([1]))


@pytest.mark.parametrize(
    "rope_fields",
    [
        {"rope_theta": 1_000_000.0},
        {"rope_parameters": {"rope_theta": 1_000_000.0}},
    ],
)
def test_model_contract_accepts_legacy_and_migrated_rope_config(rope_fields):
    config = SmallKVAdapterConfig(
        target_layer_ids=(1,),
        draft_layer_ids=(2,),
        num_key_value_heads=2,
        head_dim=4,
        rope_theta=1_000_000.0,
    )
    model_config = SimpleNamespace(
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=3,
        **rope_fields,
    )

    validate_model_contract(SimpleNamespace(config=model_config), config)
