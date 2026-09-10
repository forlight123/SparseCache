from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.small_kv_adapter.cross_attention import (
    CrossAttentionConfig,
    SparseTargetKVCrossAttention,
)


class IdentityLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        del kwargs
        return hidden_states


class FakeModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList(layers))


def tiny_adapter(initial_gate=0.05):
    return SparseTargetKVCrossAttention(
        CrossAttentionConfig(
            target_layer_ids=(1,),
            draft_layer_ids=(0,),
            hidden_size=8,
            num_key_value_heads=2,
            head_dim=4,
            rope_theta=10_000.0,
            initial_gate=initial_gate,
        )
    )


def test_cross_attention_hook_changes_only_active_execution():
    torch.manual_seed(7)
    adapter = tiny_adapter()
    model = FakeModel([IdentityLayer()])
    keys = torch.randn(1, 1, 2, 3, 4)
    values = torch.randn_like(keys)
    positions = torch.tensor([0, 1, 2])
    memory = adapter.prepare_memory(keys, values, positions)
    hidden = torch.randn(1, 2, 8)
    baseline = model.model.layers[0](hidden)

    with adapter.activate(model, memory, torch.tensor([3, 4])):
        conditioned = model.model.layers[0](hidden)

    assert not torch.equal(conditioned, baseline)
    torch.testing.assert_close(model.model.layers[0](hidden), baseline)


def test_zero_gate_is_exact_identity_and_hooks_are_removed_on_error():
    adapter = tiny_adapter(initial_gate=0.0)
    model = FakeModel([IdentityLayer()])
    target = torch.randn(1, 1, 2, 2, 4)
    positions = torch.tensor([0, 1])
    memory = adapter.prepare_memory(target, target, positions)
    hidden = torch.randn(1, 1, 8)
    with (
        pytest.raises(RuntimeError),
        adapter.activate(model, memory, torch.tensor([2])),
    ):
        torch.testing.assert_close(model.model.layers[0](hidden), hidden)
        raise RuntimeError("sentinel")
    torch.testing.assert_close(model.model.layers[0](hidden), hidden)


def test_cross_attention_rejects_wrong_memory_shape():
    adapter = tiny_adapter()
    target = torch.randn(1, 1, 2, 2, 4)
    with pytest.raises(ValueError):
        adapter.prepare_memory(target[..., :1, :], target, torch.tensor([0, 1]))
