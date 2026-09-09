from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.lossless_pd.lmcache_pd.online_drafter import (
    _REGISTRY,
    OnlineSparseKVProposer,
    ReadyDraft,
    pack_claimed_target_kv,
)
from experiments.lossless_pd.lmcache_pd.receiver_runtime import ClaimedLayerViews


class Owner:
    def __init__(self, tensor):
        self.tensor = tensor
        self.released = 0

    def ref_count_down(self):
        self.released += 1


def test_pack_claimed_target_kv_restores_position_and_head_layout():
    late = torch.arange(2 * 4 * 2 * 4, dtype=torch.float32).reshape(2, 4, 2, 4)
    early = 1000 + torch.arange(2 * 4 * 2 * 4, dtype=torch.float32).reshape(2, 4, 2, 4)
    owners = [Owner(late), Owner(early)]
    layers = (1, 3)
    claim = ClaimedLayerViews(
        objects=owners,
        chunk_indices=(2, 0),
        token_ranges=((4, 6), (0, 2)),
        prompt_tokens=8,
        layers=layers,
        views={
            layer: [owner.tensor[:, layer : layer + 1] for owner in owners]
            for layer in layers
        },
    )

    packed = pack_claimed_target_kv(claim, num_key_value_heads=2, head_dim=2)

    assert packed.keys.shape == (1, 2, 2, 4, 2)
    assert packed.values.shape == packed.keys.shape
    assert packed.positions.tolist() == [0, 1, 4, 5]
    expected_early_key = early[0, 1].reshape(2, 2, 2).permute(1, 0, 2)
    expected_late_key = late[0, 1].reshape(2, 2, 2).permute(1, 0, 2)
    assert torch.equal(packed.keys[0, 0, :, :2], expected_early_key)
    assert torch.equal(packed.keys[0, 0, :, 2:], expected_late_key)


def test_pack_claimed_target_kv_requires_exact_ranges():
    tensor = torch.empty(2, 2, 2, 4)
    owner = Owner(tensor)
    claim = ClaimedLayerViews(
        objects=[owner],
        chunk_indices=(0,),
        token_ranges=(),
        prompt_tokens=2,
        layers=(1,),
        views={1: [tensor[:, 1:2]]},
    )
    with pytest.raises(ValueError, match="exact token range"):
        pack_claimed_target_kv(claim, num_key_value_heads=2, head_dim=2)


def ready_draft(*, first=17):
    return ReadyDraft(
        request_id="external-1",
        pd_request_id="pd-1",
        prompt_tokens=3,
        seed_token_id=11,
        proposals=(first, 19, 23, 29),
        anchor_received_ns=1,
        draft_started_ns=2,
        draft_finished_ns=3,
        pack_gpu_ms=0.1,
        model_gpu_ms=1.0,
        total_gpu_ms=1.1,
        wall_ms=1.2,
        visible_tokens=2,
        actual_fraction=2 / 3,
    )


class RepairService:
    def repair_suffix(self, draft, first_target_token):
        assert draft.continuation == "prepared"
        return [first_target_token + 1, first_target_token + 2], 0.2, 0.3


def proposer(mode="inject", speculative_tokens=2):
    value = OnlineSparseKVProposer.__new__(OnlineSparseKVProposer)
    value.num_speculative_tokens = speculative_tokens
    value.mode = mode
    value.trace_path = ""
    value.service = SimpleNamespace()
    value._pending_feedback = []
    return value


def test_inject_reconciles_first_target_token_then_returns_only_suffix():
    _REGISTRY.clear()
    _REGISTRY.publish(ready_draft())
    output = proposer().propose(
        [[17]],
        np.array([4]),
        np.array([[2, 3, 5, 11]]),
    )
    assert output == [[19, 23]]
    assert len(_REGISTRY) == 0


def test_inject_drops_divergent_or_unmatched_blocks():
    _REGISTRY.clear()
    _REGISTRY.publish(ready_draft())
    assert proposer().propose([[31]], np.array([4]), np.array([[2, 3, 5, 11]])) == [[]]
    _REGISTRY.publish(ready_draft())
    assert proposer().propose([[17]], np.array([5]), np.array([[2, 3, 5, 7, 11]])) == [
        []
    ]


def test_observe_mode_never_injects_even_when_first_token_matches():
    _REGISTRY.clear()
    _REGISTRY.publish(ready_draft())
    assert proposer("observe").propose(
        [[17]], np.array([4]), np.array([[2, 3, 5, 11]])
    ) == [[]]


def test_target_conditioned_repair_salvages_a_first_token_mismatch():
    _REGISTRY.clear()
    draft = ready_draft(first=17)
    object.__setattr__(draft, "continuation", "prepared")
    _REGISTRY.publish(draft)
    value = proposer()
    value.service = RepairService()
    assert value.propose([[31]], np.array([4]), np.array([[2, 3, 5, 11]])) == [[32, 33]]


def test_next_step_feedback_recovers_verified_prefix_length():
    _REGISTRY.clear()
    value = proposer(speculative_tokens=3)
    _REGISTRY.publish(ready_draft())
    assert value.propose(
        [[17]],
        np.array([5]),
        np.array([[2, 3, 5, 11, 17, 0, 0]]),
    ) == [[19, 23, 29]]
    assert len(value._pending_feedback) == 1

    # The next proposer call contains prompt + seed + two accepted drafter
    # tokens + the target correction token.  The original first proposal was
    # reconciled before injection, so the inferred accepted prefix is two.
    assert value.propose(
        [[31]],
        np.array([7]),
        np.array([[2, 3, 5, 11, 17, 19, 31]]),
    ) == [[]]
    assert value._pending_feedback == []
