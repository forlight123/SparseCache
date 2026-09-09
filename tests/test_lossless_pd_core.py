import pytest
import torch
from torch import nn

from experiments.blockdraft.model import BlockKVConfig, BlockKVDraft
from experiments.lossless_pd.core import (
    ProgressiveBlock, accepted_prefix, attention_state, attention_value,
    expand_gqa, fixed_query_bound, merge_attention, page_order, visible_positions,
)
from experiments.lossless_pd.schedule_analysis import arrivals, finish, switch_arrivals
from experiments.lossless_pd.integrated_probe import contiguous_ranges
from experiments.lossless_pd.sequence_equivalence import greedy_commit
from experiments.lossless_pd.shape_invariant_attention import (
    shape_invariant_attention_forward,
)


def fixture_model(fusion="per_layer"):
    torch.manual_seed(55)
    base = BlockKVDraft(BlockKVConfig(
        hidden_size=32, intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, num_draft_layers=2,
        num_target_kv_layers=2, block_size=4, vocab_size=64,
        target_kv_fusion=fusion,
    )).eval()
    return ProgressiveBlock(base).eval(), nn.Embedding(64,32), nn.Linear(32,64,bias=False)


@pytest.mark.parametrize("fusion",["per_layer","projected","dflash_hidden"])
def test_full_view_matches_imported_block_reference(fusion):
    model, embedding, head = fixture_model(fusion)
    k = torch.randn(1,2,2,9,8)
    v = torch.randn_like(k)
    seed = torch.tensor([[7]])
    actual = model(seed,embedding,head,k,v,torch.arange(9),9)
    expected = model.base.forward_logits(seed,embedding,head,k,v,9)
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize("fusion",["per_layer","projected","dflash_hidden"])
def test_sparse_position_permutation_does_not_renumber_rope(fusion):
    model,embedding,head = fixture_model(fusion)
    k = torch.randn(1,2,2,4,8)
    v = torch.randn_like(k)
    pos = torch.tensor([0,3,8,11])
    seed = torch.tensor([[7]])
    perm = torch.tensor([2,0,3,1])
    original = model(seed,embedding,head,k,v,pos,12)
    reordered = model(seed,embedding,head,k[...,perm,:],v[...,perm,:],pos[perm],12)
    torch.testing.assert_close(original,reordered,atol=1e-6,rtol=1e-5)


def test_only_arrived_kv_receives_gradient_and_no_future_labels_are_inputs():
    model,embedding,head = fixture_model("projected")
    k = torch.randn(1,2,2,12,8,requires_grad=True)
    v = torch.randn_like(k,requires_grad=True)
    pos = torch.tensor([0,3,8,11])
    logits = model(torch.tensor([[7]]),embedding,head,k[...,pos,:],v[...,pos,:],pos,12)
    logits.square().mean().backward()
    missing = torch.tensor([1,2,4,5,6,7,9,10])
    assert k.grad[...,missing,:].abs().sum() == 0
    assert v.grad[...,missing,:].abs().sum() == 0
    assert v.grad[...,pos,:].abs().sum() > 0
    assert model.stage.weight.grad.abs().sum() > 0


def test_future_seed_kv_is_rejected():
    model,embedding,head = fixture_model()
    k = torch.randn(1,2,2,2,8)
    with pytest.raises(ValueError,match="future"):
        model(torch.tensor([[7]]),embedding,head,k,k,torch.tensor([0,12]),12)


def test_progressive_correction_is_causal_and_needs_no_target_hidden_state():
    torch.manual_seed(91)
    base = BlockKVDraft(BlockKVConfig(
        hidden_size=32, intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, num_draft_layers=2,
        num_target_kv_layers=2, block_size=4, vocab_size=64,
        correction_hidden_size=8, correction_bottleneck_size=8,
    )).eval()
    nn.init.normal_(base.correction_head[-1].weight)
    model = ProgressiveBlock(base).eval()
    embedding, head = nn.Embedding(64, 32), nn.Linear(32, 64, bias=False)
    key = torch.randn(1, 2, 2, 9, 8)
    value = torch.randn_like(key)
    previous = torch.tensor([[7, 11, 12]])
    changed = previous.clone()
    changed[:, 2] = 13
    first = model(
        previous[:, :1], embedding, head, key, value, torch.arange(9), 9,
        previous_tokens=previous,
    )
    base_logits, correction = model.forward_components(
        previous[:, :1], embedding, head, key, value, torch.arange(9), 9,
        previous_tokens=previous,
    )
    torch.testing.assert_close(first, base_logits + correction, atol=0, rtol=0)
    second = model(
        previous[:, :1], embedding, head, key, value, torch.arange(9), 9,
        previous_tokens=changed,
    )
    torch.testing.assert_close(first[:, :2], second[:, :2], atol=0, rtol=0)
    assert not torch.equal(first[:, 2], second[:, 2])
    assert model.propose(
        previous[:, :1], embedding, head, key, value, torch.arange(9), 9
    ).shape == (1, 3)


def test_progressive_reranker_stays_inside_base_topk():
    torch.manual_seed(92)
    base = BlockKVDraft(BlockKVConfig(
        hidden_size=32, intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, num_draft_layers=2,
        num_target_kv_layers=2, block_size=4, vocab_size=64,
        correction_hidden_size=8, correction_bottleneck_size=8,
        correction_mode="rerank", correction_topk=5,
    )).eval()
    model = ProgressiveBlock(base).eval()
    embedding, head = nn.Embedding(64, 32), nn.Linear(32, 64, bias=False)
    key = torch.randn(1, 2, 2, 9, 8)
    value = torch.randn_like(key)
    seed = torch.tensor([[7]])
    previous = torch.tensor([[7, 11, 12]])
    candidates, base_scores, corrections = model.teacher_forced_rerank(
        seed, embedding, head, key, value, torch.arange(9), 9,
        previous_tokens=previous,
    )
    assert candidates.shape == base_scores.shape == corrections.shape == (1, 3, 5)
    proposal = model.propose(
        seed, embedding, head, key, value, torch.arange(9), 9
    )
    inherited = head(model.sparse_hidden(
        seed, embedding, key, value, torch.arange(9), 9
    )).argmax(-1)
    torch.testing.assert_close(proposal, inherited, atol=0, rtol=0)
    assert all(
        int(proposal[0, position]) in candidates[0, position].tolist()
        for position in range(3)
    )


def test_views_are_nested_and_account_for_protected_pages():
    order = page_order(19,4,"random",22)
    previous = set()
    for fraction in [.05,.1,.2,.5,1]:
        pos = visible_positions(19,4,fraction,order,torch.device("cpu"))
        current = set(pos.tolist())
        assert previous <= current
        assert {0,1,2,3,16,17,18} <= current
        previous = current
    assert previous == set(range(19))


def test_exact_streaming_attention_with_gqa_and_out_of_order_pages():
    torch.manual_seed(97)
    q = torch.randn(1,4,3,8)
    k = torch.randn(1,2,19,8)
    v = torch.randn_like(k)
    dense = attention_value(attention_state(q,k,v))
    state = None
    for start,end in [(7,13),(0,7),(13,19)]:
        state = merge_attention(state,attention_state(q,k[...,start:end,:],v[...,start:end,:]))
    torch.testing.assert_close(attention_value(state),dense,atol=1e-6,rtol=1e-5)


def test_fixed_exact_query_bounds_cover_missing_attention_mass():
    torch.manual_seed(44)
    q = torch.randn(1,4,1,8)
    k = torch.randn(1,2,19,8)
    v = torch.randn_like(k)
    for pos in [torch.tensor([0,1,16,17,18]),torch.arange(19)]:
        result = fixed_query_bound(q,k,v,pos,4)
        assert result["bound_violations_fp64_tolerance_1e_9"] == 0
        assert result["mass_violations_fp64_tolerance_1e_9"] == 0
    assert result["mean_bound"] == 0


def test_acceptance_excludes_tokens_after_eos():
    assert accepted_prefix([1,2,3,4],[1,2,3,4],{2}) == 2
    assert accepted_prefix([1,4,3],[1,2,3]) == 1
    assert accepted_prefix([1],[]) == 0


def test_anchor_schedule_does_not_duplicate_payload_bytes():
    payload = 36_000_000
    bandwidth = 100
    ordinary, _ = arrivals(payload, 0, bandwidth, False)
    anchored, anchor_ready = arrivals(payload, .1, bandwidth, True)
    expected_total_ms = payload * 8 / (bandwidth * 1e6)
    assert ordinary[-1] == pytest.approx(expected_total_ms)
    assert anchored[-1] == pytest.approx(expected_total_ms)
    assert anchor_ready == pytest.approx(5 * .1 * (payload / 36) * 8 /
                                         (bandwidth * 1e6))
    for prefix_layers in range(37):
        switched, _ = switch_arrivals(payload, .1, bandwidth, prefix_layers)
        assert switched[-1] == pytest.approx(expected_total_ms)


def test_layer_dependency_recurrence_waits_for_release_and_predecessor():
    assert finish([2, 10, 11], [3, 4, 5], initial=7, terminal=1) == 20


def test_anchor_and_residual_ranges_partition_prompt():
    positions = torch.tensor([0, 1, 4, 7, 8])
    assert contiguous_ranges(positions, 10, True) == [(0, 2), (4, 5), (7, 9)]
    assert contiguous_ranges(positions, 10, False) == [(2, 4), (5, 7), (9, 10)]


def test_greedy_speculation_commits_target_correction():
    assert greedy_commit([4, 5, 6], [4, 9, 3, 8]) == ([4, 9], 1)
    assert greedy_commit([4, 5], [4, 5, 8]) == ([4, 5, 8], 2)


def test_shape_invariant_attention_matches_explicit_causal_rows():
    class Attention:
        num_key_value_groups = 2
        training = False

    torch.manual_seed(88)
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 2, 10, 8)
    value = torch.randn_like(key)
    actual, weights = shape_invariant_attention_forward(
        Attention(), query, key, value, None, 8 ** -0.5
    )
    expected = []
    expanded_key, expanded_value = expand_gqa(key, 4), expand_gqa(value, 4)
    for row in range(3):
        visible = 8 + row
        probability = torch.softmax(
            query[:, :, row:row + 1] @ expanded_key[:, :, :visible].transpose(-1, -2)
            * (8 ** -0.5), dim=-1, dtype=torch.float32
        ).to(query.dtype)
        expected.append(probability @ expanded_value[:, :, :visible])
    expected = torch.cat(expected, dim=2).transpose(1, 2).contiguous()
    assert weights is None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
