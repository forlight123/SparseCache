import torch

from sparsecache.sparse_kv_draft import (
    SparseKVDraftConfig,
    SparseKVDrafter,
    accepted_prefix_length,
    select_visible_token_indices,
)


def test_sparse_kv_drafter_forward_and_checkpoint_exclude_target_weights():
    config = SparseKVDraftConfig(
        target_hidden_size=32,
        draft_hidden_size=16,
        head_dim=4,
        num_draft_heads=4,
        num_memory_heads=4,
        num_memory_layers=2,
        num_seed_layers=2,
        mlp_ratio=2,
    )
    model = SparseKVDrafter(config)
    embedding = torch.randn(64, 32)
    lm_head = torch.randn(64, 32)
    model.bind_target_weights(embedding, lm_head)
    logits = model(
        torch.tensor([[1, 2, 3]]),
        torch.randn(1, 2, 32),
        torch.randn(2, 4, 7, 4),
        torch.randn(2, 4, 7, 4),
        torch.randn(3, 4),
        torch.randn(3, 4),
        visible_fraction=0.25,
        prompt_tokens=32,
    )
    assert logits.shape == (1, 3, 64)
    assert all("target_" not in name for name in model.state_dict())
    logits.sum().backward()
    assert model.blocks[0].memory_attention.query[0].weight.grad is not None


def test_sparse_kv_drafter_can_return_target_space_features():
    config = SparseKVDraftConfig(
        target_hidden_size=16,
        draft_hidden_size=8,
        head_dim=4,
        num_draft_heads=2,
        num_memory_heads=2,
        num_memory_layers=1,
        num_seed_layers=1,
        mlp_ratio=2,
    )
    model = SparseKVDrafter(config)
    model.bind_target_weights(torch.randn(20, 16), torch.randn(20, 16))
    logits, features = model(
        torch.tensor([[1, 2]]),
        torch.randn(1, 1, 16),
        torch.randn(1, 2, 5, 4),
        torch.randn(1, 2, 5, 4),
        torch.randn(2, 4),
        torch.randn(2, 4),
        visible_fraction=0.2,
        prompt_tokens=10,
        return_features=True,
    )
    assert logits.shape == (1, 2, 20)
    assert features.shape == (1, 2, 16)


def test_page_selection_protects_boundaries_and_is_deterministic():
    first = select_visible_token_indices(
        1000,
        page_size=64,
        visible_fraction=0.25,
        seed=7,
    )
    second = select_visible_token_indices(
        1000,
        page_size=64,
        visible_fraction=0.25,
        seed=7,
    )
    assert torch.equal(first, second)
    assert set(range(64)).issubset(set(first.tolist()))
    assert set(range(960, 1000)).issubset(set(first.tolist()))
    assert torch.all(first[1:] > first[:-1])


def test_full_visibility_and_accepted_prefix():
    indices = select_visible_token_indices(
        130,
        page_size=64,
        visible_fraction=1.0,
        seed=0,
        mode="strided",
    )
    assert indices.tolist() == list(range(130))
    assert accepted_prefix_length([1, 2, 9], [1, 2, 3]) == 2
    assert accepted_prefix_length([1], [1, 2]) == 1


def test_random_page_views_are_nested_for_one_arrival_order():
    small = select_visible_token_indices(
        4096,
        page_size=64,
        visible_fraction=0.05,
        seed=17,
    )
    medium = select_visible_token_indices(
        4096,
        page_size=64,
        visible_fraction=0.10,
        seed=17,
    )
    large = select_visible_token_indices(
        4096,
        page_size=64,
        visible_fraction=0.20,
        seed=17,
    )
    assert set(small.tolist()) < set(medium.tolist()) < set(large.tolist())


def test_priority_page_selection_uses_highest_unprotected_scores():
    scores = torch.arange(10, dtype=torch.float32)
    selected = select_visible_token_indices(
        640,
        page_size=64,
        visible_fraction=0.3,
        seed=0,
        mode="priority",
        page_scores=scores,
    )
    pages = {token // 64 for token in selected.tolist()}
    assert pages == {0, 8, 9}
