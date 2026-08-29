from pathlib import Path

import torch

from experiments.train_sparse_kv_eagle import (
    configure_trainable_modules,
    load_adapter_checkpoint,
)
from sparsecache.sparse_kv_eagle import (
    SparseKVEagleConfig,
    SparseKVEagleDrafter,
)


def tiny_config() -> SparseKVEagleConfig:
    return SparseKVEagleConfig(
        target_hidden_size=16,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_memory_heads=2,
        num_memory_layers=2,
        num_seed_layers=2,
        target_vocab_size=32,
        draft_vocab_size=16,
        rms_norm_eps=1e-5,
    )


def test_sparse_kv_eagle_forward_and_adapter_gradients() -> None:
    model = SparseKVEagleDrafter(tiny_config())
    model.bind_target_embedding(torch.randn(32, 16))
    model.freeze_eagle()
    logits = model(
        torch.tensor([[1, 2, 3]]),
        torch.randn(1, 2, 16),
        torch.randn(2, 2, 7, 4),
        torch.randn(2, 2, 7, 4),
        torch.randn(3, 4),
        torch.randn(3, 4),
        visible_fraction=0.25,
        prompt_tokens=64,
    )
    assert logits.shape == (1, 3, 16)
    logits.sum().backward()
    assert model.memory_attention.query[0].weight.grad is not None
    assert model.midlayer.self_attn.q_proj.weight.grad is None


def test_sparse_kv_eagle_can_disable_memory_exactly() -> None:
    model = SparseKVEagleDrafter(tiny_config())
    model.bind_target_embedding(torch.randn(32, 16))
    arguments = (
        torch.tensor([[1, 2]]),
        torch.randn(1, 2, 16),
        torch.randn(2, 2, 5, 4),
        torch.randn(2, 2, 5, 4),
        torch.randn(2, 4),
        torch.randn(2, 4),
    )
    first = model(
        *arguments,
        visible_fraction=0.1,
        prompt_tokens=32,
        use_memory=False,
    )
    arguments = (*arguments[:2], torch.randn_like(arguments[2]), *arguments[3:])
    second = model(
        *arguments,
        visible_fraction=0.1,
        prompt_tokens=32,
        use_memory=False,
    )
    assert torch.equal(first, second)


def test_eagle_checkpoint_mapping_and_adapter_only_state(tmp_path: Path) -> None:
    source = SparseKVEagleDrafter(tiny_config())
    checkpoint_state = {
        name: tensor
        for name, tensor in source.state_dict().items()
        if not name.startswith(("memory_", "stage_"))
        and name not in {"adapter_scale", "draft_to_target", "target_to_draft"}
    }
    checkpoint_state["d2t"] = torch.arange(16)
    checkpoint_state["t2d"] = torch.zeros(32, dtype=torch.bool)
    checkpoint = tmp_path / "eagle.pt"
    torch.save(checkpoint_state, checkpoint)

    model = SparseKVEagleDrafter(tiny_config())
    model.load_eagle_checkpoint(checkpoint)
    model.freeze_eagle()
    assert model.draft_to_target.tolist() == list(range(0, 32, 2))
    assert model.draft_ids(torch.tensor([0, 2, 31])).tolist() == [0, 1, -1]
    adapter_state = model.adapter_state_dict()
    assert adapter_state
    assert all(
        name.startswith(("memory_", "stage_")) or name == "adapter_scale"
        for name in adapter_state
    )


def test_training_scope_and_resume_adapter(tmp_path: Path) -> None:
    model = SparseKVEagleDrafter(tiny_config())
    model.freeze_eagle()
    configure_trainable_modules(
        model,
        train_fc=True,
        train_midlayer=True,
        adapter_scale_init=0.0,
    )
    assert model.fc.weight.requires_grad
    assert model.midlayer.self_attn.q_proj.weight.requires_grad
    assert model.adapter_scale.item() == 0.0

    state = model.adapter_state_dict()
    state["adapter_scale"] = torch.tensor(1.25)
    checkpoint = tmp_path / "adapter.pt"
    torch.save(
        {
            "adapter_state_dict": state,
            "target_fingerprint": "tiny",
            "completed_steps": 17,
        },
        checkpoint,
    )
    loaded = load_adapter_checkpoint(model, checkpoint)
    assert loaded["completed_steps"] == 17
    assert model.adapter_scale.item() == 1.25
