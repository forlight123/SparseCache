from types import SimpleNamespace

import torch

from experiments.blockdraft.model import BlockKVConfig, BlockKVDraft
from experiments.lossless_pd.causal_correction_train import (
    convert_checkpoint,
    prefix_supervision_mask,
    select_trainable_parameters,
)
from experiments.lossless_pd.core import ProgressiveBlock


def test_convert_checkpoint_preserves_reranker_when_only_topk_changes(tmp_path):
    source = BlockKVDraft(
        BlockKVConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_draft_layers=2,
            num_target_kv_layers=2,
            block_size=4,
            vocab_size=64,
            correction_hidden_size=16,
            correction_bottleneck_size=8,
            correction_mode="rerank",
            correction_topk=4,
        )
    )
    with torch.no_grad():
        source.correction_head[-1].weight.fill_(0.25)
    source.save_checkpoint(tmp_path)
    args = SimpleNamespace(
        base_checkpoint=str(tmp_path),
        correction_hidden=16,
        correction_bottleneck=8,
        correction_mode="rerank",
        correction_topk=16,
    )

    converted = convert_checkpoint(args)

    assert converted.base.config.correction_topk == 16
    torch.testing.assert_close(
        converted.base.correction_head[-1].weight,
        source.correction_head[-1].weight,
    )


def test_adapter_scope_trains_projector_stage_and_correction_only():
    model = ProgressiveBlock(
        BlockKVDraft(
            BlockKVConfig(
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                num_draft_layers=2,
                num_target_kv_layers=2,
                block_size=4,
                vocab_size=64,
                target_kv_fusion="dflash_hidden",
                correction_hidden_size=16,
                correction_bottleneck_size=8,
                correction_mode="rerank",
                correction_topk=4,
            )
        )
    ).to(dtype=torch.bfloat16)

    selected = select_trainable_parameters(model, "adapter")

    assert selected
    assert all(parameter.dtype == torch.float32 for parameter in selected)
    assert all(parameter.requires_grad for parameter in model.stage.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.base.target_hidden_projector.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in model.base.layers.parameters()
    )


def test_prefix_supervision_stops_at_first_error_and_excludes_uncovered_target():
    correct = torch.tensor(
        [
            [True, True, False, True, True],
            [True, True, True, True, True],
            [True, False, True, True, True],
        ]
    )
    covered = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, False, True, True, True],
        ]
    )

    mask = prefix_supervision_mask(correct, covered, loss_start_position=1)

    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [False, True, True, False, False],
                [False, True, True, True, True],
                [False, False, False, False, False],
            ]
        ),
    )
