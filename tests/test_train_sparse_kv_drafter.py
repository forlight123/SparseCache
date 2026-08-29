import json

import pytest
from safetensors.torch import save_file
import torch

from experiments.train_sparse_kv_drafter import (
    TeacherSample,
    feature_distillation_loss,
    load_teacher_dataset,
    parse_fraction_list,
    sha256_file,
    target_sequence_score,
    training_window_inputs,
    weighted_distillation_loss,
)


def test_parse_fraction_list():
    assert parse_fraction_list("0.05,0.1,1") == (0.05, 0.1, 1.0)
    with pytest.raises(ValueError):
        parse_fraction_list("0")
    with pytest.raises(ValueError):
        parse_fraction_list("1.1")


def test_weighted_distillation_loss_prefers_teacher_logits():
    labels = torch.tensor([2, 3])
    teacher_ids = torch.tensor([[2, 1], [3, 0]])
    teacher_logprobs = torch.tensor([[-0.1, -2.0], [-0.2, -1.5]])
    poor = torch.zeros(1, 2, 5)
    good = poor.clone()
    good[0, 0, 2] = 5.0
    good[0, 1, 3] = 5.0
    poor_loss, _ = weighted_distillation_loss(
        poor,
        labels,
        teacher_ids,
        teacher_logprobs,
        prefix_decay=0.9,
        hard_weight=1.0,
        soft_weight=0.5,
    )
    good_loss, components = weighted_distillation_loss(
        good,
        labels,
        teacher_ids,
        teacher_logprobs,
        prefix_decay=0.9,
        hard_weight=1.0,
        soft_weight=0.5,
    )
    assert good_loss < poor_loss
    assert components["hard_loss"] > 0.0


def test_feature_loss_and_sequence_score_prefer_matching_targets():
    teacher = torch.randn(2, 8)
    matching = feature_distillation_loss(
        teacher.unsqueeze(0),
        teacher,
        prefix_decay=0.9,
    )
    opposite = feature_distillation_loss(
        -teacher.unsqueeze(0),
        teacher,
        prefix_decay=0.9,
    )
    assert matching < opposite

    labels = torch.tensor([1, 2])
    good = torch.zeros(1, 2, 4)
    bad = torch.zeros(1, 2, 4)
    good[0, 0, 1] = good[0, 1, 2] = 3.0
    bad[0, 0, 0] = bad[0, 1, 0] = 3.0
    assert target_sequence_score(
        good, labels, prefix_decay=0.9
    ) > target_sequence_score(bad, labels, prefix_decay=0.9)


def test_v4_teacher_loader_merges_trace_with_v3_static_kv(tmp_path):
    base_dir = tmp_path / "base"
    trace_dir = tmp_path / "trace"
    base_dir.mkdir()
    trace_dir.mkdir()
    base_file = base_dir / "sample.safetensors"
    trace_file = trace_dir / "trace.safetensors"
    static = {
        "memory_keys": torch.randn(1, 2, 4, 3),
        "memory_values": torch.randn(1, 2, 4, 3),
        "seed_hidden": torch.randn(1, 6),
        "priority_page_scores": torch.ones(1),
        "prompt_ids": torch.arange(4, dtype=torch.int32),
    }
    original_trace = {
        "input_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([2, 3]),
        "teacher_topk_ids": torch.tensor([[2, 1], [3, 1]]),
        "teacher_topk_logprobs": torch.randn(2, 2),
        "teacher_hidden": torch.randn(2, 6),
        "query_cos": torch.randn(2, 3),
        "query_sin": torch.randn(2, 3),
    }
    save_file(static | original_trace, base_file)
    save_file(original_trace, trace_file)
    base_manifest = {
        "format": "sparsecache.sparse-kv-draft-teacher.v3",
        "target_fingerprint": "target",
        "samples": [
            {
                "request_index": 0,
                "file": base_file.name,
                "file_sha256": sha256_file(base_file),
                "prompt_tokens": 4,
            }
        ],
    }
    base_manifest_path = base_dir / "manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")
    trace_manifest = {
        "format": "sparsecache.sparse-kv-draft-teacher.v4",
        "base_manifest": str(base_manifest_path),
        "target_fingerprint": "target",
        "samples": [
            {
                "request_index": 0,
                "file": trace_file.name,
                "file_sha256": sha256_file(trace_file),
                "prompt_tokens": 4,
                "source_dataset": "test",
            }
        ],
    }
    trace_manifest_path = trace_dir / "manifest.json"
    trace_manifest_path.write_text(json.dumps(trace_manifest), encoding="utf-8")
    _, samples = load_teacher_dataset(trace_manifest_path)
    sample = samples[0]
    assert sample.horizon == 2
    assert sample.source_group == "test"
    assert torch.equal(sample.tensors["memory_keys"], static["memory_keys"])


def test_training_window_uses_previous_verifier_boundary() -> None:
    tensors = {
        "input_ids": torch.tensor([10, 11, 12, 13]),
        "labels": torch.tensor([11, 12, 13, 14]),
        "seed_hidden": torch.full((2, 3), -1.0),
        "continuation_seed_hidden": torch.arange(24).view(4, 2, 3),
        "teacher_topk_ids": torch.arange(8).view(4, 2),
        "teacher_topk_logprobs": torch.randn(4, 2),
        "query_cos": torch.randn(4, 2),
        "query_sin": torch.randn(4, 2),
    }
    sample = TeacherSample(0, 8, tensors)
    inputs = training_window_inputs(
        sample,
        torch.device("cpu"),
        offset=2,
        window_tokens=2,
    )
    assert inputs[0].tolist() == [[12, 13]]
    assert torch.equal(inputs[1][0], tensors["continuation_seed_hidden"][1])
    assert inputs[4].tolist() == [13, 14]
