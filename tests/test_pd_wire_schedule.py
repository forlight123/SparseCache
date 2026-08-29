from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import pd_progressive_kv_pipeline as pipeline
import progressive_kv_feasibility as feasibility


def test_eager_wire_uses_cumulative_deadlines(monkeypatch):
    stager = pipeline.PinnedKVStager.__new__(pipeline.PinnedKVStager)
    stager.transport_gbps = 8e-9  # one byte per second after multiplying by 8
    stager.eager_serial_wire = True
    stager.stages = (
        pipeline.StageTransfer((), 0, 2),
        pipeline.StageTransfer((), 2, 5),
        pipeline.StageTransfer((), 5, 9),
    )
    stager.layer_elements_per_token = 1
    stager.element_size = 1
    stager.launched = [False, False, False]
    stager.wire_started_at = [None, None, None]
    stager.wire_ready_at = [None, None, None]
    monkeypatch.setattr(pipeline, "perf_counter", lambda: 10.0)

    stager.launch(0)

    assert stager.launched == [True, True, True]
    assert stager.wire_started_at == [10.0, 12.0, 15.0]
    assert stager.wire_ready_at == [12.0, 15.0, 19.0]


def test_continuous_draft_expands_view_between_tokens(monkeypatch):
    class FakeModel:
        def __call__(self, *, past_key_values, **_kwargs):
            past_key_values.update(
                torch.zeros((1, 1, 1, 1)),
                torch.zeros((1, 1, 1, 1)),
                0,
            )
            token = min(7, past_key_values.get_seq_length())
            logits = torch.zeros((1, 1, 8))
            logits[0, 0, token] = 10.0
            logits[0, 0, (token + 1) % 8] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    monkeypatch.setattr(
        feasibility,
        "synchronized_call",
        lambda _device, function, **_kwargs: (function(), 0.5),
    )

    def fake_teacher(_model, _cache, _mask, _suffix, proposals, **_kwargs):
        return {
            "predictions": list(proposals),
            "top1_margins": [9.0] * len(proposals),
            "mean_top1_margin": 9.0,
            "forward_ms": 1.0,
        }

    monkeypatch.setattr(feasibility, "teacher_predictions", fake_teacher)
    caches = tuple(
        ((torch.zeros((1, 1, length, 1)), torch.zeros((1, 1, length, 1))),)
        for length in (2, 4, 6)
    )
    masks = tuple(torch.ones((1, length), dtype=torch.long) for length in (2, 4, 6))
    stage_one_polls = 0
    waited = []

    def ready(stage):
        nonlocal stage_one_polls
        if stage == 2:
            return False
        stage_one_polls += 1
        return stage_one_polls >= 2

    result = feasibility.continuous_graft_chain(
        FakeModel(),
        masks,
        [1],
        draft_tokens=3,
        max_new_tokens=3,
        eos_ids=set(),
        device="cpu",
        stage_caches=caches,
        position_base=6,
        stage_wait=lambda stage: waited.append(stage),
        stage_ready=ready,
        synchronize_device=False,
        reuse_final_verify=False,
    )

    assert result["draft_visibility_stages"] == [0, 1, 1]
    assert result["trace"][-1]["verified"] is True
    assert result["trace"][-1]["accepted_pending"] == 3
    assert waited == [0, 1, 2]


def test_continuous_cli_protocol_is_accepted(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pd_progressive_kv_pipeline.py",
            "--dataset",
            "dataset.jsonl",
            "--model",
            "model",
            "--output",
            "result.jsonl",
            "--sample-count",
            "2",
            "--count",
            "1",
            "--max-new-tokens",
            "8",
            "--draft-tokens",
            "4",
            "--seed-tokens",
            "1",
            "--stage-fractions",
            "0.02,0.5,1.0",
            "--draft-cache-mode",
            "continuous",
            "--transport-gbps",
            "25",
            "--eager-serial-wire",
            "--verification-mode",
            "final_only",
            "--draft-stage-policy",
            "first",
            "--commit-windows",
            "inf",
            "--paired-fixed-control",
            "--skip-serial",
        ],
    )
    args = pipeline.parse_args()
    assert args.draft_cache_mode == "continuous"
    assert args.stage_fractions == (0.02, 0.5, 1.0)
    assert args.paired_fixed_control is True
    assert args.skip_serial is True
