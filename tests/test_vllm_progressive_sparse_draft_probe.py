import json
from types import SimpleNamespace

from experiments.vllm_progressive_sparse_draft_probe import (
    launch_worker,
    parse_completion_trace,
    subtract_metrics,
    token_agreement,
)


def test_subtract_metrics_reports_acceptance() -> None:
    before = {
        "vllm:spec_decode_num_drafts": 2,
        "vllm:spec_decode_num_accepted_tokens": 5,
        "vllm:spec_decode_num_accepted_tokens_per_pos": [2, 2, 1],
    }
    after = {
        "vllm:spec_decode_num_drafts": 4,
        "vllm:spec_decode_num_accepted_tokens": 8,
        "vllm:spec_decode_num_accepted_tokens_per_pos": [4, 3, 1],
    }
    result = subtract_metrics(after, before)
    assert result["mean_acceptance_length"] == 2.5
    assert result["acceptance_rate_per_position"] == [1.0, 0.5, 0.0]


def test_token_agreement_charges_length_mismatch() -> None:
    assert token_agreement([1, 2, 3], [1, 9]) == 1 / 3


def test_completion_trace_parser_matches_times_to_fractions() -> None:
    assert parse_completion_trace("0.02,0.5,1", "0,4,9") == (
        (0.0, 0.02),
        (4.0, 0.5),
        (9.0, 1.0),
    )


def test_worker_launch_wires_completion_and_external_handoff(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.json"
    prompt.write_text(json.dumps({"doc_start_token": 8, "doc_end_token": 80}))
    args = SimpleNamespace(
        dataset="dataset.jsonl",
        model="model",
        output_dir=str(tmp_path),
        offset=0,
        max_prompt_tokens=128,
        max_new_tokens=5,
        draft_tokens=4,
        block_size=8,
        visible_fractions="0.1,1.0",
        completion_trace_ms="0,4",
        completion_trace=((0.0, 0.1), (4.0, 1.0)),
        page_order="uniform",
        gpu_memory_utilization=0.8,
        cuda_visible_devices="0",
        require_exclusive_gpu=True,
    )
    calls = []
    monkeypatch.setattr(
        "experiments.vllm_progressive_sparse_draft_probe.subprocess.run",
        lambda command, *, env, check: calls.append((command, env, check)),
    )

    launch_worker(
        args,
        "draft_only",
        prompt,
        tmp_path / "draft.json",
        tmp_path / "draft_stats.jsonl",
    )
    launch_worker(
        args,
        "external_verify",
        prompt,
        tmp_path / "verify.json",
        tmp_path / "verify_stats.jsonl",
    )

    draft_command, draft_env, _ = calls[0]
    verify_command, verify_env, _ = calls[1]
    assert "--completion-file" in draft_command
    assert draft_env["VLLM_PROGRESSIVE_KV_COMPLETION_PATH"].endswith(
        "draft_only_completion.json"
    )
    assert "--proposal-file" in verify_command
    assert verify_env["VLLM_PROGRESSIVE_EXTERNAL_DRAFT_PATH"].endswith(
        "external_proposal.json"
    )
