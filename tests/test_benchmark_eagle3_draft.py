import json
from types import SimpleNamespace

import pytest

from experiments.benchmark_eagle3_draft import (
    bootstrap_mean_ci,
    common_prefix_length,
    draft_weight_file,
    load_prompts,
    parse_horizons,
    spec_metric_snapshot,
    subtract_spec_metrics,
)


def test_parse_horizons_sorts_and_rejects_invalid_values() -> None:
    assert parse_horizons("8,2,4") == (2, 4, 8)
    with pytest.raises(ValueError):
        parse_horizons("4,4")
    with pytest.raises(ValueError):
        parse_horizons("0,4")


def test_common_prefix_length_stops_at_first_mismatch() -> None:
    assert common_prefix_length([1, 2, 9], [1, 2, 3]) == 2
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2


def test_bootstrap_singleton_is_exact() -> None:
    assert bootstrap_mean_ci([3.5]) == (3.5, 3.5)


def test_draft_weight_file_supports_one_known_format(tmp_path) -> None:
    safetensors = tmp_path / "model.safetensors"
    safetensors.write_bytes(b"weights")
    assert draft_weight_file(tmp_path) == safetensors
    (tmp_path / "pytorch_model.bin").write_bytes(b"other")
    with pytest.raises(ValueError):
        draft_weight_file(tmp_path)


def test_load_prompts_retains_variable_aligned_lengths(tmp_path) -> None:
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        "\n".join(
            json.dumps({"prompt": list(range(length))}) for length in (64, 128)
        )
        + "\n",
        encoding="utf-8",
    )
    prompts = load_prompts(
        requests_jsonl=requests,
        request_offset=0,
        num_requests=2,
        max_context_tokens=128,
        block_size=64,
        output_tokens=8,
        model_limit=256,
    )
    assert [row["prompt_tokens"] for row in prompts] == [64, 128]


def test_load_prompts_rejects_head_truncation(tmp_path) -> None:
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps({"prompt": list(range(192))}) + "\n")
    with pytest.raises(ValueError, match="middle-truncated"):
        load_prompts(
            requests_jsonl=requests,
            request_offset=0,
            num_requests=1,
            max_context_tokens=128,
            block_size=64,
            output_tokens=8,
            model_limit=256,
        )


def test_metric_snapshot_and_subtraction() -> None:
    metrics = [
        SimpleNamespace(name="vllm:spec_decode_num_drafts", value=7),
        SimpleNamespace(name="vllm:spec_decode_num_draft_tokens", value=19),
        SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens", value=13),
        SimpleNamespace(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            values=[7, 4, 2],
        ),
    ]
    engine = SimpleNamespace(get_metrics=lambda: metrics)
    after = spec_metric_snapshot(engine)
    before = {
        "drafts": 2,
        "draft_tokens": 5,
        "accepted_tokens": 3,
        "accepted_per_position": [2, 1, 0],
    }
    assert subtract_spec_metrics(after, before) == {
        "drafts": 5,
        "draft_tokens": 14,
        "accepted_tokens": 10,
        "accepted_per_position": [5, 3, 2],
    }
