from argparse import Namespace

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from experiments.benchmark_progressive_pd_live import (
    _execution_plan,
    _parse_arms,
    _read_priority_rows,
    _read_requests,
    _summary,
)
from experiments.progressive_pd_proxy import (
    ProxyConfig,
    _extract_seed_token,
    _prefill_fingerprint,
    _prepare_requests,
    build_config,
)


@pytest.fixture
def config() -> ProxyConfig:
    return ProxyConfig(
        host="127.0.0.1",
        port=8000,
        telemetry_port=5768,
        prefiller_url="http://127.0.0.1:8100",
        decoder_url="http://127.0.0.1:8200",
        default_mode="progressive",
        default_visibility_mode="continuous",
        max_draft_tokens=8,
        start_fraction=0.05,
        require_greedy=True,
    )


def test_progressive_request_passes_exact_seed_only_to_decoder(
    config: ProxyConfig,
) -> None:
    request = {
        "model": "test-model",
        "prompt": "test",
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
        "sparsecache_mode": "progressive",
    }

    prefiller, decoder, mode = _prepare_requests(request, config, 123)

    assert mode == "progressive"
    assert prefiller["max_tokens"] == 1
    assert prefiller["stream"] is False
    assert prefiller["return_token_ids"] is True
    assert decoder["max_tokens"] == 32
    assert decoder["stream"] is True
    assert decoder["kv_transfer_params"]["progressive_sparse_draft"] == {
        "seed_token_id": 123,
        "max_draft_tokens": 8,
        "start_fraction": 0.05,
        "visibility_mode": "continuous",
    }
    assert decoder["kv_transfer_params"]["require_full_remote_kv"] is True
    assert "sparsecache_mode" not in prefiller
    assert "sparsecache_mode" not in decoder


def test_baseline_does_not_enable_progressive_scheduler(config: ProxyConfig) -> None:
    request = {
        "model": "test-model",
        "prompt": "test",
        "max_tokens": 32,
        "temperature": 0,
        "sparsecache_mode": "baseline",
    }

    _, decoder, mode = _prepare_requests(request, config, 123)

    assert mode == "baseline"
    assert decoder["kv_transfer_params"] == {
        "producer_seed_token_id": 123,
        "require_full_remote_kv": True,
    }


def test_visibility_mode_is_request_scoped(config: ProxyConfig) -> None:
    request = {
        "model": "test-model",
        "prompt": "test",
        "temperature": 0,
        "sparsecache_mode": "progressive",
        "sparsecache_visibility_mode": "fixed_s1",
    }

    _, decoder, _ = _prepare_requests(request, config, 123)

    assert (
        decoder["kv_transfer_params"]["progressive_sparse_draft"]["visibility_mode"]
        == "fixed_s1"
    )


def test_request_scoped_priority_reaches_only_decoder(config: ProxyConfig) -> None:
    request = {
        "model": "test-model",
        "prompt": "test",
        "temperature": 0,
        "sparsecache_mode": "progressive",
        "sparsecache_priority_chunks": [3, 0, 2, 1],
    }

    prefiller, decoder, _ = _prepare_requests(request, config, 123)

    assert "sparsecache_priority_chunks" not in prefiller
    assert "sparsecache_priority_chunks" not in decoder
    assert decoder["kv_transfer_params"]["progressive_priority_chunks"] == [
        3,
        0,
        2,
        1,
    ]


def test_paired_prefill_group_is_proxy_owned(config: ProxyConfig) -> None:
    request = {
        "model": "test-model",
        "prompt": "test",
        "temperature": 0,
        "sparsecache_prefill_group": "run:0:0",
    }

    prefiller, decoder, _ = _prepare_requests(request, config, 123)

    assert "sparsecache_prefill_group" not in prefiller
    assert "sparsecache_prefill_group" not in decoder
    assert _prefill_fingerprint("/v1/completions", prefiller) == (
        _prefill_fingerprint("/v1/completions", dict(prefiller))
    )
    assert _prefill_fingerprint("/v1/completions", prefiller) != (
        _prefill_fingerprint("/v1/chat/completions", prefiller)
    )


def test_request_scoped_priority_rejects_duplicates(config: ProxyConfig) -> None:
    with pytest.raises(ValueError, match="unique"):
        _prepare_requests(
            {
                "model": "test-model",
                "prompt": "test",
                "temperature": 0,
                "sparsecache_priority_chunks": [0, 0],
            },
            config,
            123,
        )


def test_proxy_correctness_gate_rejects_sampling(config: ProxyConfig) -> None:
    with pytest.raises(ValueError, match="temperature=0"):
        _prepare_requests(
            {"model": "test-model", "prompt": "test", "temperature": 0.7},
            config,
            123,
        )


def test_extract_seed_token_validates_response_shape() -> None:
    assert _extract_seed_token({"choices": [{"token_ids": [321]}]}) == 321
    with pytest.raises(RuntimeError, match="return_token_ids"):
        _extract_seed_token({"choices": [{"text": "hello"}]})


def test_benchmark_attaches_request_scoped_priority(tmp_path) -> None:
    requests_path = tmp_path / "requests.jsonl"
    priorities_path = tmp_path / "priority.jsonl"
    requests_path.write_text(
        '{"model":"m","prompt":[1],"max_tokens":1}\n', encoding="utf-8"
    )
    priorities_path.write_text(
        '{"request_index":0,"priority_chunks":[1,0]}\n', encoding="utf-8"
    )

    priorities = _read_priority_rows(priorities_path)
    requests = _read_requests(requests_path, 1, priorities=priorities)

    assert requests[0]["sparsecache_priority_chunks"] == [1, 0]


def test_build_config_validates_progressive_controls() -> None:
    args = Namespace(
        host="127.0.0.1",
        port=8000,
        telemetry_port=5768,
        prefiller_url="http://127.0.0.1:8100/",
        decoder_url="http://127.0.0.1:8200/",
        default_mode="progressive",
        default_visibility_mode="continuous",
        max_draft_tokens=8,
        start_fraction=1.0,
        require_greedy=True,
    )
    with pytest.raises(ValueError, match="start_fraction"):
        build_config(args)


def test_live_summary_reports_all_pairwise_method_comparisons() -> None:
    arms = _parse_arms("baseline,fixed_s1,continuous")
    records = []
    for request_index in range(2):
        for arm, latency in (
            ("baseline", 12.0),
            ("fixed_s1", 10.0),
            ("continuous", 8.0),
        ):
            records.append(
                {
                    "request_index": request_index,
                    "source_request_index": request_index,
                    "arm": arm,
                    "decode_completion_ms": latency,
                    "decode_ttft_ms": latency / 2,
                    "token_ids": [1, 2, 3],
                }
            )

    summary = _summary(records, arms)

    assert summary["pairs"] == 2
    assert set(summary["comparisons"]) == {
        "baseline_vs_fixed_s1",
        "baseline_vs_continuous",
        "fixed_s1_vs_continuous",
    }
    assert (
        summary["comparisons"]["fixed_s1_vs_continuous"]["decode_completion_gain_ms"][
            "mean"
        ]
        == 2.0
    )


def test_warmups_replay_without_consuming_measured_rows() -> None:
    requests = [{"row": index} for index in range(3)]

    plan = _execution_plan(requests, warmup_requests=4, source_offset=100)

    assert [(item[0], item[1]) for item in plan] == [
        (None, 100),
        (None, 101),
        (None, 102),
        (None, 100),
        (0, 100),
        (1, 101),
        (2, 102),
    ]
