import pytest

from experiments.lossless_pd.lmcache_pd.layer_ready_proxy import (
    external_draft_payload,
    external_draft_request,
    external_token_ids,
    matched_external_suffix,
    parse_bool,
    parse_udp_endpoint,
    prepare_decoder_request,
)


def test_layer_ready_proxy_preserves_exact_p_token_and_progressive_contract():
    request, token = prepare_decoder_request(
        {"prompt": [1, 2, 3], "max_tokens": 8, "kv_transfer_params": {}},
        {"choices": [{"token_ids": [42]}]},
        original_max_tokens=8,
        pd_request_id="pd-9",
        prompt_tokens=3,
    )
    assert token == 42
    assert request["prompt"] == [1, 2, 3, 42]
    assert request["max_tokens"] == 7
    assert request["stream"] is True
    assert request["kv_transfer_params"] == {
        "sparsecache_progressive": {
            "request_id": "pd-9",
            "prompt_tokens": 3,
        }
    }


def test_layer_ready_proxy_rejects_missing_exact_token_ids():
    with pytest.raises(RuntimeError):
        prepare_decoder_request(
            {"prompt": [1], "max_tokens": 8},
            {"choices": [{"text": "ambiguous"}]},
            original_max_tokens=8,
            pd_request_id="pd-9",
            prompt_tokens=1,
        )


def test_layer_ready_proxy_udp_endpoint_validation():
    assert parse_udp_endpoint("udp://127.0.0.1:17610") == ("127.0.0.1", 17610)
    with pytest.raises(ValueError):
        parse_udp_endpoint("tcp://127.0.0.1:17610")


def test_layer_ready_proxy_boolean_contract():
    assert parse_bool("true") is True
    assert parse_bool("0") is False
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_external_draft_request_strips_target_transfer_state():
    request = external_draft_request(
        {
            "model": "target",
            "prompt": [9],
            "max_tokens": 32,
            "kv_transfer_params": {"unsafe": True},
            "stream_options": {"include_usage": True},
        },
        [1, 2, 3],
        model="draft",
        max_tokens=8,
    )

    assert request["model"] == "draft"
    assert request["prompt"] == [1, 2, 3]
    assert request["max_tokens"] == 8
    assert request["temperature"] == 0.0
    assert "kv_transfer_params" not in request


def test_external_draft_payload_preserves_join_and_timing():
    payload = external_draft_payload(
        pd_request_id="42",
        prompt_tokens=8192,
        seed_token_id=11,
        proposals=[17, 19],
        started_ns=1_000_000,
        finished_ns=3_500_000,
        model_ms=2.25,
    )

    assert payload["request_id"] == payload["pd_request_id"] == "42"
    assert payload["wall_ms"] == 2.5
    assert payload["proposals"] == [17, 19]


def test_external_token_ids_fail_closed():
    assert external_token_ids({"choices": [{"token_ids": [3, 5]}]}) == [3, 5]
    with pytest.raises(RuntimeError):
        external_token_ids({"choices": [{"text": "ambiguous"}]})


def test_external_seed_branch_is_reused_only_on_exact_root():
    assert matched_external_suffix(
        [11, 17, 19, 23], seed_token_id=11, max_tokens=2
    ) == [17, 19]
    assert (
        matched_external_suffix([13, 17, 19], seed_token_id=11, max_tokens=2)
        is None
    )
