import pytest

from experiments.lossless_pd.lmcache_pd.runahead_token_proxy import (
    prepare_decoder_request,
    runahead_budget,
)


def test_runahead_budget_is_bounded_by_client_request():
    assert runahead_budget(8, 4) == 4
    assert runahead_budget(2, 4) == 2
    with pytest.raises(ValueError):
        runahead_budget(0, 4)


def test_decoder_continues_after_exact_p_token_ids():
    request, token_ids = prepare_decoder_request(
        {
            "prompt": [1, 2, 3],
            "max_tokens": 4,
            "stream": False,
            "return_token_ids": True,
            "kv_transfer_params": {"ret_first_tok": True},
        },
        {
            "choices": [
                {
                    "text": "answer",
                    "token_ids": [7, 8],
                    "finish_reason": "length",
                }
            ]
        },
        original_max_tokens=8,
    )
    assert token_ids == [7, 8]
    assert request["prompt"] == [1, 2, 3, 7, 8]
    assert request["max_tokens"] == 6
    assert request["stream"] is True
    assert "kv_transfer_params" not in request
    assert request["return_token_ids"] is True


def test_p_eos_prevents_decoder_from_generating_more_tokens():
    request, _ = prepare_decoder_request(
        {"prompt": [1], "max_tokens": 4},
        {
            "choices": [
                {"text": "done", "token_ids": [2], "finish_reason": "stop"}
            ]
        },
        original_max_tokens=8,
    )
    assert request["max_tokens"] == 0


def test_missing_exact_ids_fails_closed():
    with pytest.raises(RuntimeError, match="exact generated token IDs"):
        prepare_decoder_request(
            {"prompt": [1]},
            {"choices": [{"text": "x", "token_ids": None}]},
            original_max_tokens=8,
        )
