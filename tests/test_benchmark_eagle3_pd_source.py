import json

import pytest

from experiments.benchmark_eagle3_pd_source import (
    common_prefix_length,
    parse_bandwidths,
    target_kv_bytes_per_token,
    transfer_ms,
)


def test_parse_bandwidths_and_prefix():
    assert parse_bandwidths("100,25,50") == (25.0, 50.0, 100.0)
    assert common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    with pytest.raises(ValueError):
        parse_bandwidths("25,25")
    with pytest.raises(ValueError):
        parse_bandwidths("0")


def test_target_kv_bytes_and_transfer(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            }
        ),
        encoding="utf-8",
    )
    assert target_kv_bytes_per_token(model) == 131_072
    assert transfer_ms(8_000_000_000, 100.0) == pytest.approx(640.0)
