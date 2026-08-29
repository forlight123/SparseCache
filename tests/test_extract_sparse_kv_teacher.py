import json

import pytest

from experiments.extract_sparse_kv_teacher import (
    load_request_rows,
    parse_int_list,
)


def test_parse_int_list():
    assert parse_int_list("8,20,31", name="layers") == (8, 20, 31)
    with pytest.raises(ValueError):
        parse_int_list("1,1", name="layers")
    with pytest.raises(ValueError):
        parse_int_list("-1", name="layers")


def test_load_request_rows_is_exact_and_bounded(tmp_path):
    source = tmp_path / "requests.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"prompt": list(range(length))}) for length in (4, 8, 12)
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_request_rows(source, (0, 2), 12)
    assert [row["request_index"] for row in rows] == [0, 2]
    with pytest.raises(ValueError):
        load_request_rows(source, (2,), 8)
    with pytest.raises(ValueError):
        load_request_rows(source, (4,), 20)
