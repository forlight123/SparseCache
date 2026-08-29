import json

import pytest

from experiments.benchmark_layer_subset_draft import load_patterns


def test_load_patterns_validates_frozen_subsets(tmp_path) -> None:
    path = tmp_path / "patterns.json"
    path.write_text(
        json.dumps(
            {"patterns": [{"id": "uniform4", "layer_indices": [0, 10, 20, 31]}]}
        ),
        encoding="utf-8",
    )
    assert load_patterns(path, full_depth=32)[0]["active_layers"] == 4


@pytest.mark.parametrize(
    "indices",
    ([1, 31], [0, 30], [0, 2, 2, 31], [0, 2.0, 31]),
)
def test_load_patterns_rejects_invalid_subsets(tmp_path, indices) -> None:
    path = tmp_path / "patterns.json"
    path.write_text(
        json.dumps({"patterns": [{"id": "bad", "layer_indices": indices}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_patterns(path, full_depth=32)
