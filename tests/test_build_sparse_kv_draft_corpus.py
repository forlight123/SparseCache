import json

from experiments.build_sparse_kv_draft_corpus import (
    build_corpus,
    length_stratified_indices,
)


def test_length_stratified_indices_cover_length_range():
    requests = [{"prompt": list(range(length))} for length in range(1, 11)]
    selected = length_stratified_indices(requests, 5)
    assert selected == [1, 3, 5, 7, 9]


def test_build_corpus_round_robins_datasets(tmp_path):
    for dataset in ("a", "b"):
        root = tmp_path / dataset
        root.mkdir()
        requests = [
            {"prompt": list(range(index + 1))} for index in range(4)
        ]
        metadata = [
            {"request_index": index, "dataset": dataset}
            for index in range(4)
        ]
        (root / "requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in requests)
        )
        (root / "metadata.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metadata)
        )
        (root / "manifest.json").write_text(
            json.dumps({"dataset": dataset})
        )
    requests, metadata, manifest = build_corpus(
        tmp_path,
        ("a", "b"),
        2,
    )
    assert [row["source_dataset"] for row in requests] == ["a", "b", "a", "b"]
    assert [row["corpus_index"] for row in metadata] == list(range(4))
    assert manifest["requests"] == 4
