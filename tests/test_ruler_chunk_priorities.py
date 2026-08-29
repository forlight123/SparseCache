from experiments.build_ruler_chunk_priorities import (
    bm25_scores,
    document_spans_from_prompt_markers,
    priority_for_mode,
)


def test_bm25_and_oracle_prioritize_relevant_document_chunks() -> None:
    documents = [
        {"text": "irrelevant apples", "supporting": False},
        {"text": "Scott Derrickson nationality", "supporting": True},
    ]
    spans = [
        {"start_token": 256, "end_token": 512},
        {"start_token": 512, "end_token": 768},
    ]
    common = {
        "request_id": "row-1",
        "question": "What is Scott Derrickson nationality?",
        "documents": documents,
        "spans": spans,
        "total_chunks": 6,
        "chunk_tokens": 256,
        "protected_prefix_chunks": 1,
        "protected_suffix_chunks": 1,
        "seed": 7,
    }

    bm25 = priority_for_mode("bm25", **common)
    oracle = priority_for_mode("oracle", **common)

    assert set(bm25[:2]) == {0, 5}
    assert set(oracle[:2]) == {0, 5}
    assert bm25.index(2) < bm25.index(1)
    assert oracle.index(2) < oracle.index(1)
    assert sorted(bm25) == list(range(6))
    assert bm25_scores(common["question"], [item["text"] for item in documents])[1] > 0


def test_random_priority_is_stable_and_protected_first() -> None:
    common = {
        "request_id": "row-9",
        "question": "question",
        "documents": [{"text": "document", "supporting": False}],
        "spans": [{"start_token": 256, "end_token": 1024}],
        "total_chunks": 8,
        "chunk_tokens": 256,
        "protected_prefix_chunks": 1,
        "protected_suffix_chunks": 2,
        "seed": 42,
    }

    first = priority_for_mode("random", **common)
    second = priority_for_mode("random", **common)

    assert first == second
    assert set(first[:3]) == {0, 6, 7}


def test_document_spans_can_be_recovered_from_frozen_prompt_markers() -> None:
    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return list(text.encode())

    source = {
        "placements": {"original": [1, 0]},
        "documents": [
            {"text": "zero", "supporting": False},
            {"text": "one", "supporting": True},
        ],
    }
    prompt = list("prefix Document 1:\none Document 2:\nzero suffix".encode())

    spans, documents = document_spans_from_prompt_markers(
        Tokenizer(), prompt, source, "original", protected_suffix_tokens=7
    )

    assert spans[0]["supporting"] is True
    assert spans[0]["end_token"] == spans[1]["start_token"]
    assert [item["text"] for item in documents] == ["one", "zero"]
