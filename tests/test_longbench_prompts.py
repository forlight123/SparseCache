import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from longbench_prompts import (
    LONG_BENCH_MAX_NEW_TOKENS,
    LONG_BENCH_PROMPTS,
    LONG_BENCH_V2_PROMPT,
)
from pd_progressive_kv_pipeline import (
    categorical_stratified_indices,
    tokenize_longbench_prompt,
    tokenize_longbench_v2_prompt,
    tokenize_ruler_prompt,
    trim_documents_to_prompt_budget,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(item) for item in token_ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "<user>" + messages[-1]["content"] + "<assistant>"


def test_official_prompt_split_reconstructs_exact_text():
    row = {
        "dataset": "qmsum",
        "context": "abcdef",
        "input": "What happened?",
    }
    tokenizer = CharacterTokenizer()
    prefix, documents, suffix = tokenize_longbench_prompt(
        tokenizer, row, page_tokens=2, prompt_mode="official"
    )
    reconstructed = list(prefix)
    for document in documents:
        reconstructed.extend(document.token_ids)
    reconstructed.extend(suffix)
    expected = LONG_BENCH_PROMPTS["qmsum"].format(**row)
    assert tokenizer.decode(reconstructed) == expected
    assert [len(document.token_ids) for document in documents] == [2, 2, 2]


def test_prompt_and_output_tables_cover_paper_dataset_matrix():
    required = {
        "2wikimqa",
        "hotpotqa",
        "musique",
        "qasper",
        "multifieldqa_en",
        "narrativeqa",
        "qmsum",
        "gov_report",
        "multi_news",
        "passage_retrieval_en",
        "passage_count",
        "lcc",
        "repobench-p",
    }
    assert required <= LONG_BENCH_PROMPTS.keys()
    assert required <= LONG_BENCH_MAX_NEW_TOKENS.keys()


def test_longbench_v2_prompt_and_middle_truncation():
    row = {
        "context": "abcdefghij",
        "question": "Which?",
        "choice_A": "a",
        "choice_B": "b",
        "choice_C": "c",
        "choice_D": "d",
    }
    tokenizer = CharacterTokenizer()
    prefix, documents, suffix = tokenize_longbench_v2_prompt(
        tokenizer,
        row,
        page_tokens=2,
        max_context_tokens=6,
        prompt_mode="official",
        context_truncation="middle",
    )
    context = tokenizer.decode(
        [token for document in documents for token in document.token_ids]
    )
    assert context == "abchij"
    reconstructed = tokenizer.decode(
        list(prefix)
        + [token for document in documents for token in document.token_ids]
        + list(suffix)
    )
    assert reconstructed == LONG_BENCH_V2_PROMPT.format(**{**row, "context": "abchij"})


def test_longbench_v2_sampling_preserves_strata():
    rows = [
        {"domain": domain, "difficulty": difficulty, "length": length}
        for domain in ("qa", "code")
        for difficulty in ("easy", "hard")
        for length in ("short", "long")
        for _ in range(4)
    ]
    selected = categorical_stratified_indices(
        rows, 16, 7, ("domain", "difficulty", "length")
    )
    strata = {
        (rows[index]["domain"], rows[index]["difficulty"], rows[index]["length"])
        for index in selected
    }
    assert len(strata) == 8


def test_ruler_placement_preserves_document_evidence_labels():
    row = {
        "question": "Who?",
        "documents": [
            {"text": "Evidence\nanswer", "supporting": True},
            {"text": "Distractor\nnoise", "supporting": False},
        ],
        "placements": {
            "evidence_last": [1, 0],
        },
    }
    _, documents, _ = tokenize_ruler_prompt(
        CharacterTokenizer(), row, placement="evidence_last"
    )
    assert [document.supporting for document in documents] == [False, True]
    assert "Distractor" in documents[0].text
    assert "Evidence" in documents[1].text


def test_prompt_budget_trims_only_distractors_and_recomputes_positions():
    tokenizer = CharacterTokenizer()
    prefix = tuple(tokenizer.encode("PRE", add_special_tokens=False))
    suffix = tuple(tokenizer.encode("POST", add_special_tokens=False))
    row = {
        "question": "Who?",
        "documents": [
            {"text": "noise-abcdefghij", "supporting": False},
            {"text": "evidence-answer", "supporting": True},
            {"text": "noise-klmnopqrst", "supporting": False},
        ],
        "placements": {"original": [0, 1, 2]},
    }
    _, documents, _ = tokenize_ruler_prompt(tokenizer, row, placement="original")
    supporting_tokens = documents[1].token_ids
    original_distractor_lengths = [
        len(documents[0].token_ids),
        len(documents[2].token_ids),
    ]
    budget = len(prefix) + len(suffix) + len(supporting_tokens) + 20

    trimmed = trim_documents_to_prompt_budget(
        tokenizer, prefix, documents, suffix, budget
    )

    assert sum(len(document.token_ids) for document in trimmed) == budget - 7
    assert trimmed[1].token_ids == supporting_tokens
    assert [document.supporting for document in trimmed] == [False, True, False]
    assert [len(trimmed[0].token_ids), len(trimmed[2].token_ids)] == [10, 10]
    assert all(
        len(trimmed[index].token_ids) < original_distractor_lengths[position]
        for position, index in enumerate((0, 2))
    )
    assert [document.start for document in trimmed] == [
        len(prefix),
        trimmed[0].end,
        trimmed[1].end,
    ]
