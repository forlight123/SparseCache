from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from longbench_metrics import score_prediction


def test_qa_and_rouge_metrics():
    name, em, score = score_prediction(
        "2wikimqa", "The Eiffel Tower", ["Eiffel Tower"]
    )
    assert name == "qa_f1"
    assert em == 1.0
    assert score == 1.0

    name, _, score = score_prediction(
        "qmsum", "alpha beta gamma", ["alpha gamma"]
    )
    assert name == "rouge_l_f1"
    assert score == pytest.approx(0.8, abs=1e-7)


def test_retrieval_count_and_code_metrics():
    name, _, score = score_prediction(
        "passage_retrieval_en", "Paragraph 7", ["Paragraph 7"]
    )
    assert name == "retrieval_accuracy"
    assert score == 1.0

    name, _, score = score_prediction("passage_count", "There are 12.", ["12"])
    assert name == "count_accuracy"
    assert score == 1.0

    name, _, score = score_prediction("lcc", "value = item", ["value = item"])
    assert name == "code_similarity"
    assert score == 1.0


def test_longbench_v2_choice_accuracy():
    name, em, score = score_prediction(
        "longbench_v2",
        "After checking the text, The correct answer is (C)",
        ["C"],
    )
    assert name == "choice_accuracy"
    assert em == 1.0
    assert score == 1.0
