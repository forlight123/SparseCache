# SPDX-License-Identifier: Apache-2.0
"""Dependency-free LongBench metrics used by the P/D experiment runner.

The dataset-to-metric mapping follows the public LongBench evaluator.  Scores
are kept in [0, 1] so paired per-request deltas can be bootstrapped later.
"""

from __future__ import annotations

from collections import Counter
import difflib
import re
import string

try:
    from fuzzywuzzy import fuzz
except ImportError:  # pragma: no cover - dependency-free mechanism fallback
    fuzz = None

try:
    from rouge import Rouge
except ImportError:  # pragma: no cover - dependency-free mechanism fallback
    Rouge = None


QA_DATASETS = {
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "triviaqa",
}
ROUGE_DATASETS = {"gov_report", "qmsum", "multi_news", "samsum"}
CLASSIFICATION_DATASETS = {"trec", "lsht"}
CODE_DATASETS = {"lcc", "repobench-p"}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, expected: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(expected))


def qa_f1(prediction: str, expected: str) -> float:
    predicted = normalize_answer(prediction).split()
    target = normalize_answer(expected).split()
    common = Counter(predicted) & Counter(target)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(target)
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    prior = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(prior[index - 1] + 1)
            else:
                current.append(max(current[-1], prior[index]))
        prior = current
    return prior[-1]


def rouge_l_f1(prediction: str, expected: str) -> float:
    if Rouge is not None:
        try:
            return float(
                Rouge().get_scores([prediction], [expected], avg=True)["rouge-l"]["f"]
            )
        except (ValueError, AssertionError):
            return 0.0
    predicted = prediction.lower().split()
    target = expected.lower().split()
    if not predicted or not target:
        return 0.0
    overlap = lcs_length(predicted, target)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def retrieval_accuracy(prediction: str, expected: str) -> float:
    matches = re.findall(r"Paragraph (\d+)", expected)
    if not matches:
        return 0.0
    expected_id = matches[0]
    predicted_ids = re.findall(r"\d+", prediction)
    if not predicted_ids:
        return 0.0
    return sum(item == expected_id for item in predicted_ids) / len(predicted_ids)


def count_accuracy(prediction: str, expected: str) -> float:
    predicted_numbers = re.findall(r"\d+", prediction)
    if not predicted_numbers:
        return 0.0
    return sum(item == str(expected) for item in predicted_numbers) / len(
        predicted_numbers
    )


def code_similarity(prediction: str, expected: str) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    if fuzz is not None:
        return fuzz.ratio(candidate, expected) / 100
    return difflib.SequenceMatcher(None, candidate, expected).ratio()


def classification_accuracy(
    prediction: str, expected: str, all_classes: list[str] | None
) -> float:
    matches = [
        class_name
        for class_name in (all_classes or [])
        if class_name in prediction
        and not (class_name in expected and class_name != expected)
    ]
    return 1.0 / len(matches) if expected in matches else 0.0


def longbench_v2_choice(prediction: str) -> str:
    cleaned = prediction.replace("*", "")
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
    ):
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def score_prediction(
    dataset: str,
    prediction: str,
    expected_answers: list[str],
    all_classes: list[str] | None = None,
) -> tuple[str, float, float]:
    """Return metric name, normalized EM, and the official-style task score."""
    if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
        prediction = prediction.lstrip("\n").split("\n")[0]
    if dataset == "longbench_v2":
        predicted = longbench_v2_choice(prediction)
        score = max(
            (float(predicted == str(item).upper()) for item in expected_answers),
            default=0.0,
        )
        return "choice_accuracy", score, score
    em = max((exact_match(prediction, item) for item in expected_answers), default=0.0)
    if dataset in QA_DATASETS:
        metric_name = "qa_f1"
        scorer = lambda expected: qa_f1(prediction, expected)
    elif dataset in ROUGE_DATASETS:
        metric_name = "rouge_l_f1"
        scorer = lambda expected: rouge_l_f1(prediction, expected)
    elif dataset == "passage_retrieval_en":
        metric_name = "retrieval_accuracy"
        scorer = lambda expected: retrieval_accuracy(prediction, expected)
    elif dataset == "passage_count":
        metric_name = "count_accuracy"
        scorer = lambda expected: count_accuracy(prediction, expected)
    elif dataset in CODE_DATASETS:
        metric_name = "code_similarity"
        scorer = lambda expected: code_similarity(prediction, expected)
    elif dataset in CLASSIFICATION_DATASETS:
        metric_name = "classification_accuracy"
        scorer = lambda expected: classification_accuracy(
            prediction, expected, all_classes
        )
    else:
        metric_name = "qa_f1"
        scorer = lambda expected: qa_f1(prediction, expected)
    score = max((scorer(item) for item in expected_answers), default=0.0)
    return metric_name, em, score
