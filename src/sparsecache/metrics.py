# SPDX-License-Identifier: Apache-2.0
"""Short-answer metrics used by the RAG probes."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize a short English answer for EM and token F1."""
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, golds: list[str]) -> float:
    """Return one when the normalized prediction matches any gold answer."""
    prediction = normalize_answer(prediction)
    return float(any(prediction == normalize_answer(gold) for gold in golds))


def answer_f1(prediction: str, golds: list[str]) -> float:
    """Return the maximum normalized word-overlap F1 over gold answers."""
    return max((_pair_f1(prediction, gold) for gold in golds), default=0.0)


def clean_short_answer(text: str) -> str:
    """Remove chat terminators and retain the first non-empty answer line."""
    for marker in (
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|im_end|>",
        "<|endoftext|>",
    ):
        text = text.replace(marker, "")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    answer = re.sub(
        r"^(?:final\s+answer|answer)\s*:\s*",
        "",
        lines[0],
        flags=re.IGNORECASE,
    )
    return answer.strip().strip('"').strip()


def _pair_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
