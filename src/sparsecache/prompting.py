# SPDX-License-Identifier: Apache-2.0
"""Stable prompt assembly for independently precomputed document chunks."""

from __future__ import annotations

from dataclasses import dataclass

from .data import QACase

PLACEHOLDER = "<|SPARSECACHE_DOCUMENTS|>"
SYSTEM_INSTRUCTION = (
    "You are a careful question-answering assistant. Answer the user's question "
    "using the supplied documents. Return only the exact short answer without "
    "explanation."
)


@dataclass(frozen=True)
class TokenizedPrompt:
    """Token IDs for the system prefix, chunks, and online question suffix."""

    system_ids: tuple[int, ...]
    chunks: tuple[tuple[str, str, tuple[int, ...]], ...]
    suffix_ids: tuple[int, ...]

    @property
    def document_ids(self) -> tuple[int, ...]:
        """Return all document token IDs in their target order."""
        return tuple(token for _, _, ids in self.chunks for token in ids)

    @property
    def full_ids(self) -> tuple[int, ...]:
        """Return the exact full-prompt token sequence."""
        return self.system_ids + self.document_ids + self.suffix_ids


def tokenize_case(tokenizer, case: QACase) -> TokenizedPrompt:
    """Build a deterministic chat prompt while preserving chunk boundaries."""
    user = (
        f"<DOCUMENTS>\n{PLACEHOLDER}\n</DOCUMENTS>\n\n"
        f"Question: {case.question.strip()}\n"
        "Return only the exact short answer without explanation."
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    pivot = rendered.index(PLACEHOLDER)
    system_ids = tokenizer.encode(rendered[:pivot], add_special_tokens=True)
    suffix_ids = tokenizer.encode(
        rendered[pivot + len(PLACEHOLDER) :],
        add_special_tokens=False,
    )
    chunks = []
    for index, chunk in enumerate(case.chunks, 1):
        text = f"Document {index}: {chunk.title}\n{chunk.text.strip()}\n"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks.append((chunk.chunk_id, chunk.title, tuple(int(x) for x in token_ids)))
    return TokenizedPrompt(
        system_ids=tuple(int(x) for x in system_ids),
        chunks=tuple(chunks),
        suffix_ids=tuple(int(x) for x in suffix_ids),
    )
