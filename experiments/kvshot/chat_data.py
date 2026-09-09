"""Conversation tokenization and supervised-window sampling for KV drafting.

The regenerated ShareGPT records contain complete conversations.  A draft
training position is valid only when every next token in the TTT window belongs
to an assistant response.  This module intentionally keeps that contract
separate from the model/training loop so it can be audited without loading the
Target model.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class ChatTrainingRecord:
    record_id: str
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    candidate_cuts: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.numel())

    @property
    def supervised_tokens(self) -> int:
        return int(self.loss_mask.sum().item())


def _as_token_ids(value: object) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value, list) and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"chat template returned unsupported token container: {type(value)}")
    return [int(token) for token in value]


def tokenize_conversation(
    tokenizer: object,
    conversations: Sequence[dict[str, object]],
    *,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render Qwen-style chat and mark only assistant response tokens.

    Qwen renders historical and final assistant turns differently when thinking
    is disabled, so rendering each prefix independently is not reliable.  We
    instead render the complete conversation once, align its assistant sections
    with the source messages, and convert character boundaries back to token
    boundaries using the same tokenizer.
    """

    if not conversations:
        raise ValueError("conversation is empty")
    messages = [
        {"role": str(message["role"]), "content": str(message.get("content", ""))}
        for message in conversations
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"chat template returned non-text rendering: {type(rendered)}")
    full_ids = _as_token_ids(
        tokenizer(rendered, add_special_tokens=False)["input_ids"]
    )
    full_ids = full_ids[:max_seq_len]
    loss_mask = [0] * len(full_ids)

    assistant_messages = [
        message for message in messages if message["role"] == "assistant"
    ]
    assistant_sections = list(
        re.finditer(
            r"<\|im_start\|>assistant\n([\s\S]*?)<\|im_end\|>\n",
            rendered,
        )
    )
    if len(assistant_sections) != len(assistant_messages):
        raise ValueError(
            "could not align rendered assistant sections with conversation: "
            f"{len(assistant_sections)} sections != {len(assistant_messages)} messages"
        )

    for message, section in zip(assistant_messages, assistant_sections):
        content = message["content"]
        content_offset = section.group(1).find(content)
        if content_offset < 0:
            raise ValueError("assistant content was not found inside its rendered section")
        content_start = section.start(1) + content_offset
        section_end = section.end()
        prefix_ids = _as_token_ids(
            tokenizer(rendered[:content_start], add_special_tokens=False)["input_ids"]
        )
        through_ids = _as_token_ids(
            tokenizer(rendered[:section_end], add_special_tokens=False)["input_ids"]
        )
        prefix_limit = min(len(prefix_ids), len(full_ids))
        through_limit = min(len(through_ids), len(full_ids))
        # Include the assistant end-of-turn marker.  It is part of the Target's
        # generated distribution and therefore a legitimate draft target.
        for token_index in range(prefix_limit, through_limit):
            loss_mask[token_index] = 1

    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(
        loss_mask, dtype=torch.bool
    )


def eligible_cut_positions(
    loss_mask: torch.Tensor,
    *,
    ttt_length: int,
    min_prefix_length: int,
) -> tuple[int, ...]:
    """Return cuts whose next ``ttt_length`` tokens are all supervised."""

    if loss_mask.ndim != 1:
        raise ValueError("loss_mask must be one-dimensional")
    if ttt_length <= 0:
        raise ValueError("ttt_length must be positive")
    cuts = []
    sequence_length = int(loss_mask.numel())
    # A cut denotes the current token.  The model logits at cut+i predict the
    # ground-truth token at cut+i+1.
    for cut in range(max(0, min_prefix_length - 1), sequence_length - ttt_length):
        if bool(loss_mask[cut + 1 : cut + 1 + ttt_length].all()):
            cuts.append(cut)
    return tuple(cuts)


def load_regenerated_chat(
    path: str | Path,
    tokenizer: object,
    *,
    max_seq_len: int,
    ttt_length: int,
    min_prefix_length: int = 8,
    cache_path: str | Path | None = None,
) -> list[ChatTrainingRecord]:
    source = Path(path)
    tokenizer_name = str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__))
    cache_contract = {
        "version": 1,
        "source": str(source.resolve()),
        "source_size": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "tokenizer": tokenizer_name,
        "max_seq_len": max_seq_len,
        "ttt_length": ttt_length,
        "min_prefix_length": min_prefix_length,
    }
    if cache_path is not None and Path(cache_path).exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("contract") == cache_contract:
            return [
                ChatTrainingRecord(
                    record_id=row["record_id"],
                    input_ids=row["input_ids"],
                    loss_mask=row["loss_mask"],
                    candidate_cuts=tuple(row["candidate_cuts"]),
                )
                for row in payload["records"]
            ]

    records = []
    with source.open() as handle:
        for row_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("status", "success") != "success":
                continue
            input_ids, loss_mask = tokenize_conversation(
                tokenizer, row["conversations"], max_seq_len=max_seq_len
            )
            cuts = eligible_cut_positions(
                loss_mask,
                ttt_length=ttt_length,
                min_prefix_length=min_prefix_length,
            )
            if not cuts:
                continue
            records.append(
                ChatTrainingRecord(
                    record_id=str(row.get("id", row_number)),
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    candidate_cuts=cuts,
                )
            )
    if not records:
        raise RuntimeError(f"no eligible assistant windows found in {path}")
    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "contract": cache_contract,
                "records": [
                    {
                        "record_id": record.record_id,
                        "input_ids": record.input_ids,
                        "loss_mask": record.loss_mask,
                        "candidate_cuts": record.candidate_cuts,
                    }
                    for record in records
                ],
            },
            cache,
        )
    return records


def split_records(
    records: Sequence[ChatTrainingRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ChatTrainingRecord], list[ChatTrainingRecord]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    validation_size = max(1, round(len(records) * validation_fraction))
    validation_indices = set(order[:validation_size])
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    if not train:
        raise RuntimeError("validation split consumed every record")
    return train, validation


def record_statistics(records: Iterable[ChatTrainingRecord]) -> dict[str, float | int]:
    materialized = list(records)
    lengths = [record.sequence_length for record in materialized]
    supervised = [record.supervised_tokens for record in materialized]
    candidates = [len(record.candidate_cuts) for record in materialized]
    return {
        "records": len(materialized),
        "tokens": sum(lengths),
        "supervised_tokens": sum(supervised),
        "candidate_windows": sum(candidates),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "mean_supervised_tokens": sum(supervised) / len(supervised),
        "mean_candidate_windows": sum(candidates) / len(candidates),
        "max_sequence_length": max(lengths),
    }
