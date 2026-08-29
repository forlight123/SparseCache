# SPDX-License-Identifier: Apache-2.0
"""Dataset adapters and preparation CLI for SparseCache experiments."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    """One independently reusable RAG document."""

    chunk_id: str
    title: str
    text: str
    supporting: bool | None = None


@dataclass(frozen=True)
class QACase:
    """A question, its gold answers, and an ordered list of document chunks."""

    case_id: str
    source: str
    question: str
    answers: tuple[str, ...]
    chunks: tuple[DocumentChunk, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the case."""
        result = asdict(self)
        result["answers"] = list(self.answers)
        result["chunks"] = [asdict(chunk) for chunk in self.chunks]
        return result


def load_prepared_case(path: str | Path) -> QACase:
    """Load a normalized SparseCache QA case from JSON."""
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    return QACase(
        case_id=str(row["case_id"]),
        source=str(row["source"]),
        question=str(row["question"]),
        answers=tuple(str(answer) for answer in row["answers"]),
        chunks=tuple(DocumentChunk(**chunk) for chunk in row["chunks"]),
    )


def load_hotpotqa_case(
    path: str | Path,
    *,
    index: int,
    max_chunks: int,
) -> QACase:
    """Load one HotpotQA-E/LongBench-style JSONL record."""
    row = _jsonl_row(path, index)
    passages = _split_hotpot_passages(str(row["context"]))
    chunks = []
    for chunk_index, passage in enumerate(passages[:max_chunks]):
        lines = passage.strip().splitlines()
        if lines and re.fullmatch(r"Passage \d+:", lines[0]):
            lines = lines[1:]
        title = lines[0].strip() if lines else f"Passage {chunk_index + 1}"
        text = "\n".join(lines[1:]).strip() if len(lines) > 1 else title
        chunks.append(
            DocumentChunk(
                chunk_id=f"passage-{chunk_index + 1:02d}",
                title=title,
                text=text,
            )
        )
    return QACase(
        case_id=str(row.get("_id", index)),
        source=str(row.get("dataset", "hotpotqa")),
        question=str(row["input"]),
        answers=tuple(str(answer) for answer in row.get("answers", [])),
        chunks=tuple(chunks),
    )


def load_musique_case(
    path: str | Path,
    *,
    index: int,
    max_chunks: int,
) -> QACase:
    """Load one official MuSiQue JSONL record."""
    row = _jsonl_row(path, index)
    chunks = tuple(
        DocumentChunk(
            chunk_id=f"paragraph-{int(paragraph['idx']):02d}",
            title=str(paragraph["title"]),
            text=str(paragraph["paragraph_text"]),
            supporting=bool(paragraph.get("is_supporting", False)),
        )
        for paragraph in row["paragraphs"][:max_chunks]
    )
    answer = row.get("answer")
    aliases = row.get("answer_aliases", [])
    answers = [str(answer)] if answer else []
    answers.extend(str(alias) for alias in aliases if alias not in answers)
    return QACase(
        case_id=str(row.get("id", index)),
        source="musique",
        question=str(row["question"]),
        answers=tuple(answers),
        chunks=chunks,
    )


def main() -> None:
    """Prepare one normalized ten-chunk QA case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "musique"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=10)
    args = parser.parse_args()
    if args.index < 0 or args.max_chunks < 1:
        parser.error("index must be non-negative and max-chunks positive")

    if args.dataset == "hotpotqa":
        case = load_hotpotqa_case(
            args.input,
            index=args.index,
            max_chunks=args.max_chunks,
        )
    else:
        case = load_musique_case(
            args.input,
            index=args.index,
            max_chunks=args.max_chunks,
        )
    if len(case.chunks) != args.max_chunks:
        raise ValueError(
            f"requested {args.max_chunks} chunks but case contains {len(case.chunks)}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(case.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "prepared",
                "output": str(output),
                "case_id": case.case_id,
                "chunks": len(case.chunks),
            },
            ensure_ascii=False,
        )
    )


def _jsonl_row(path: str | Path, index: int) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        for row_index, line in enumerate(source):
            if row_index == index:
                return json.loads(line)
    raise ValueError(f"dataset does not contain row {index}")


def _split_hotpot_passages(context: str) -> list[str]:
    passages = [
        passage
        for passage in re.split(r"(?=^Passage \d+:\n)", context.strip(), flags=re.M)
        if passage.strip()
    ]
    return passages or [context]


if __name__ == "__main__":
    main()
