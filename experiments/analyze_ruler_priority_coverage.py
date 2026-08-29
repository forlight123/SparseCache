"""Analyze supporting-evidence coverage of frozen RULER page schedules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--selection-requests", type=int, default=100)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def _span_chunks(span: dict[str, Any], chunk_tokens: int) -> set[int]:
    start = int(span["start_token"])
    end = int(span["end_token"])
    if start < 0 or end <= start:
        raise ValueError("document span has invalid token coordinates")
    return set(range(start // chunk_tokens, math.ceil(end / chunk_tokens)))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty coverage split")
    return {
        "requests": len(rows),
        "any_support_chunk_rate": _mean(
            [float(row["selected_support_chunks"] > 0) for row in rows]
        ),
        "all_support_chunks_rate": _mean(
            [float(row["support_chunk_recall"] == 1.0) for row in rows]
        ),
        "mean_support_chunk_recall": _mean(
            [float(row["support_chunk_recall"]) for row in rows]
        ),
        "all_support_documents_touched_rate": _mean(
            [float(row["support_document_recall"] == 1.0) for row in rows]
        ),
        "mean_support_document_recall": _mean(
            [float(row["support_document_recall"]) for row in rows]
        ),
        "all_support_chunks_rank_p50": _percentile(
            [int(row["all_support_chunks_rank"]) for row in rows], 0.50
        ),
        "all_support_chunks_rank_p95": _percentile(
            [int(row["all_support_chunks_rank"]) for row in rows], 0.95
        ),
    }


def analyze_priority_coverage(
    schedule_root: Path,
    *,
    fractions: tuple[float, ...],
    selection_requests: int,
) -> dict[str, Any]:
    if (
        not fractions
        or len(fractions) != len(set(fractions))
        or any(not 0.0 < fraction <= 1.0 for fraction in fractions)
    ):
        raise ValueError("fractions must be unique and lie in (0, 1]")
    audit_path = schedule_root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("priority schedule audit has not passed")
    requests = int(audit["requests"])
    if not 0 < selection_requests < requests:
        raise ValueError("selection split must leave a non-empty confirmation split")
    chunk_tokens = int(audit["chunk_tokens"])
    total_chunks = int(audit["prompt_tokens"]) // chunk_tokens
    modes = tuple(str(mode) for mode in audit["modes"])

    span_rows = _read_jsonl(schedule_root / "document_spans.jsonl")
    if len(span_rows) != requests:
        raise ValueError("document span row count differs from audit")
    priorities = {
        mode: _read_jsonl(schedule_root / f"priority_{mode}.jsonl")
        for mode in modes
    }
    if any(len(rows) != requests for rows in priorities.values()):
        raise ValueError("priority row count differs from audit")

    by_mode: dict[str, dict[str, Any]] = {}
    for mode, priority_rows in priorities.items():
        fraction_reports: dict[str, Any] = {}
        for fraction in fractions:
            budget_chunks = math.ceil(total_chunks * fraction)
            coverage_rows = []
            for request_index, (span_row, priority_row) in enumerate(
                zip(span_rows, priority_rows)
            ):
                if (
                    int(span_row.get("request_index", -1)) != request_index
                    or int(priority_row.get("request_index", -1)) != request_index
                ):
                    raise ValueError("request order differs across scheduling files")
                priority = priority_row.get("priority_chunks")
                if (
                    not isinstance(priority, list)
                    or len(priority) != total_chunks
                    or set(priority) != set(range(total_chunks))
                ):
                    raise ValueError("priority row is not a complete permutation")
                support_documents = [
                    _span_chunks(span, chunk_tokens)
                    for span in span_row["document_spans"]
                    if span.get("supporting") is True
                ]
                if not support_documents:
                    raise ValueError("request has no supporting document span")
                support_chunks = set().union(*support_documents)
                rank = {chunk: index + 1 for index, chunk in enumerate(priority)}
                selected = set(priority[:budget_chunks])
                selected_support = selected & support_chunks
                touched_documents = sum(
                    bool(selected & document) for document in support_documents
                )
                coverage_rows.append(
                    {
                        "request_index": request_index,
                        "support_chunks": len(support_chunks),
                        "selected_support_chunks": len(selected_support),
                        "support_chunk_recall": (
                            len(selected_support) / len(support_chunks)
                        ),
                        "support_document_recall": (
                            touched_documents / len(support_documents)
                        ),
                        "all_support_chunks_rank": max(
                            rank[chunk] for chunk in support_chunks
                        ),
                    }
                )
            fraction_reports[str(fraction)] = {
                "fraction": fraction,
                "budget_chunks": budget_chunks,
                "budget_tokens": budget_chunks * chunk_tokens,
                "all": _summarize(coverage_rows),
                "selection": _summarize(coverage_rows[:selection_requests]),
                "confirmation": _summarize(coverage_rows[selection_requests:]),
            }
        by_mode[mode] = fraction_reports

    return {
        "schema_version": 1,
        "status": "passed",
        "schedule_root": str(schedule_root),
        "requests": requests,
        "selection_rows": [0, selection_requests],
        "confirmation_rows": [selection_requests, requests],
        "prompt_tokens": int(audit["prompt_tokens"]),
        "chunk_tokens": chunk_tokens,
        "total_chunks": total_chunks,
        "fractions": list(fractions),
        "modes": by_mode,
    }


def main() -> None:
    args = parse_args()
    fractions = tuple(
        float(item.strip()) for item in args.fractions.split(",") if item.strip()
    )
    report = analyze_priority_coverage(
        args.schedule_root,
        fractions=fractions,
        selection_requests=args.selection_requests,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
