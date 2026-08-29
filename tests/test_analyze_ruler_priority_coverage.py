import json
from pathlib import Path

from experiments.analyze_ruler_priority_coverage import analyze_priority_coverage


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_priority_coverage_reports_disjoint_split_metrics(tmp_path: Path) -> None:
    audit = {
        "status": "passed",
        "requests": 2,
        "prompt_tokens": 1024,
        "chunk_tokens": 256,
        "modes": ["weak", "oracle"],
    }
    (tmp_path / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    spans = [
        {
            "request_index": index,
            "document_spans": [
                {
                    "start_token": 256,
                    "end_token": 512,
                    "supporting": True,
                },
                {
                    "start_token": 768,
                    "end_token": 1024,
                    "supporting": True,
                },
            ],
        }
        for index in range(2)
    ]
    _write_jsonl(tmp_path / "document_spans.jsonl", spans)
    _write_jsonl(
        tmp_path / "priority_weak.jsonl",
        [
            {"request_index": 0, "priority_chunks": [0, 1, 2, 3]},
            {"request_index": 1, "priority_chunks": [0, 2, 1, 3]},
        ],
    )
    _write_jsonl(
        tmp_path / "priority_oracle.jsonl",
        [
            {"request_index": index, "priority_chunks": [1, 3, 0, 2]}
            for index in range(2)
        ],
    )

    report = analyze_priority_coverage(
        tmp_path, fractions=(0.25, 0.5), selection_requests=1
    )

    weak = report["modes"]["weak"]["0.25"]
    oracle = report["modes"]["oracle"]["0.5"]
    assert weak["selection"]["any_support_chunk_rate"] == 0.0
    assert weak["confirmation"]["any_support_chunk_rate"] == 0.0
    assert oracle["all"]["all_support_chunks_rate"] == 1.0
    assert oracle["all"]["all_support_documents_touched_rate"] == 1.0
