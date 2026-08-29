# SPDX-License-Identifier: Apache-2.0
"""Restore document/evidence metadata to generated RULER qa_2 prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


DOCUMENT_PATTERN = re.compile(r"(?:^|\n\n)Document (\d+):\n")
CONTEXT_START = "The following are given documents.\n\n"
CONTEXT_END = (
    "\n\nAnswer the question based on the given documents. Only give me the "
    "answer and do not output any other words.\n\nQuestion: "
)


def parse_documents(prompt):
    start = prompt.index(CONTEXT_START) + len(CONTEXT_START)
    end = prompt.index(CONTEXT_END, start)
    body = prompt[start:end]
    matches = list(DOCUMENT_PATTERN.finditer(body))
    documents = []
    for index, match in enumerate(matches):
        document_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[match.end():document_end].strip()
        title = text.splitlines()[0].strip()
        documents.append({"title": title, "text": text})
    return documents


def uniform_order(supporting, other, total):
    result = list(other)
    for index, document in enumerate(supporting):
        target = round((index + 1) * (total - 1) / (len(supporting) + 1))
        result.insert(min(target, len(result)), document)
    return result


def placement_orders(documents):
    supporting = [index for index, item in enumerate(documents) if item["supporting"]]
    other = [index for index, item in enumerate(documents) if not item["supporting"]]
    adversarial = (
        [supporting[0], *other, *supporting[1:]]
        if len(supporting) > 1
        else [*other, *supporting]
    )
    return {
        "original": list(range(len(documents))),
        "evidence_first": [*supporting, *other],
        "evidence_uniform": uniform_order(supporting, other, len(documents)),
        "evidence_last": [*other, *supporting],
        "adversarial_split": adversarial,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-hotpot", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lengths", default="16384,32768,65536,131072")
    return parser.parse_args()


def main():
    args = parse_args()
    source = json.loads(Path(args.source_hotpot).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    audit = {}
    for length in (int(item) for item in args.lengths.split(",")):
        input_path = Path(args.input_root) / str(length) / "qa_2" / "validation.jsonl"
        rows = [json.loads(line) for line in input_path.open(encoding="utf-8")]
        output_path = output_root / f"qa2_{length}.jsonl"
        normalized = []
        supporting_counts = []
        for row in rows:
            source_row = source[row["index"]]
            documents = parse_documents(row["input"])
            supporting_titles = {item[0] for item in source_row["supporting_facts"]}
            for document in documents:
                document["supporting"] = document["title"] in supporting_titles
            found = {item["title"] for item in documents if item["supporting"]}
            if found != supporting_titles:
                raise ValueError(
                    f"row {row['index']} evidence mismatch: {found} != {supporting_titles}"
                )
            question_marker = CONTEXT_END
            question_start = row["input"].index(question_marker) + len(question_marker)
            question_end = row["input"].index("<|eot_id|>", question_start)
            question = row["input"][question_start:question_end].strip()
            if question != source_row["question"].strip():
                raise ValueError(f"row {row['index']} question mismatch")
            supporting_counts.append(len(found))
            normalized.append(
                {
                    "id": f"ruler_qa2_{length}_{row['index']}",
                    "index": row["index"],
                    "dataset": "ruler_qa2",
                    "target_length": length,
                    "source_length_tokens": row["length"],
                    "question": question,
                    "answers": list(row["outputs"]),
                    "documents": documents,
                    "placements": placement_orders(documents),
                }
            )
        with output_path.open("w", encoding="utf-8") as stream:
            for row in normalized:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        audit[str(length)] = {
            "input": str(input_path.resolve()),
            "output": str(output_path.resolve()),
            "rows": len(normalized),
            "mean_documents": sum(len(row["documents"]) for row in normalized)
            / len(normalized),
            "supporting_document_counts": sorted(set(supporting_counts)),
            "placements": sorted(normalized[0]["placements"]),
        }
    audit_path = output_root / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
