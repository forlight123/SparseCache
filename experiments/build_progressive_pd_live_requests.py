"""Build fixed-length token-ID requests for the live SparseCache-PD gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--served-model-name", default="sparsecache-llama31-8b")
    parser.add_argument("--placement", default="evidence_first")
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=104)
    return parser.parse_args()


def _allocate_document_lengths(
    documents: list[dict[str, Any]], budget: int
) -> list[int] | None:
    """Fit every support in full and proportionally trim distractors."""
    original = [len(document["token_ids"]) for document in documents]
    if sum(original) < budget:
        return None
    support_indices = [
        index for index, document in enumerate(documents) if document["supporting"]
    ]
    distractor_indices = [
        index for index, document in enumerate(documents) if not document["supporting"]
    ]
    support_tokens = sum(original[index] for index in support_indices)
    distractor_budget = budget - support_tokens
    if distractor_budget < len(distractor_indices):
        return None
    lengths = original.copy()
    if not distractor_indices:
        return lengths if sum(lengths) == budget else None
    removable = sum(original[index] - 1 for index in distractor_indices)
    extra = distractor_budget - len(distractor_indices)
    if extra > removable:
        return None
    exact = {
        index: extra * (original[index] - 1) / removable
        for index in distractor_indices
    }
    for index in distractor_indices:
        lengths[index] = 1 + math.floor(exact[index])
    remainder = budget - sum(lengths)
    order = sorted(
        distractor_indices,
        key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
    )
    for index in order[:remainder]:
        lengths[index] += 1
    if sum(lengths) != budget:
        raise AssertionError("document allocation did not fill the prompt budget")
    return lengths


def _encode_row(tokenizer, row: dict[str, Any], placement: str, target: int):
    placeholder = "{DOCUMENTS}"
    rendered = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        "You are a helpful assistant<|eot_id|><|start_header_id|>user"
        "<|end_header_id|>Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\nThe "
        f"following are given documents.\n\n{placeholder}\n\nAnswer the question "
        "based on the given documents. Only give me the answer and do not "
        f"output any other words.\n\nQuestion: {row['question']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|> Answer:"
    )
    pivot = rendered.index(placeholder)
    prefix = tokenizer.encode(rendered[:pivot], add_special_tokens=False)
    suffix = tokenizer.encode(
        rendered[pivot + len(placeholder) :], add_special_tokens=False
    )
    document_budget = target - len(prefix) - len(suffix)
    if document_budget <= 0:
        raise ValueError("prompt target leaves no room for documents")
    try:
        order = row["placements"][placement]
    except KeyError as error:
        raise ValueError(f"row does not define placement {placement}") from error
    documents = []
    for displayed_index, source_index in enumerate(order, start=1):
        source = row["documents"][source_index]
        separator = "\n\n" if displayed_index < len(order) else ""
        text = f"Document {displayed_index}:\n{source['text']}{separator}"
        documents.append(
            {
                "token_ids": tokenizer.encode(text, add_special_tokens=False),
                "supporting": bool(source["supporting"]),
            }
        )
    lengths = _allocate_document_lengths(documents, document_budget)
    if lengths is None:
        return None
    prompt = prefix.copy()
    for document, length in zip(documents, lengths, strict=True):
        prompt.extend(document["token_ids"][:length])
    prompt.extend(suffix)
    if len(prompt) != target:
        raise AssertionError("built prompt does not have the requested exact length")
    return prompt


def main() -> None:
    args = parse_args()
    if args.prompt_tokens <= 0 or args.prompt_tokens % 256:
        raise ValueError("prompt_tokens must be a positive multiple of 256")
    if args.max_tokens <= 0 or args.num_requests <= 0:
        raise ValueError("request and output counts must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    requests = []
    skipped = 0
    with args.dataset.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            prompt = _encode_row(tokenizer, row, args.placement, args.prompt_tokens)
            if prompt is None:
                skipped += 1
                continue
            requests.append(
                {
                    "model": args.served_model_name,
                    "prompt": prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream": True,
                    "return_token_ids": True,
                }
            )
            if len(requests) == args.num_requests:
                break
    if len(requests) < args.num_requests:
        raise ValueError(
            f"only {len(requests)} rows fit exactly; requested {args.num_requests}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for request in requests:
            output.write(json.dumps(request) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "requests": len(requests),
                "skipped_short_or_infeasible": skipped,
                "prompt_tokens": args.prompt_tokens,
                "placement": args.placement,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
