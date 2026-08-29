"""Build exact-length controlled RULER packs for SparseCache-PD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--served-model-name", default="sparsecache-llama31-8b")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="pass enable_thinking=False to chat templates such as Qwen3",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_parts(
    tokenizer,
    row: dict[str, Any],
    placement: str,
    *,
    disable_thinking: bool = False,
):
    placeholder = "{DOCUMENTS}"
    user_prompt = (
        "Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\nThe "
        f"following are given documents.\n\n{placeholder}\n\nAnswer the question "
        "based on the given documents. Only give me the answer and do not "
        f"output any other words.\n\nQuestion: {row['question']}"
    )
    template_kwargs = {"enable_thinking": False} if disable_thinking else {}
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    pivot = rendered.index(placeholder)
    prefix = tokenizer.encode(rendered[:pivot], add_special_tokens=False)
    suffix = tokenizer.encode(
        rendered[pivot + len(placeholder) :], add_special_tokens=False
    )
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
                "source_document_index": source_index,
            }
        )
    return prefix, documents, suffix


def _allocate_trim_lengths(documents: list[dict[str, Any]], budget: int) -> list[int]:
    original = [len(document["token_ids"]) for document in documents]
    support = [index for index, item in enumerate(documents) if item["supporting"]]
    distractors = [index for index, item in enumerate(documents) if not item["supporting"]]
    support_tokens = sum(original[index] for index in support)
    distractor_budget = budget - support_tokens
    if distractor_budget < len(distractors):
        raise ValueError("budget cannot preserve supporting documents and distractors")
    removable = sum(original[index] - 1 for index in distractors)
    extra = distractor_budget - len(distractors)
    if extra > removable:
        raise ValueError("trim allocation exceeds original document tokens")
    lengths = original.copy()
    exact = {
        index: extra * (original[index] - 1) / removable for index in distractors
    }
    for index in distractors:
        lengths[index] = 1 + math.floor(exact[index])
    remainder = budget - sum(lengths)
    order = sorted(
        distractors,
        key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
    )
    for index in order[:remainder]:
        lengths[index] += 1
    return lengths


def _allocate_expand_lengths(
    documents: list[dict[str, Any]], budget: int
) -> list[int]:
    original = [len(document["token_ids"]) for document in documents]
    distractors = [index for index, item in enumerate(documents) if not item["supporting"]]
    if not distractors:
        raise ValueError("cannot expand a row without distractor documents")
    deficit = budget - sum(original)
    total_distractor_tokens = sum(original[index] for index in distractors)
    exact = {
        index: deficit * original[index] / total_distractor_tokens
        for index in distractors
    }
    lengths = original.copy()
    for index in distractors:
        lengths[index] += math.floor(exact[index])
    remainder = budget - sum(lengths)
    order = sorted(
        distractors,
        key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
    )
    for index in order[:remainder]:
        lengths[index] += 1
    return lengths


def _fit_documents(documents: list[dict[str, Any]], budget: int):
    original_total = sum(len(item["token_ids"]) for item in documents)
    if original_total > budget:
        lengths = _allocate_trim_lengths(documents, budget)
        mode = "trim_distractors"
    elif original_total < budget:
        lengths = _allocate_expand_lengths(documents, budget)
        mode = "repeat_distractors"
    else:
        lengths = [len(item["token_ids"]) for item in documents]
        mode = "unchanged"
    fitted = []
    for document, length in zip(documents, lengths, strict=True):
        tokens = document["token_ids"]
        if length <= len(tokens):
            fitted.append(tokens[:length])
        else:
            repeats = math.ceil(length / len(tokens))
            fitted.append((tokens * repeats)[:length])
    if sum(map(len, fitted)) != budget:
        raise AssertionError("fitted documents do not fill the exact budget")
    return fitted, original_total, mode


def main() -> None:
    args = parse_args()
    if args.prompt_tokens <= 0 or args.prompt_tokens % 256:
        raise ValueError("prompt_tokens must be a positive multiple of 256")
    if args.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    if args.num_requests is not None and args.num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output_dir.mkdir(parents=True)
    request_path = args.output_dir / "requests.jsonl"
    metadata_path = args.output_dir / "metadata.jsonl"
    rows = 0
    fit_modes: dict[str, int] = {}
    with (
        args.dataset.open(encoding="utf-8") as source,
        request_path.open("w", encoding="utf-8") as requests_file,
        metadata_path.open("w", encoding="utf-8") as metadata_file,
    ):
        for source_index, line in enumerate(source):
            row = json.loads(line)
            prefix, documents, suffix = _render_parts(
                tokenizer,
                row,
                args.placement,
                disable_thinking=args.disable_thinking,
            )
            document_budget = args.prompt_tokens - len(prefix) - len(suffix)
            if document_budget <= 0:
                raise ValueError("prompt target leaves no document budget")
            fitted, original_document_tokens, fit_mode = _fit_documents(
                documents, document_budget
            )
            document_token_spans = []
            cursor = len(prefix)
            for displayed_index, (document, token_ids) in enumerate(
                zip(documents, fitted, strict=True), start=1
            ):
                start = cursor
                cursor += len(token_ids)
                document_token_spans.append(
                    {
                        "displayed_index": displayed_index,
                        "source_document_index": document["source_document_index"],
                        "start_token": start,
                        "end_token": cursor,
                        "supporting": document["supporting"],
                    }
                )
            prompt = prefix + [token for item in fitted for token in item] + suffix
            if len(prompt) != args.prompt_tokens:
                raise AssertionError("RULER prompt length is not exact")
            request = {
                "model": args.served_model_name,
                "prompt": prompt,
                "max_tokens": args.output_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
            }
            metadata = {
                "request_index": rows,
                "source_index": source_index,
                "id": row["id"],
                "dataset": "ruler_qa2",
                "format": "ruler",
                "placement": args.placement,
                "answers": row["answers"],
                "all_classes": None,
                "prompt_tokens": len(prompt),
                "original_document_tokens": original_document_tokens,
                "fitted_document_tokens": document_budget,
                "document_fit_mode": fit_mode,
                "supporting_documents": sum(
                    bool(item["supporting"]) for item in documents
                ),
                "prefix_tokens": len(prefix),
                "suffix_tokens": len(suffix),
                "document_token_spans": document_token_spans,
                "output_tokens": args.output_tokens,
                "fixed_output_horizon": True,
                "alignment_removed_context_tokens": max(
                    0, original_document_tokens - document_budget
                ),
                "alignment_added_newline_tokens": 0,
            }
            requests_file.write(json.dumps(request) + "\n")
            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            rows += 1
            fit_modes[fit_mode] = fit_modes.get(fit_mode, 0) + 1
            if args.num_requests is not None and rows == args.num_requests:
                break
    if rows == 0 or (args.num_requests is not None and rows != args.num_requests):
        raise ValueError("RULER dataset did not provide the requested row count")
    manifest = {
        "schema_version": 1,
        "status": "controlled exact-length P/D pack; no experiment result",
        "dataset": "ruler_qa2",
        "source": str(args.dataset),
        "placement": args.placement,
        "tokenizer": args.tokenizer,
        "served_model_name": args.served_model_name,
        "prompt_mode": "tokenizer_chat_template",
        "thinking_mode": "disabled" if args.disable_thinking else "template_default",
        "requests": rows,
        "prompt_tokens": args.prompt_tokens,
        "chunk_tokens": 256,
        "output_tokens": args.output_tokens,
        "document_fit_modes": fit_modes,
        "requests_sha256": _sha256(request_path),
        "metadata_sha256": _sha256(metadata_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
