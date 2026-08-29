"""Build reproducible, LMCache-aligned LongBench P/D request packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from experiments.longbench_prompts import (
        LONG_BENCH_MAX_NEW_TOKENS,
        LONG_BENCH_PROMPTS,
        LONG_BENCH_V2_MAX_NEW_TOKENS,
        LONG_BENCH_V2_PROMPT,
    )
except ModuleNotFoundError:  # Direct ``python experiments/...`` execution.
    from longbench_prompts import (  # type: ignore[no-redef]
        LONG_BENCH_MAX_NEW_TOKENS,
        LONG_BENCH_PROMPTS,
        LONG_BENCH_V2_MAX_NEW_TOKENS,
        LONG_BENCH_V2_PROMPT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--format", choices=("longbench", "longbench_v2"), default="longbench")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--served-model-name", default="sparsecache-llama31-8b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--fixed-output-tokens", type=int)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="pass enable_thinking=False to chat templates such as Qwen3",
    )
    return parser.parse_args()


def _middle_truncate(tokens: list[int], length: int) -> list[int]:
    if len(tokens) <= length:
        return tokens
    head = math.ceil(length / 2)
    tail = length - head
    return tokens[:head] + (tokens[-tail:] if tail else [])


def align_context_to_chunk(
    prefix: list[int],
    context: list[int],
    suffix: list[int],
    *,
    max_prompt_tokens: int,
    chunk_tokens: int,
    padding_token_id: int = 198,
) -> tuple[list[int], int, int, int]:
    """Trim only context so total prompt is a complete LMCache chunk set.

    Returns the aligned prompt, context tokens removed by the model-length cap,
    the additional 0--(chunk_tokens-1) tokens removed for alignment, and the
    padding tokens added only when trimming would remove the entire context.
    """
    fixed = len(prefix) + len(suffix)
    if fixed >= max_prompt_tokens:
        raise ValueError("prompt scaffolding leaves no context budget")
    capped_context_len = min(len(context), max_prompt_tokens - fixed)
    cap_removed = len(context) - capped_context_len
    capped_context = _middle_truncate(context, capped_context_len)
    total = fixed + len(capped_context)
    aligned_total = total - total % chunk_tokens
    aligned_context_len = aligned_total - fixed
    if aligned_context_len <= 0:
        padded_total = math.ceil(total / chunk_tokens) * chunk_tokens
        if padded_total > max_prompt_tokens:
            raise ValueError("short prompt cannot be padded within the model limit")
        padding_added = padded_total - total
        prompt = prefix + capped_context + [padding_token_id] * padding_added + suffix
        return prompt, cap_removed, 0, padding_added
    alignment_removed = len(capped_context) - aligned_context_len
    aligned_context = _middle_truncate(capped_context, aligned_context_len)
    prompt = prefix + aligned_context + suffix
    if len(prompt) % chunk_tokens:
        raise AssertionError("aligned prompt is not chunk divisible")
    return prompt, cap_removed, alignment_removed, 0


def _chat_template_kwargs(disable_thinking: bool) -> dict[str, bool]:
    return {"enable_thinking": False} if disable_thinking else {}


def _render_longbench(
    tokenizer,
    dataset: str,
    row: dict[str, Any],
    *,
    disable_thinking: bool = False,
):
    placeholder = "{CONTEXT}"
    if dataset not in LONG_BENCH_PROMPTS:
        raise ValueError(f"no official LongBench prompt for {dataset}")
    official = LONG_BENCH_PROMPTS[dataset].format(
        context=placeholder, input=row["input"]
    )
    if dataset not in {"lcc", "repobench-p"}:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": official}],
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs(disable_thinking),
        )
    else:
        rendered = official
    return rendered, placeholder


def _render_longbench_v2(
    tokenizer, row: dict[str, Any], *, disable_thinking: bool = False
):
    placeholder = "{CONTEXT}"
    official = LONG_BENCH_V2_PROMPT.format(
        context=placeholder,
        question=row["question"],
        choice_A=row["choice_A"],
        choice_B=row["choice_B"],
        choice_C=row["choice_C"],
        choice_D=row["choice_D"],
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": official}],
        tokenize=False,
        add_generation_prompt=True,
        **_chat_template_kwargs(disable_thinking),
    )
    return rendered, placeholder


def _encode_parts(tokenizer, rendered: str, placeholder: str, context: str):
    pivot = rendered.index(placeholder)
    prefix = tokenizer.encode(rendered[:pivot], add_special_tokens=False)
    suffix = tokenizer.encode(
        rendered[pivot + len(placeholder) :], add_special_tokens=False
    )
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    return prefix, context_ids, suffix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_dataset(path: Path):
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, list):
            raise TypeError("JSON dataset must contain a list of request objects")
        for row in payload:
            if not isinstance(row, dict):
                raise TypeError("dataset rows must be objects")
            yield row
        return
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("dataset rows must be objects")
            yield row


def main() -> None:
    args = parse_args()
    if args.chunk_tokens <= 0 or args.max_prompt_tokens <= 0:
        raise ValueError("token limits must be positive")
    if args.max_prompt_tokens % args.chunk_tokens:
        raise ValueError("max_prompt_tokens must be divisible by chunk_tokens")
    if args.num_requests is not None and args.num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if args.fixed_output_tokens is not None and args.fixed_output_tokens <= 0:
        raise ValueError("fixed_output_tokens must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_tokens) != 1:
        raise ValueError("tokenizer must encode a newline as one alignment token")
    alignment_padding_token_id = int(newline_tokens[0])
    args.output_dir.mkdir(parents=True)
    request_path = args.output_dir / "requests.jsonl"
    metadata_path = args.output_dir / "metadata.jsonl"
    prompt_lengths = []
    cap_removed_values = []
    alignment_removed_values = []
    alignment_added_values = []
    rows_written = 0

    with (
        request_path.open("w", encoding="utf-8") as requests_file,
        metadata_path.open("w", encoding="utf-8") as metadata_file,
    ):
        for source_index, row in enumerate(_iter_dataset(args.dataset)):
            if args.format == "longbench":
                rendered, placeholder = _render_longbench(
                    tokenizer,
                    args.dataset_name,
                    row,
                    disable_thinking=args.disable_thinking,
                )
                answers = list(row.get("answers") or [])
                output_tokens = (
                    args.fixed_output_tokens
                    or LONG_BENCH_MAX_NEW_TOKENS[args.dataset_name]
                )
            else:
                rendered, placeholder = _render_longbench_v2(
                    tokenizer, row, disable_thinking=args.disable_thinking
                )
                raw_answer = row.get("answer", row.get("answers", []))
                answers = (
                    list(raw_answer)
                    if isinstance(raw_answer, list)
                    else [str(raw_answer)]
                )
                output_tokens = (
                    args.fixed_output_tokens or LONG_BENCH_V2_MAX_NEW_TOKENS
                )
            prefix, context, suffix = _encode_parts(
                tokenizer, rendered, placeholder, str(row["context"])
            )
            prompt, cap_removed, alignment_removed, alignment_added = (
                align_context_to_chunk(
                prefix,
                context,
                suffix,
                max_prompt_tokens=args.max_prompt_tokens,
                chunk_tokens=args.chunk_tokens,
                padding_token_id=alignment_padding_token_id,
                )
            )
            request = {
                "model": args.served_model_name,
                "prompt": prompt,
                "max_tokens": output_tokens,
                "temperature": 0,
                "ignore_eos": args.fixed_output_tokens is not None,
                "stream": True,
                "return_token_ids": True,
            }
            requests_file.write(json.dumps(request) + "\n")
            item_id = row.get("_id", row.get("id", source_index))
            metadata = {
                "request_index": rows_written,
                "source_index": source_index,
                "id": str(item_id),
                "dataset": args.dataset_name,
                "format": args.format,
                "answers": answers,
                "all_classes": row.get("all_classes"),
                "prompt_tokens": len(prompt),
                "original_context_tokens": len(context),
                "cap_removed_context_tokens": cap_removed,
                "alignment_removed_context_tokens": alignment_removed,
                "alignment_added_newline_tokens": alignment_added,
                "output_tokens": output_tokens,
                "fixed_output_horizon": args.fixed_output_tokens is not None,
            }
            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            rows_written += 1
            prompt_lengths.append(len(prompt))
            cap_removed_values.append(cap_removed)
            alignment_removed_values.append(alignment_removed)
            alignment_added_values.append(alignment_added)
            if args.num_requests is not None and rows_written == args.num_requests:
                break

    if rows_written == 0:
        raise ValueError("dataset produced no requests")
    if args.num_requests is not None and rows_written != args.num_requests:
        raise ValueError(
            f"dataset produced {rows_written} requests, expected {args.num_requests}"
        )
    manifest = {
        "schema_version": 1,
        "status": "deterministic P/D request pack; no experiment result",
        "dataset": args.dataset_name,
        "format": args.format,
        "source": str(args.dataset),
        "tokenizer": args.tokenizer,
        "served_model_name": args.served_model_name,
        "prompt_mode": "official_chat",
        "thinking_mode": "disabled" if args.disable_thinking else "template_default",
        "context_truncation": "middle",
        "transport_alignment": (
            "trim_context_down_to_complete_lmcache_chunks; pad only when "
            "down-alignment would remove all context"
        ),
        "alignment_padding_token_id": alignment_padding_token_id,
        "chunk_tokens": args.chunk_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "fixed_output_tokens": args.fixed_output_tokens,
        "requests": rows_written,
        "prompt_tokens": {
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "mean": sum(prompt_lengths) / rows_written,
        },
        "alignment_removed_context_tokens": {
            "max": max(alignment_removed_values),
            "mean": sum(alignment_removed_values) / rows_written,
        },
        "alignment_added_newline_tokens": {
            "max": max(alignment_added_values),
            "mean": sum(alignment_added_values) / rows_written,
            "affected_requests": sum(value > 0 for value in alignment_added_values),
        },
        "cap_removed_context_tokens": {
            "max": max(cap_removed_values),
            "mean": sum(cap_removed_values) / rows_written,
        },
        "requests_sha256": _sha256(request_path),
        "metadata_sha256": _sha256(metadata_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
