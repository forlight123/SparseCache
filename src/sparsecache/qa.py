# SPDX-License-Identifier: Apache-2.0
"""Online QA consumer for direct reuse and CacheBlend-style recomputation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import answer_em, answer_f1, clean_short_answer
from .modeling import baseline_generate, cacheblend_generate, direct_reuse_generate
from .store import cache_bytes, compose_cache, load_manifest, model_fingerprint


def main() -> None:
    """Load a ten-chunk CPU cache and run paired QA modes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("baseline", "direct", "cacheblend"),
        default=("direct", "cacheblend"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--check-layer", type=int, default=1)
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")

    manifest = load_manifest(args.manifest)
    model_path = manifest["model"]["path"]
    if model_fingerprint(model_path) != manifest["model"]["fingerprint"]:
        raise RuntimeError("model files do not match the cache manifest")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = getattr(torch, manifest["model"]["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    compose_started = perf_counter()
    prefix_cache, layout = compose_cache(model, manifest)
    compose_ms = (perf_counter() - compose_started) * 1000
    system_ids = tuple(int(x) for x in manifest["system"]["token_ids"])
    document_ids = tuple(
        int(token)
        for chunk in manifest["chunks"]
        for token in chunk["token_ids"]
    )
    suffix_ids = tuple(
        int(x) for x in manifest["online_suffix"]["token_ids"]
    )
    full_ids = system_ids + document_ids + suffix_ids
    eos_ids = _eos_ids(model, tokenizer)
    case = manifest["case"]
    rows = []

    for mode in args.modes:
        if mode == "baseline":
            result = baseline_generate(
                model,
                full_ids=full_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_ids,
            )
        elif mode == "direct":
            result = direct_reuse_generate(
                model,
                prefix_cache=prefix_cache,
                suffix_ids=suffix_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_ids,
            )
        else:
            result = cacheblend_generate(
                model,
                full_ids=full_ids,
                prefix_cache=prefix_cache,
                system_tokens=layout["system_tokens"],
                suffix_tokens=len(suffix_ids),
                check_layer=args.check_layer,
                recompute_ratio=args.recompute_ratio,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_ids,
            )
        text = clean_short_answer(
            tokenizer.decode(result.token_ids, skip_special_tokens=False)
        )
        row = {
            "mode": mode,
            "question": case["question"],
            "gold_answers": case["answers"],
            "answer": text,
            "token_ids": list(result.token_ids),
            "em": answer_em(text, case["answers"]),
            "f1": answer_f1(text, case["answers"]),
            "prefill_ms": result.prefill_ms,
            "decode_ms": result.decode_ms,
            "metadata": result.metadata,
        }
        rows.append(row)
        print(json.dumps({"event": "qa", **row}, ensure_ascii=False), flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "manifest": str(Path(args.manifest).resolve()),
        "model": model_path,
        "case_id": case["case_id"],
        "chunks": len(manifest["chunks"]),
        "system_tokens": layout["system_tokens"],
        "document_tokens": layout["document_tokens"],
        "suffix_tokens": len(suffix_ids),
        "relocated_tokens": layout["relocated_tokens"],
        "prefix_kv_bytes": cache_bytes(prefix_cache),
        "compose_ms": compose_ms,
        "results": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"event": "complete", "output": str(output_path)},
            ensure_ascii=False,
        ),
        flush=True,
    )


def _eos_ids(model, tokenizer) -> set[int]:
    value = model.generation_config.eos_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token) for token in value}


if __name__ == "__main__":
    main()
