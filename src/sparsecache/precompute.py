# SPDX-License-Identifier: Apache-2.0
"""Offline producer for independently reusable document chunk KV."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_prepared_case
from .prompting import tokenize_case
from .store import model_fingerprint, save_cache, slice_cache


def main() -> None:
    """Precompute one system cache and one S-conditioned KV file per chunk."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; pass --overwrite to replace it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    case = load_prepared_case(args.case)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = tokenize_case(tokenizer, case)
    if len(prompt.chunks) != 10:
        raise ValueError(f"the initial producer requires 10 chunks, got {len(prompt.chunks)}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    started = perf_counter()
    system_output = _prefill(model, prompt.system_ids, device)
    system_cache = slice_cache(
        system_output.past_key_values,
        0,
        len(prompt.system_ids),
    )
    system_bytes = save_cache(output_dir / "system.safetensors", system_cache)
    del system_output, system_cache
    _release(device)

    chunk_rows = []
    for index, (chunk_id, title, token_ids) in enumerate(prompt.chunks):
        chunk_started = perf_counter()
        combined = prompt.system_ids + token_ids
        output = _prefill(model, combined, device)
        cache = slice_cache(
            output.past_key_values,
            len(prompt.system_ids),
            len(combined),
        )
        filename = f"chunk_{index:02d}.safetensors"
        logical_bytes = save_cache(output_dir / filename, cache)
        chunk_rows.append(
            {
                "chunk_id": chunk_id,
                "title": title,
                "file": filename,
                "source_start": len(prompt.system_ids),
                "tokens": len(token_ids),
                "token_ids": list(token_ids),
                "kv_bytes": logical_bytes,
                "producer_ms": (perf_counter() - chunk_started) * 1000,
            }
        )
        print(
            json.dumps(
                {
                    "event": "chunk_saved",
                    "index": index,
                    "title": title,
                    "tokens": len(token_ids),
                    "kv_bytes": logical_bytes,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del output, cache
        _release(device)

    config = model.config
    manifest = {
        "format": "sparsecache.chunk-kv.v1",
        "producer": "independent-system-conditioned",
        "model": {
            "path": str(Path(args.model).resolve()),
            "fingerprint": model_fingerprint(args.model),
            "dtype": args.dtype,
            "layers": int(config.num_hidden_layers),
            "attention_heads": int(config.num_attention_heads),
            "kv_heads": int(config.num_key_value_heads),
            "head_dim": int(config.hidden_size // config.num_attention_heads),
        },
        "case": case.as_dict(),
        "system": {
            "file": "system.safetensors",
            "tokens": len(prompt.system_ids),
            "token_ids": list(prompt.system_ids),
            "kv_bytes": system_bytes,
        },
        "chunks": chunk_rows,
        "online_suffix": {
            "tokens": len(prompt.suffix_ids),
            "token_ids": list(prompt.suffix_ids),
        },
        "full_prompt_tokens": len(prompt.full_ids),
        "document_tokens": len(prompt.document_ids),
        "total_kv_bytes": system_bytes + sum(row["kv_bytes"] for row in chunk_rows),
        "producer_wall_ms": (perf_counter() - started) * 1000,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "manifest": str(manifest_path),
                "chunks": len(chunk_rows),
                "document_tokens": len(prompt.document_ids),
                "total_kv_bytes": manifest["total_kv_bytes"],
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def _prefill(model, token_ids: tuple[int, ...], device: torch.device):
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return model(input_ids=input_ids, use_cache=True, return_dict=True)


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
