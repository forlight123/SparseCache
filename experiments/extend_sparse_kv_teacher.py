# SPDX-License-Identifier: Apache-2.0
"""Extend teacher continuations while reusing previously materialized prompt KV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any

import torch

from experiments.extract_sparse_kv_teacher import (
    _atomic_save,
    _dtype,
    _eos_ids,
    load_request_rows,
    parse_int_list,
    sha256_file,
)
from sparsecache.rope import model_rope_cos_sin
from sparsecache.store import model_fingerprint


@torch.inference_mode()
def extract_trace(
    *,
    model: Any,
    prompt_ids: list[int],
    request_index: int,
    continuation_tokens: int,
    teacher_topk: int,
    seed_layers: tuple[int, ...],
    device: torch.device,
    storage_dtype: torch.dtype,
    output_file: Path,
) -> dict[str, Any]:
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = perf_counter()
    prompt_output = model(
        input_ids=prompt,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    seed_token = int(prompt_output.logits[0, -1].argmax().item())
    eos_ids = _eos_ids(model.config)
    if seed_token in eos_ids:
        raise RuntimeError(f"request {request_index} produces EOS as its seed token")
    cache = prompt_output.past_key_values
    current = seed_token
    input_ids = []
    labels = []
    topk_ids = []
    topk_logprobs = []
    teacher_hidden = []
    continuation_seed_hidden = []
    for _ in range(continuation_tokens):
        step_output = model(
            input_ids=torch.tensor([[current]], dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        cache = step_output.past_key_values
        logits = step_output.logits[0, -1].float()
        logprobs = logits.log_softmax(dim=-1)
        values, ids = logprobs.topk(teacher_topk)
        target = int(logits.argmax().item())
        input_ids.append(current)
        labels.append(target)
        topk_ids.append(ids.to("cpu", dtype=torch.int64))
        topk_logprobs.append(values.to("cpu", dtype=torch.float32))
        teacher_hidden.append(
            step_output.hidden_states[-1][0, -1]
            .detach()
            .to("cpu", dtype=storage_dtype)
        )
        continuation_seed_hidden.append(
            torch.stack(
                [
                    step_output.hidden_states[layer_index + 1][0, -1]
                    .detach()
                    .to("cpu", dtype=storage_dtype)
                    for layer_index in seed_layers
                ]
            )
        )
        current = target
        if target in eos_ids:
            break
    if not labels:
        raise RuntimeError("teacher produced no draft-training labels")
    positions = torch.arange(
        len(prompt_ids),
        len(prompt_ids) + len(input_ids),
        dtype=torch.long,
        device=device,
    )
    query_cos, query_sin = model_rope_cos_sin(
        model,
        positions,
        device=device,
        dtype=storage_dtype,
    )
    tensors = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "teacher_topk_ids": torch.stack(topk_ids),
        "teacher_topk_logprobs": torch.stack(topk_logprobs),
        "teacher_hidden": torch.stack(teacher_hidden),
        "continuation_seed_hidden": torch.stack(continuation_seed_hidden),
        "query_cos": query_cos.to("cpu", dtype=storage_dtype).contiguous(),
        "query_sin": query_sin.to("cpu", dtype=storage_dtype).contiguous(),
    }
    _atomic_save(tensors, output_file)
    return {
        "request_index": request_index,
        "file": output_file.name,
        "file_sha256": sha256_file(output_file),
        "draft_training_tokens": len(labels),
        "seed_token_id": seed_token,
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "logical_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        ),
        "elapsed_ms": (perf_counter() - started) * 1000.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-indices")
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--teacher-topk", type=int, default=64)
    parser.add_argument("--device", default="cuda:1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.continuation_tokens, args.teacher_topk) <= 0:
        raise ValueError("continuation length and teacher top-k must be positive")
    base_manifest_path = Path(args.base_manifest).resolve()
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base.get("format") != "sparsecache.sparse-kv-draft-teacher.v3":
        raise ValueError("trace extension requires a v3 base teacher manifest")
    base_rows = {int(row["request_index"]): row for row in base["samples"]}
    if args.request_indices is None:
        indices = tuple(sorted(base_rows))
    else:
        indices = parse_int_list(args.request_indices, name="request indices")
    missing = set(indices) - set(base_rows)
    if missing:
        raise ValueError(f"base teacher data is missing {sorted(missing)}")
    requests_path = Path(base["requests_jsonl"])
    if sha256_file(requests_path) != base["requests_sha256"]:
        raise ValueError("request corpus hash differs from the base manifest")
    rows = load_request_rows(requests_path, indices, int(base["max_context_tokens"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    storage_dtype = _dtype(str(base["dtype"]))
    model_path = str(base["target_model"])
    if model_fingerprint(model_path) != base["target_fingerprint"]:
        raise ValueError("target model fingerprint differs from the base manifest")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=storage_dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sample_rows = []
    for row in rows:
        request_index = int(row["request_index"])
        trace = extract_trace(
            model=model,
            prompt_ids=list(row["prompt"]),
            request_index=request_index,
            continuation_tokens=args.continuation_tokens,
            teacher_topk=args.teacher_topk,
            seed_layers=tuple(int(value) for value in base["seed_layers"]),
            device=device,
            storage_dtype=storage_dtype,
            output_file=output_dir / f"trace_{request_index:05d}.safetensors",
        )
        base_row = base_rows[request_index]
        if trace["seed_token_id"] != int(base_row["seed_token_id"]):
            raise ValueError(f"request {request_index} seed token is not reproducible")
        if trace["prompt_sha256"] != base_row["prompt_sha256"]:
            raise ValueError(f"request {request_index} prompt hash differs")
        trace["source_dataset"] = base_row.get("source_dataset")
        trace["source_index"] = base_row.get("source_index")
        sample_rows.append(trace)
        print(json.dumps({"event": "teacher_trace", **trace}), flush=True)

    manifest = {
        key: base[key]
        for key in (
            "target_model",
            "target_fingerprint",
            "requests_jsonl",
            "requests_sha256",
            "max_context_tokens",
            "priority_page_size",
            "kv_layers",
            "seed_layers",
            "dtype",
            "head_dim",
            "kv_heads",
            "target_hidden_size",
            "vocab_size",
        )
    }
    manifest.update(
        {
            "schema_version": 5,
            "format": "sparsecache.sparse-kv-draft-teacher.v5",
            "created_at_ns": time_ns(),
            "base_manifest": str(base_manifest_path),
            "request_indices": list(indices),
            "continuation_tokens": args.continuation_tokens,
            "teacher_topk": args.teacher_topk,
            "samples": sample_rows,
            "extractor_sha256": sha256_file(Path(__file__)),
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "teacher_trace_manifest",
                "samples": len(sample_rows),
                "tokens": sum(row["draft_training_tokens"] for row in sample_rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
