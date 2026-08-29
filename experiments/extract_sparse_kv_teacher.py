# SPDX-License-Identifier: Apache-2.0
"""Materialize exact target KV and greedy teacher traces for drafter training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from sparsecache.rope import apply_rope, model_rope_cos_sin
from sparsecache.store import cache_to_legacy, model_fingerprint


def parse_int_list(raw: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_request_rows(
    path: Path, indices: tuple[int, ...], max_context_tokens: int
) -> list[dict[str, Any]]:
    requested = set(indices)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index not in requested:
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt")
            if (
                not isinstance(prompt, list)
                or not prompt
                or any(type(token) is not int for token in prompt)
            ):
                raise ValueError(f"request {index} has invalid prompt token IDs")
            if len(prompt) > max_context_tokens:
                raise ValueError(
                    f"request {index} has {len(prompt)} prompt tokens, exceeding "
                    f"the frozen {max_context_tokens} cap"
                )
            rows.append(
                {
                    "request_index": index,
                    "prompt": prompt,
                    "source_dataset": payload.get("source_dataset"),
                    "source_index": payload.get("source_index"),
                }
            )
    observed = {int(row["request_index"]) for row in rows}
    if observed != requested:
        raise ValueError(f"missing request indices: {sorted(requested - observed)}")
    return sorted(rows, key=lambda row: int(row["request_index"]))


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _eos_ids(config: Any) -> set[int]:
    raw = config.eos_token_id
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {raw}
    return {int(token) for token in raw}


def _atomic_save(tensors: dict[str, torch.Tensor], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(tensors, temporary)
    temporary.replace(destination)


@torch.inference_mode()
def extract_one(
    *,
    model: Any,
    prompt_ids: list[int],
    request_index: int,
    kv_layers: tuple[int, ...],
    seed_layers: tuple[int, ...],
    continuation_tokens: int,
    teacher_topk: int,
    device: torch.device,
    storage_dtype: torch.dtype,
    priority_page_size: int,
    output_file: Path,
) -> dict[str, Any]:
    captured: dict[int, torch.Tensor] = {}
    query_inputs: dict[int, torch.Tensor] = {}
    handles = []

    def capture(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = hidden[:, -1].detach()

        return hook

    def capture_query(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            query_inputs[layer_index] = output[:, -1:].detach()

        return hook

    for layer_index in seed_layers:
        handles.append(
            model.model.layers[layer_index].register_forward_hook(
                capture(layer_index)
            )
        )
    for layer_index in kv_layers:
        handles.append(
            model.model.layers[layer_index]
            .input_layernorm.register_forward_hook(
                capture_query(layer_index)
            )
        )

    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = perf_counter()
    prompt_output = model(
        input_ids=prompt,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    for handle in handles:
        handle.remove()
    if set(captured) != set(seed_layers):
        raise RuntimeError("not every requested seed hidden state was captured")
    if set(query_inputs) != set(kv_layers):
        raise RuntimeError("not every requested priority query was captured")

    legacy_cache = cache_to_legacy(prompt_output.past_key_values)
    memory_keys = torch.stack(
        [
            legacy_cache[layer][0][0]
            .detach()
            .to("cpu", dtype=storage_dtype)
            for layer in kv_layers
        ]
    ).contiguous()
    memory_values = torch.stack(
        [
            legacy_cache[layer][1][0]
            .detach()
            .to("cpu", dtype=storage_dtype)
            for layer in kv_layers
        ]
    ).contiguous()
    seed_hidden = torch.stack(
        [
            captured[layer][0].detach().to("cpu", dtype=storage_dtype)
            for layer in seed_layers
        ]
    ).contiguous()

    priority_layers = []
    final_position = torch.tensor(
        [len(prompt_ids) - 1],
        dtype=torch.long,
        device=device,
    )
    priority_cos, priority_sin = model_rope_cos_sin(
        model,
        final_position,
        device=device,
        dtype=storage_dtype,
    )
    for layer_index in kv_layers:
        attention = model.model.layers[layer_index].self_attn
        hidden = query_inputs[layer_index]
        query = attention.q_proj(hidden).view(
            1,
            1,
            int(model.config.num_attention_heads),
            int(model.config.head_dim),
        )
        query = query.transpose(1, 2)
        if hasattr(attention, "q_norm"):
            query = attention.q_norm(query)
        query = apply_rope(
            query,
            priority_cos,
            priority_sin,
            sequence_dim=2,
        )
        key = legacy_cache[layer_index][0]
        groups = query.shape[1] // key.shape[1]
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
        scaling = float(
            getattr(attention, "scaling", attention.head_dim**-0.5)
        )
        logits = torch.einsum(
            "bhqd,bhkd->bhqk",
            query.float(),
            key.float(),
        ) * scaling
        priority_layers.append(logits.softmax(dim=-1).mean(dim=(1, 2))[0])
    token_priority = torch.stack(priority_layers).mean(dim=0)
    num_pages = (
        len(prompt_ids) + priority_page_size - 1
    ) // priority_page_size
    padded_tokens = num_pages * priority_page_size
    if padded_tokens != len(prompt_ids):
        token_priority = F.pad(
            token_priority,
            (0, padded_tokens - len(prompt_ids)),
        )
    priority_page_scores = token_priority.view(
        num_pages,
        priority_page_size,
    ).sum(dim=1)

    seed_token = int(prompt_output.logits[0, -1].argmax().item())
    if seed_token in _eos_ids(model.config):
        raise RuntimeError(f"request {request_index} produces EOS as its seed token")
    cache = prompt_output.past_key_values
    current = seed_token
    input_ids = []
    labels = []
    topk_ids = []
    topk_logprobs = []
    teacher_hidden = []
    eos_ids = _eos_ids(model.config)
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
        "memory_keys": memory_keys,
        "memory_values": memory_values,
        "seed_hidden": seed_hidden,
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "teacher_topk_ids": torch.stack(topk_ids),
        "teacher_topk_logprobs": torch.stack(topk_logprobs),
        "teacher_hidden": torch.stack(teacher_hidden),
        "priority_page_scores": priority_page_scores.to(
            "cpu", dtype=torch.float32
        ).contiguous(),
        "query_cos": query_cos.to("cpu", dtype=storage_dtype).contiguous(),
        "query_sin": query_sin.to("cpu", dtype=storage_dtype).contiguous(),
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int32),
    }
    _atomic_save(tensors, output_file)
    logical_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    return {
        "request_index": request_index,
        "file": output_file.name,
        "prompt_tokens": len(prompt_ids),
        "draft_training_tokens": len(labels),
        "seed_token_id": seed_token,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "logical_bytes": logical_bytes,
        "file_sha256": sha256_file(output_file),
        "elapsed_ms": (perf_counter() - started) * 1000.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-indices", default="0,1,5,8")
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--continuation-tokens", type=int, default=8)
    parser.add_argument("--kv-layers", default="8,20,31")
    parser.add_argument("--seed-layers", default="2,16,29")
    parser.add_argument("--teacher-topk", type=int, default=64)
    parser.add_argument("--priority-page-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    indices = parse_int_list(args.request_indices, name="request indices")
    kv_layers = parse_int_list(args.kv_layers, name="KV layers")
    seed_layers = parse_int_list(args.seed_layers, name="seed layers")
    if (
        min(
            args.max_context_tokens,
            args.continuation_tokens,
            args.teacher_topk,
            args.priority_page_size,
        )
        <= 0
    ):
        raise ValueError("context, continuation, and top-k sizes must be positive")
    requests_path = Path(args.requests_jsonl)
    rows = load_request_rows(requests_path, indices, args.max_context_tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = int(model.config.num_hidden_layers)
    if max((*kv_layers, *seed_layers)) >= layers:
        raise ValueError("requested target layer is out of range")
    if len(kv_layers) == 0 or len(seed_layers) == 0:
        raise ValueError("at least one KV and seed layer is required")

    sample_rows = []
    for row in rows:
        request_index = int(row["request_index"])
        sample = extract_one(
            model=model,
            prompt_ids=list(row["prompt"]),
            request_index=request_index,
            kv_layers=kv_layers,
            seed_layers=seed_layers,
            continuation_tokens=args.continuation_tokens,
            teacher_topk=args.teacher_topk,
            device=device,
            storage_dtype=dtype,
            priority_page_size=args.priority_page_size,
            output_file=output_dir / f"sample_{request_index:05d}.safetensors",
        )
        sample["source_dataset"] = row.get("source_dataset")
        sample["source_index"] = row.get("source_index")
        sample_rows.append(sample)
        print(json.dumps({"event": "teacher_sample", **sample}), flush=True)

    manifest = {
        "schema_version": 3,
        "format": "sparsecache.sparse-kv-draft-teacher.v3",
        "created_at_ns": time_ns(),
        "target_model": str(Path(args.model).resolve()),
        "target_fingerprint": model_fingerprint(args.model),
        "requests_jsonl": str(requests_path),
        "requests_sha256": sha256_file(requests_path),
        "request_indices": list(indices),
        "max_context_tokens": args.max_context_tokens,
        "continuation_tokens": args.continuation_tokens,
        "teacher_topk": args.teacher_topk,
        "priority_page_size": args.priority_page_size,
        "kv_layers": list(kv_layers),
        "seed_layers": list(seed_layers),
        "dtype": args.dtype,
        "head_dim": int(
            getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // model.config.num_attention_heads,
            )
        ),
        "kv_heads": int(model.config.num_key_value_heads),
        "target_hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "samples": sample_rows,
        "extractor_sha256": sha256_file(Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "teacher_manifest", "samples": len(sample_rows)}))


if __name__ == "__main__":
    main()
