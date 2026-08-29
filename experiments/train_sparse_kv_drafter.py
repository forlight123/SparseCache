# SPDX-License-Identifier: Apache-2.0
"""Train a lightweight drafter against sparse, exact target-model KV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter, time_ns
from typing import Any

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from sparsecache.sparse_kv_draft import (
    SparseKVDraftConfig,
    SparseKVDrafter,
    accepted_prefix_length,
    select_visible_token_indices,
)
from sparsecache.store import model_fingerprint


def parse_int_list(raw: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if (
        not values
        or any(value < 0 for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(
            f"{name} must contain unique, non-negative integers"
        )
    return values


def parse_fraction_list(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "visibility fractions must be comma-separated numbers"
        ) from error
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("visibility fractions must be in (0, 1]")
    return values


def weighted_distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    *,
    prefix_decay: float,
    hard_weight: float,
    soft_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine greedy hard labels with a renormalized top-k teacher."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1, tokens, vocabulary]")
    tokens = logits.shape[1]
    if labels.shape != (tokens,):
        raise ValueError("labels do not match logits")
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError("teacher top-k IDs and log-probabilities differ")
    if teacher_topk_ids.shape[0] != tokens:
        raise ValueError("teacher top-k length does not match logits")
    if not 0.0 < prefix_decay <= 1.0:
        raise ValueError("prefix_decay must be in (0, 1]")
    if min(hard_weight, soft_weight) < 0.0 or hard_weight + soft_weight <= 0.0:
        raise ValueError("loss weights must be non-negative and non-zero")

    logprobs = logits.float().log_softmax(dim=-1)[0]
    hard = -logprobs.gather(1, labels[:, None]).squeeze(1)
    teacher_probs = teacher_topk_logprobs.float().softmax(dim=-1)
    student_topk = logprobs.gather(1, teacher_topk_ids)
    soft = -(teacher_probs * student_topk).sum(dim=-1)
    weights = prefix_decay ** torch.arange(
        tokens, dtype=torch.float32, device=logits.device
    )
    weights = weights / weights.sum()
    hard_loss = (weights * hard).sum()
    soft_loss = (weights * soft).sum()
    total = hard_weight * hard_loss + soft_weight * soft_loss
    return total, {
        "hard_loss": float(hard_loss.detach().item()),
        "soft_loss": float(soft_loss.detach().item()),
    }


def prefix_weights(
    tokens: int,
    *,
    prefix_decay: float,
    device: torch.device,
) -> torch.Tensor:
    if tokens <= 0 or not 0.0 < prefix_decay <= 1.0:
        raise ValueError("invalid prefix-weight parameters")
    weights = prefix_decay ** torch.arange(
        tokens,
        dtype=torch.float32,
        device=device,
    )
    return weights / weights.sum()


def feature_distillation_loss(
    features: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    prefix_decay: float,
) -> torch.Tensor:
    """Match the full verifier's normalized LM-head input features."""
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError("student features must have shape [1, tokens, hidden]")
    if teacher_hidden.shape != features.shape[1:]:
        raise ValueError("teacher hidden states do not match student features")
    student = F.normalize(features.float()[0], dim=-1)
    teacher = F.normalize(teacher_hidden.float(), dim=-1)
    per_token = 1.0 - (student * teacher).sum(dim=-1)
    weights = prefix_weights(
        features.shape[1],
        prefix_decay=prefix_decay,
        device=features.device,
    )
    return (weights * per_token).sum()


def target_sequence_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    prefix_decay: float,
) -> torch.Tensor:
    """Return prefix-weighted target log probability for contrastive training."""
    logprobs = logits.float().log_softmax(dim=-1)[0]
    target = logprobs.gather(1, labels[:, None]).squeeze(1)
    weights = prefix_weights(
        labels.numel(),
        prefix_decay=prefix_decay,
        device=logits.device,
    )
    return (weights * target).sum()


@dataclass
class TeacherSample:
    request_index: int
    prompt_tokens: int
    tensors: dict[str, torch.Tensor]
    source_group: str | None = None

    @property
    def horizon(self) -> int:
        return int(self.tensors["labels"].numel())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher_dataset(
    manifest_path: Path,
    requested_indices: set[int] | None = None,
) -> tuple[dict[str, Any], dict[int, TeacherSample]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teacher_format = manifest.get("format")
    if teacher_format not in {
        "sparsecache.sparse-kv-draft-teacher.v1",
        "sparsecache.sparse-kv-draft-teacher.v2",
        "sparsecache.sparse-kv-draft-teacher.v3",
        "sparsecache.sparse-kv-draft-teacher.v4",
        "sparsecache.sparse-kv-draft-teacher.v5",
    }:
        raise ValueError("unsupported teacher-data format")
    root = manifest_path.parent
    base_rows = {}
    base_root = root
    if teacher_format.endswith((".v4", ".v5")):
        base_manifest_path = Path(manifest["base_manifest"])
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        if base_manifest.get("format") != "sparsecache.sparse-kv-draft-teacher.v3":
            raise ValueError("v4 teacher data requires a v3 static-KV manifest")
        if base_manifest.get("target_fingerprint") != manifest.get(
            "target_fingerprint"
        ):
            raise ValueError("v4 trace and static KV target fingerprints differ")
        base_root = base_manifest_path.parent
        base_rows = {
            int(row["request_index"]): row for row in base_manifest["samples"]
        }
    samples = {}
    for row in manifest.get("samples", []):
        request_index = int(row["request_index"])
        if (
            requested_indices is not None
            and request_index not in requested_indices
        ):
            continue
        if request_index in samples:
            raise ValueError(f"duplicate teacher request {request_index}")
        sample_path = root / row["file"]
        digest = sha256_file(sample_path)
        if digest != row.get("file_sha256"):
            raise ValueError(f"teacher request {request_index} hash differs")
        tensors = dict(load_file(str(sample_path), device="cpu"))
        if teacher_format.endswith((".v4", ".v5")):
            base_row = base_rows.get(request_index)
            if base_row is None:
                raise ValueError(f"teacher request {request_index} has no static KV")
            base_path = base_root / base_row["file"]
            if sha256_file(base_path) != base_row.get("file_sha256"):
                raise ValueError(f"static KV request {request_index} hash differs")
            base_tensors = load_file(str(base_path), device="cpu")
            static_fields = {
                "memory_keys",
                "memory_values",
                "seed_hidden",
                "priority_page_scores",
                "prompt_ids",
            }
            tensors.update(
                {name: base_tensors[name] for name in static_fields}
            )
        required = {
            "memory_keys",
            "memory_values",
            "seed_hidden",
            "input_ids",
            "labels",
            "teacher_topk_ids",
            "teacher_topk_logprobs",
            "query_cos",
            "query_sin",
            "prompt_ids",
        }
        if teacher_format.endswith(".v2"):
            required.add("teacher_hidden")
        if teacher_format.endswith((".v3", ".v4")):
            required.update({"teacher_hidden", "priority_page_scores"})
        if teacher_format.endswith(".v5"):
            required.update(
                {
                    "teacher_hidden",
                    "priority_page_scores",
                    "continuation_seed_hidden",
                }
            )
        if set(tensors) != required:
            raise ValueError(f"teacher request {request_index} has invalid fields")
        if teacher_format.endswith(".v5"):
            initial_fields = {
                "input_ids",
                "labels",
                "teacher_topk_ids",
                "teacher_topk_logprobs",
                "teacher_hidden",
                "query_cos",
                "query_sin",
            }
            tensors.update(
                {
                    f"initial_{name}": base_tensors[name]
                    for name in initial_fields
                }
            )
        prompt_tokens = int(tensors["prompt_ids"].numel())
        if prompt_tokens != int(row["prompt_tokens"]):
            raise ValueError(f"teacher request {request_index} length differs")
        samples[request_index] = TeacherSample(
            request_index=request_index,
            prompt_tokens=prompt_tokens,
            tensors=tensors,
            source_group=row.get("source_dataset"),
        )
    if not samples:
        raise ValueError("teacher manifest contains no samples")
    return manifest, samples


def select_memory(
    sample: TeacherSample,
    *,
    page_size: int,
    requested_fraction: float,
    seed: int,
    device: torch.device,
    selection_mode: str = "random",
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    indices = select_visible_token_indices(
        sample.prompt_tokens,
        page_size=page_size,
        visible_fraction=requested_fraction,
        seed=seed,
        protected_prefix_pages=1,
        protected_suffix_pages=1,
        mode=selection_mode,
        page_scores=sample.tensors.get("priority_page_scores"),
    )
    keys = sample.tensors["memory_keys"].index_select(2, indices).to(device)
    values = sample.tensors["memory_values"].index_select(2, indices).to(device)
    visible_tokens = int(indices.numel())
    return keys, values, visible_tokens / sample.prompt_tokens, visible_tokens


def _sample_inputs(
    sample: TeacherSample,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    tensors = sample.tensors
    prefix = "initial_" if "initial_input_ids" in tensors else ""
    return (
        tensors[f"{prefix}input_ids"].to(device).unsqueeze(0),
        tensors["seed_hidden"].to(device).unsqueeze(0),
        tensors[f"{prefix}query_cos"].to(device),
        tensors[f"{prefix}query_sin"].to(device),
        tensors[f"{prefix}labels"].to(device),
        tensors[f"{prefix}teacher_topk_ids"].to(device),
        tensors[f"{prefix}teacher_topk_logprobs"].to(device),
    )


def training_window_inputs(
    sample: TeacherSample,
    device: torch.device,
    *,
    offset: int,
    window_tokens: int,
) -> tuple[torch.Tensor, ...]:
    if window_tokens <= 0 or not 0 <= offset < sample.horizon:
        raise ValueError("invalid teacher-window bounds")
    if offset == 0:
        inputs = _sample_inputs(sample, device)
        stop = min(inputs[0].shape[1], window_tokens)
        return (
            inputs[0][:, :stop],
            inputs[1],
            inputs[2][:stop],
            inputs[3][:stop],
            inputs[4][:stop],
            inputs[5][:stop],
            inputs[6][:stop],
        )
    else:
        tensors = sample.tensors
        continuation = sample.tensors.get("continuation_seed_hidden")
        if continuation is None:
            raise ValueError("windowed training requires v5 boundary hidden states")
        expected = (
            sample.horizon,
            tensors["seed_hidden"].shape[0],
            tensors["seed_hidden"].shape[1],
        )
        if continuation.shape != expected:
            raise ValueError("continuation boundary hidden states have invalid shape")
        seed_hidden = continuation[offset - 1].to(device).unsqueeze(0)
        stop = min(sample.horizon, offset + window_tokens)
    return (
        tensors["input_ids"][offset:stop].to(device).unsqueeze(0),
        seed_hidden,
        tensors["query_cos"][offset:stop].to(device),
        tensors["query_sin"][offset:stop].to(device),
        tensors["labels"][offset:stop].to(device),
        tensors["teacher_topk_ids"][offset:stop].to(device),
        tensors["teacher_topk_logprobs"][offset:stop].to(device),
    )


@torch.inference_mode()
def evaluate(
    model: SparseKVDrafter,
    samples: list[TeacherSample],
    fractions: tuple[float, ...],
    *,
    page_size: int,
    selection_seed: int,
    device: torch.device,
    memory_mode: str = "exact",
    seed_mode: str = "exact",
    selection_mode: str = "random",
    max_proposal_tokens: int = 8,
) -> list[dict[str, Any]]:
    if memory_mode not in {"exact", "zero", "shuffled"}:
        raise ValueError(f"unsupported memory mode: {memory_mode}")
    if seed_mode not in {"exact", "zero"}:
        raise ValueError(f"unsupported seed mode: {seed_mode}")
    if max_proposal_tokens <= 0:
        raise ValueError("maximum proposal length must be positive")
    model.eval()
    results = []
    for fraction in fractions:
        accepted = []
        evaluated_horizons = []
        teacher_forced_correct = 0
        teacher_forced_tokens = 0
        visible_counts = []
        for sample_index, sample in enumerate(samples):
            memory_sample = sample
            if memory_mode == "shuffled":
                memory_sample = samples[(sample_index + 1) % len(samples)]
            memory_keys, memory_values, actual_fraction, visible_tokens = (
                select_memory(
                    memory_sample,
                    page_size=page_size,
                    requested_fraction=fraction,
                    seed=selection_seed
                    + memory_sample.request_index * 1009,
                    device=device,
                    selection_mode=selection_mode,
                )
            )
            if memory_mode == "zero":
                memory_keys = torch.zeros_like(memory_keys)
                memory_values = torch.zeros_like(memory_values)
            (
                input_ids,
                seed_hidden,
                query_cos,
                query_sin,
                labels,
                _,
                _,
            ) = _sample_inputs(sample, device)
            evaluation_tokens = min(labels.numel(), max_proposal_tokens)
            evaluated_horizons.append(evaluation_tokens)
            input_ids = input_ids[:, :evaluation_tokens]
            query_cos = query_cos[:evaluation_tokens]
            query_sin = query_sin[:evaluation_tokens]
            labels = labels[:evaluation_tokens]
            if seed_mode == "zero":
                seed_hidden = torch.zeros_like(seed_hidden)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids,
                    seed_hidden,
                    memory_keys,
                    memory_values,
                    query_cos,
                    query_sin,
                    visible_fraction=actual_fraction,
                    prompt_tokens=sample.prompt_tokens,
                )
            predicted = logits[0].argmax(dim=-1)
            teacher_forced_correct += int((predicted == labels).sum().item())
            teacher_forced_tokens += evaluation_tokens

            proposal = []
            prefix = input_ids[:, :1]
            for position in range(evaluation_tokens):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    step_logits = model(
                        prefix,
                        seed_hidden,
                        memory_keys,
                        memory_values,
                        query_cos[: prefix.shape[1]],
                        query_sin[: prefix.shape[1]],
                        visible_fraction=actual_fraction,
                        prompt_tokens=sample.prompt_tokens,
                    )
                token = int(step_logits[0, -1].argmax().item())
                proposal.append(token)
                prefix = torch.cat(
                    [
                        prefix,
                        torch.tensor([[token]], dtype=torch.long, device=device),
                    ],
                    dim=1,
                )
            accepted.append(
                accepted_prefix_length(proposal, labels.tolist())
            )
            visible_counts.append(visible_tokens)
        results.append(
            {
                "memory_mode": memory_mode,
                "seed_mode": seed_mode,
                "selection_mode": selection_mode,
                "requested_fraction": fraction,
                "samples": len(samples),
                "mean_visible_tokens": sum(visible_counts) / len(visible_counts),
                "teacher_forced_top1": (
                    teacher_forced_correct / teacher_forced_tokens
                ),
                "mean_accepted_prefix": sum(accepted) / len(accepted),
                "full_accept_rate": sum(
                    value == horizon
                    for value, horizon in zip(
                        accepted, evaluated_horizons, strict=True
                    )
                )
                / len(samples),
                "accepted_prefixes": accepted,
            }
        )
    return results


def load_target_shared_weights(
    model_path: str,
    *,
    device: torch.device,
    attn_implementation: str,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    from transformers import AutoModelForCausalLM

    target = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        local_files_only=True,
    ).to(device).eval()
    embedding = target.get_input_embeddings().weight.detach()
    lm_head = target.get_output_embeddings().weight.detach()
    embedding.requires_grad_(False)
    lm_head.requires_grad_(False)
    config = target.config
    del target
    torch.cuda.empty_cache()
    return embedding, lm_head, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-indices", default="0,1,5")
    parser.add_argument("--eval-indices", default="8")
    parser.add_argument("--eval-modulus", type=int)
    parser.add_argument("--eval-remainder", type=int, default=0)
    parser.add_argument("--visibility-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--selection-mode",
        choices=("random", "priority"),
        default="random",
    )
    parser.add_argument("--draft-hidden-size", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--prefix-decay", type=float, default=0.9)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--soft-weight", type=float, default=0.5)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--seed-dropout-prob", type=float, default=0.25)
    parser.add_argument("--memory-contrast-weight", type=float, default=0.5)
    parser.add_argument("--memory-contrast-margin", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-eval-tokens", type=int, default=8)
    parser.add_argument("--training-window-tokens", type=int, default=0)
    parser.add_argument("--initial-window-prob", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fractions = parse_fraction_list(args.visibility_fractions)
    if min(args.page_size, args.steps, args.log_every) <= 0:
        raise ValueError("page size, steps, and logging interval must be positive")
    if not 0.0 <= args.seed_dropout_prob < 1.0:
        raise ValueError("seed-dropout probability must be in [0, 1)")
    if args.training_window_tokens < 0:
        raise ValueError("training-window length cannot be negative")
    if not 0.0 <= args.initial_window_prob <= 1.0:
        raise ValueError("initial-window probability must be in [0, 1]")
    if min(args.feature_weight, args.memory_contrast_weight) < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the reference training run requires a CUDA device")
    torch.cuda.set_device(device)

    manifest_path = Path(args.teacher_manifest)
    manifest, sample_map = load_teacher_dataset(manifest_path)
    if args.eval_modulus is None:
        train_indices = parse_int_list(args.train_indices, name="train indices")
        eval_indices = parse_int_list(args.eval_indices, name="eval indices")
    else:
        if (
            args.eval_modulus <= 1
            or not 0 <= args.eval_remainder < args.eval_modulus
        ):
            raise ValueError("invalid evaluation modulus split")
        eval_indices = tuple(
            index
            for index in sorted(sample_map)
            if index % args.eval_modulus == args.eval_remainder
        )
        eval_set = set(eval_indices)
        train_indices = tuple(
            index for index in sorted(sample_map) if index not in eval_set
        )
    if set(train_indices) & set(eval_indices):
        raise ValueError("train and evaluation request indices overlap")
    if args.selection_mode == "priority" and (
        int(manifest.get("priority_page_size", -1)) != args.page_size
    ):
        raise ValueError("priority page size differs from training page size")
    missing = (set(train_indices) | set(eval_indices)) - set(sample_map)
    if missing:
        raise ValueError(f"teacher samples are missing indices {sorted(missing)}")
    train_samples = [sample_map[index] for index in train_indices]
    eval_samples = [sample_map[index] for index in eval_indices]
    target_model = str(manifest["target_model"])
    if model_fingerprint(target_model) != manifest["target_fingerprint"]:
        raise ValueError("target model fingerprint differs from teacher data")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    embedding, lm_head, target_config = load_target_shared_weights(
        target_model,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    config = SparseKVDraftConfig(
        target_hidden_size=int(manifest["target_hidden_size"]),
        draft_hidden_size=args.draft_hidden_size,
        head_dim=int(manifest["head_dim"]),
        num_draft_heads=(
            args.draft_hidden_size // int(manifest["head_dim"])
        ),
        num_memory_heads=int(manifest["kv_heads"]),
        num_memory_layers=len(manifest["kv_layers"]),
        num_seed_layers=len(manifest["seed_layers"]),
        num_blocks=args.num_blocks,
        mlp_ratio=args.mlp_ratio,
        rms_norm_eps=float(getattr(target_config, "rms_norm_eps", 1e-5)),
    )
    model = SparseKVDrafter(config).to(device)
    model.bind_target_weights(embedding, lm_head)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    initial_eval = evaluate(
        model,
        eval_samples,
        fractions,
        page_size=args.page_size,
        selection_seed=args.seed + 700_000,
        device=device,
        selection_mode=args.selection_mode,
        max_proposal_tokens=args.max_eval_tokens,
    )
    print(json.dumps({"event": "initial_eval", "metrics": initial_eval}), flush=True)

    history = []
    started = perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        sample_index = (step - 1) % len(train_samples)
        epoch_index = (step - 1) // len(train_samples)
        sample = train_samples[sample_index]
        fraction = fractions[(sample_index + epoch_index) % len(fractions)]
        memory_keys, memory_values, actual_fraction, visible_tokens = select_memory(
            sample,
            page_size=args.page_size,
            requested_fraction=fraction,
            seed=args.seed + step * 7919 + sample.request_index,
            device=device,
            selection_mode=args.selection_mode,
        )
        window_offset = 0
        if (
            args.training_window_tokens > 0
            and sample.horizon > 1
            and random.random() >= args.initial_window_prob
        ):
            window_offset = random.randrange(1, sample.horizon)
        if args.training_window_tokens > 0:
            (
                input_ids,
                seed_hidden,
                query_cos,
                query_sin,
                labels,
                teacher_topk_ids,
                teacher_topk_logprobs,
            ) = training_window_inputs(
                sample,
                device,
                offset=window_offset,
                window_tokens=args.training_window_tokens,
            )
        else:
            (
                input_ids,
                seed_hidden,
                query_cos,
                query_sin,
                labels,
                teacher_topk_ids,
                teacher_topk_logprobs,
            ) = _sample_inputs(sample, device)
        teacher_hidden = sample.tensors.get("teacher_hidden")
        if args.feature_weight > 0.0 and teacher_hidden is None:
            raise ValueError("feature loss requires v2 teacher hidden states")
        if teacher_hidden is not None:
            if window_offset == 0 and "initial_teacher_hidden" in sample.tensors:
                teacher_hidden = sample.tensors["initial_teacher_hidden"]
            teacher_hidden = teacher_hidden[
                window_offset : window_offset + input_ids.shape[1]
            ].to(device)
        seed_dropped = random.random() < args.seed_dropout_prob
        training_seed_hidden = (
            torch.zeros_like(seed_hidden) if seed_dropped else seed_hidden
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, features = model(
                input_ids,
                training_seed_hidden,
                memory_keys,
                memory_values,
                query_cos,
                query_sin,
                visible_fraction=actual_fraction,
                prompt_tokens=sample.prompt_tokens,
                return_features=True,
            )
            loss, components = weighted_distillation_loss(
                logits,
                labels,
                teacher_topk_ids,
                teacher_topk_logprobs,
                prefix_decay=args.prefix_decay,
                hard_weight=args.hard_weight,
                soft_weight=args.soft_weight,
            )
            feature_loss = torch.zeros((), device=device)
            if args.feature_weight > 0.0:
                feature_loss = feature_distillation_loss(
                    features,
                    teacher_hidden,
                    prefix_decay=args.prefix_decay,
                )
                loss = loss + args.feature_weight * feature_loss

            contrast_loss = torch.zeros((), device=device)
            if args.memory_contrast_weight > 0.0:
                negative_sample = train_samples[step % len(train_samples)]
                negative_keys, negative_values, _, _ = select_memory(
                    negative_sample,
                    page_size=args.page_size,
                    requested_fraction=fraction,
                    seed=(
                        args.seed
                        + step * 7919
                        + negative_sample.request_index
                    ),
                    device=device,
                    selection_mode=args.selection_mode,
                )
                negative_logits = model(
                    input_ids,
                    training_seed_hidden,
                    negative_keys,
                    negative_values,
                    query_cos,
                    query_sin,
                    visible_fraction=actual_fraction,
                    prompt_tokens=sample.prompt_tokens,
                )
                positive_score = target_sequence_score(
                    logits,
                    labels,
                    prefix_decay=args.prefix_decay,
                )
                negative_score = target_sequence_score(
                    negative_logits,
                    labels,
                    prefix_decay=args.prefix_decay,
                )
                contrast_loss = F.relu(
                    args.memory_contrast_margin
                    - positive_score
                    + negative_score
                )
                loss = loss + args.memory_contrast_weight * contrast_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        optimizer.step()
        row = {
            "step": step,
            "request_index": sample.request_index,
            "requested_fraction": fraction,
            "actual_fraction": actual_fraction,
            "visible_tokens": visible_tokens,
            "window_offset": window_offset,
            "window_tokens": int(input_ids.shape[1]),
            "loss": float(loss.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
            "feature_loss": float(feature_loss.detach().item()),
            "contrast_loss": float(contrast_loss.detach().item()),
            "seed_dropped": seed_dropped,
            **components,
        }
        history.append(row)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"event": "train", **row}), flush=True)

    train_eval = evaluate(
        model,
        train_samples,
        fractions,
        page_size=args.page_size,
        selection_seed=args.seed + 700_000,
        device=device,
        selection_mode=args.selection_mode,
        max_proposal_tokens=args.max_eval_tokens,
    )
    heldout_eval = evaluate(
        model,
        eval_samples,
        fractions,
        page_size=args.page_size,
        selection_seed=args.seed + 700_000,
        device=device,
        selection_mode=args.selection_mode,
        max_proposal_tokens=args.max_eval_tokens,
    )
    elapsed_seconds = perf_counter() - started
    checkpoint = {
        "format": "sparsecache.sparse-kv-drafter.v1",
        "created_at_ns": time_ns(),
        "config": config.to_dict(),
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "target_model": target_model,
        "target_fingerprint": manifest["target_fingerprint"],
        "teacher_manifest": str(manifest_path.resolve()),
        "kv_layers": manifest["kv_layers"],
        "seed_layers": manifest["seed_layers"],
        "train_indices": list(train_indices),
        "eval_indices": list(eval_indices),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    result = {
        "schema_version": 1,
        "format": "sparsecache.sparse-kv-drafter-training.v1",
        "args": vars(args),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "elapsed_seconds": elapsed_seconds,
        "initial_heldout": initial_eval,
        "train": train_eval,
        "heldout": heldout_eval,
        "history": history,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output_dir": str(output_dir),
                "parameter_count": result["parameter_count"],
                "elapsed_seconds": elapsed_seconds,
                "train": train_eval,
                "heldout": heldout_eval,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
