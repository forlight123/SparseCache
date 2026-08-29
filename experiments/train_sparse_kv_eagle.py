# SPDX-License-Identifier: Apache-2.0
"""Train only the sparse-target-KV adapter on a frozen EAGLE-3 drafter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter, time_ns
from typing import Any

from safetensors import safe_open
import torch
import torch.nn.functional as F

from experiments.train_sparse_kv_drafter import (
    TeacherSample,
    _sample_inputs,
    load_teacher_dataset,
    parse_fraction_list,
    parse_int_list,
    prefix_weights,
    select_memory,
    training_window_inputs,
)
from sparsecache.sparse_kv_draft import accepted_prefix_length
from sparsecache.sparse_kv_eagle import (
    SparseKVEagleConfig,
    SparseKVEagleDrafter,
)
from sparsecache.store import model_fingerprint


def load_target_embedding(
    model_path: str, *, device: torch.device
) -> tuple[torch.Tensor, Any]:
    from transformers import AutoConfig

    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard = index["weight_map"]["model.embed_tokens.weight"]
        weight_path = root / shard
    else:
        weight_path = root / "model.safetensors"
    with safe_open(str(weight_path), framework="pt", device="cpu") as weights:
        embedding = weights.get_tensor("model.embed_tokens.weight")
    embedding = embedding.to(device=device, dtype=torch.bfloat16)
    embedding.requires_grad_(False)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    return embedding, config


def build_model(
    *,
    manifest: dict[str, Any],
    eagle_checkpoint: Path,
    target_embedding: torch.Tensor,
    target_config: Any,
    device: torch.device,
) -> SparseKVEagleDrafter:
    eagle_config_path = eagle_checkpoint.parent / "config.json"
    eagle_config = json.loads(eagle_config_path.read_text(encoding="utf-8"))
    config = SparseKVEagleConfig(
        target_hidden_size=int(manifest["target_hidden_size"]),
        hidden_size=int(eagle_config["hidden_size"]),
        intermediate_size=int(eagle_config["intermediate_size"]),
        head_dim=int(manifest["head_dim"]),
        num_attention_heads=int(eagle_config["num_attention_heads"]),
        num_key_value_heads=int(eagle_config["num_key_value_heads"]),
        num_memory_heads=int(manifest["kv_heads"]),
        num_memory_layers=len(manifest["kv_layers"]),
        num_seed_layers=len(manifest["seed_layers"]),
        target_vocab_size=int(manifest["vocab_size"]),
        draft_vocab_size=int(eagle_config["draft_vocab_size"]),
        rms_norm_eps=float(eagle_config.get("rms_norm_eps", 1e-5)),
    )
    if int(target_config.vocab_size) != config.target_vocab_size:
        raise ValueError("target and EAGLE vocabulary sizes differ")
    model = SparseKVEagleDrafter(config)
    model.load_eagle_checkpoint(eagle_checkpoint)
    model.freeze_eagle()
    model.to(device=device, dtype=torch.bfloat16)
    model.memory_norm.float()
    model.memory_attention.float()
    model.stage_layer_bias.float()
    model.adapter_scale.data = model.adapter_scale.data.float()
    model.bind_target_embedding(target_embedding)
    return model


def compressed_distillation_loss(
    model: SparseKVEagleDrafter,
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    *,
    prefix_decay: float,
    hard_weight: float,
    soft_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = logits.shape[1]
    if logits.shape[:2] != (1, labels.numel()) or labels.numel() != tokens:
        raise ValueError("compressed logits and labels have incompatible shapes")
    weights = prefix_weights(tokens, prefix_decay=prefix_decay, device=logits.device)
    logprobs = logits.float().log_softmax(dim=-1)[0]
    draft_labels = model.draft_ids(labels)
    valid_hard = draft_labels >= 0
    hard_weights = weights * valid_hard.float()
    if hard_weights.sum() == 0:
        raise ValueError("no teacher labels are covered by EAGLE's draft vocabulary")
    hard = -logprobs.gather(1, draft_labels.clamp_min(0)[:, None]).squeeze(1)
    hard_loss = (hard * hard_weights).sum() / hard_weights.sum()

    teacher_draft_ids = model.draft_ids(teacher_topk_ids)
    valid_soft = teacher_draft_ids >= 0
    valid_rows = valid_soft.any(dim=-1)
    teacher_logits = teacher_topk_logprobs.float().masked_fill(
        ~valid_soft, -torch.inf
    )
    teacher_logits = torch.where(
        valid_rows[:, None], teacher_logits, torch.zeros_like(teacher_logits)
    )
    teacher_probs = teacher_logits.softmax(dim=-1)
    selected_student = logprobs.gather(1, teacher_draft_ids.clamp_min(0))
    soft_per_token = -(teacher_probs * selected_student).sum(dim=-1)
    soft_weights = weights * valid_rows.float()
    soft_loss = (soft_per_token * soft_weights).sum() / soft_weights.sum()
    loss = hard_weight * hard_loss + soft_weight * soft_loss
    return loss, {
        "hard_loss": float(hard_loss.detach().item()),
        "soft_loss": float(soft_loss.detach().item()),
        "label_coverage": float(valid_hard.float().mean().item()),
        "topk_coverage": float(valid_soft.float().mean().item()),
    }


def compressed_target_score(
    model: SparseKVEagleDrafter,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    prefix_decay: float,
) -> torch.Tensor:
    draft_labels = model.draft_ids(labels)
    valid = draft_labels >= 0
    weights = prefix_weights(
        labels.numel(), prefix_decay=prefix_decay, device=logits.device
    )
    weights = weights * valid.float()
    logprobs = logits.float().log_softmax(dim=-1)[0]
    selected = logprobs.gather(1, draft_labels.clamp_min(0)[:, None]).squeeze(1)
    return (weights * selected).sum() / weights.sum()


def pick_mismatched_sample(
    samples: list[TeacherSample], index: int
) -> TeacherSample:
    sample = samples[index]
    for offset in range(1, len(samples)):
        candidate = samples[(index + offset) % len(samples)]
        if candidate.source_group != sample.source_group:
            return candidate
    return samples[(index + 1) % len(samples)]


def configure_trainable_modules(
    model: SparseKVEagleDrafter,
    *,
    train_fc: bool,
    train_midlayer: bool,
    adapter_scale_init: float,
) -> None:
    """Open the requested EAGLE modules while keeping FP32 master weights."""
    model.adapter_scale.data.fill_(adapter_scale_init)
    if train_fc:
        model.fc.float()
        for parameter in model.fc.parameters():
            parameter.requires_grad_(True)
    if train_midlayer:
        model.midlayer.float()
        for parameter in model.midlayer.parameters():
            parameter.requires_grad_(True)


def load_adapter_checkpoint(
    model: SparseKVEagleDrafter,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Load a SparseCache adapter checkpoint without changing trainability."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("resume checkpoint has no adapter_state_dict")
    model_state = model.state_dict()
    unexpected = set(state) - set(model_state)
    if unexpected:
        raise ValueError(f"resume checkpoint has unknown tensors: {sorted(unexpected)}")
    mismatched = {
        name: (tuple(tensor.shape), tuple(model_state[name].shape))
        for name, tensor in state.items()
        if tensor.shape != model_state[name].shape
    }
    if mismatched:
        raise ValueError(f"resume checkpoint tensor shapes differ: {mismatched}")
    model.load_state_dict(state, strict=False)
    return checkpoint


def build_adapter_checkpoint(
    *,
    model: SparseKVEagleDrafter,
    args: argparse.Namespace,
    eagle_checkpoint: Path,
    target_model: str,
    target_fingerprint: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    train_indices: tuple[int, ...],
    eval_indices: tuple[int, ...],
    completed_steps: int,
) -> dict[str, Any]:
    return {
        "format": "sparsecache.sparse-kv-eagle-adapter.v2",
        "created_at_ns": time_ns(),
        "config": model.config.to_dict(),
        "adapter_state_dict": model.adapter_state_dict(),
        "trainable_parameter_names": [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        "completed_steps": completed_steps,
        "training_args": vars(args),
        "base_eagle_checkpoint": str(eagle_checkpoint.resolve()),
        "target_model": target_model,
        "target_fingerprint": target_fingerprint,
        "teacher_manifest": str(manifest_path.resolve()),
        "kv_layers": manifest["kv_layers"],
        "seed_layers": manifest["seed_layers"],
        "train_indices": list(train_indices),
        "eval_indices": list(eval_indices),
    }


@torch.inference_mode()
def evaluate(
    model: SparseKVEagleDrafter,
    samples: list[TeacherSample],
    fractions: tuple[float, ...],
    *,
    page_size: int,
    selection_seed: int,
    device: torch.device,
    selection_mode: str,
    memory_mode: str = "exact",
    max_proposal_tokens: int = 8,
) -> list[dict[str, Any]]:
    if memory_mode not in {"exact", "zero", "shuffled", "disabled"}:
        raise ValueError("unsupported EAGLE memory ablation")
    model.eval()
    rows = []
    for fraction in fractions:
        accepted = []
        correct = 0
        covered_correct = 0
        covered_tokens = 0
        total_tokens = 0
        evaluated_horizons = []
        for sample_index, sample in enumerate(samples):
            memory_sample = sample
            if memory_mode == "shuffled":
                memory_sample = pick_mismatched_sample(samples, sample_index)
            memory_keys, memory_values, actual_fraction, _ = select_memory(
                memory_sample,
                page_size=page_size,
                requested_fraction=fraction,
                seed=selection_seed + memory_sample.request_index * 1009,
                device=device,
                selection_mode=selection_mode,
            )
            if memory_mode == "zero":
                memory_keys = torch.zeros_like(memory_keys)
                memory_values = torch.zeros_like(memory_values)
            input_ids, seed_hidden, query_cos, query_sin, labels, _, _ = (
                _sample_inputs(sample, device)
            )
            evaluation_tokens = min(labels.numel(), max_proposal_tokens)
            input_ids = input_ids[:, :evaluation_tokens]
            query_cos = query_cos[:evaluation_tokens]
            query_sin = query_sin[:evaluation_tokens]
            labels = labels[:evaluation_tokens]
            evaluated_horizons.append(evaluation_tokens)
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
                    use_memory=memory_mode != "disabled",
                )
            predicted = model.target_ids(logits[0].argmax(dim=-1))
            covered = model.draft_ids(labels) >= 0
            correct += int((predicted == labels).sum().item())
            covered_correct += int(((predicted == labels) & covered).sum().item())
            covered_tokens += int(covered.sum().item())
            total_tokens += evaluation_tokens

            proposal = []
            prefix = input_ids[:, :1]
            for _ in range(evaluation_tokens):
                prefix_tokens = prefix.shape[1]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    step_logits = model(
                        prefix,
                        seed_hidden,
                        memory_keys,
                        memory_values,
                        query_cos[:prefix_tokens],
                        query_sin[:prefix_tokens],
                        visible_fraction=actual_fraction,
                        prompt_tokens=sample.prompt_tokens,
                        use_memory=memory_mode != "disabled",
                    )
                draft_id = step_logits[0, -1].argmax()
                token = int(model.target_ids(draft_id).item())
                proposal.append(token)
                prefix = torch.cat(
                    (
                        prefix,
                        torch.tensor([[token]], device=device, dtype=torch.long),
                    ),
                    dim=1,
                )
            accepted.append(accepted_prefix_length(proposal, labels.tolist()))
        rows.append(
            {
                "memory_mode": memory_mode,
                "requested_fraction": fraction,
                "samples": len(samples),
                "teacher_forced_top1": correct / total_tokens,
                "covered_teacher_forced_top1": covered_correct / covered_tokens,
                "label_coverage": covered_tokens / total_tokens,
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
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--eagle-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-indices")
    parser.add_argument("--eval-indices")
    parser.add_argument("--eval-modulus", type=int)
    parser.add_argument("--eval-remainder", type=int, default=0)
    parser.add_argument("--visibility-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--selection-mode",
        choices=("random", "priority"),
        default="priority",
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--prefix-decay", type=float, default=0.9)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--soft-weight", type=float, default=0.5)
    parser.add_argument("--contrast-weight", type=float, default=0.5)
    parser.add_argument("--contrast-margin", type=float, default=0.25)
    parser.add_argument("--base-contrast-weight", type=float, default=0.25)
    parser.add_argument("--train-fc", action="store_true")
    parser.add_argument("--train-midlayer", action="store_true")
    parser.add_argument("--adapter-scale-init", type=float, default=0.0)
    parser.add_argument("--adapter-scale-warmup-steps", type=int, default=0)
    parser.add_argument("--resume-adapter")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--training-window-tokens", type=int, default=0)
    parser.add_argument("--initial-window-prob", type=float, default=0.25)
    parser.add_argument("--max-eval-tokens", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda:1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fractions = parse_fraction_list(args.visibility_fractions)
    if min(args.steps, args.page_size, args.log_every, args.max_eval_tokens) <= 0:
        raise ValueError("training and evaluation sizes must be positive")
    if min(
        args.training_window_tokens,
        args.adapter_scale_warmup_steps,
        args.save_every,
    ) < 0:
        raise ValueError("window, warmup, and checkpoint intervals cannot be negative")
    if not 0.0 <= args.initial_window_prob <= 1.0:
        raise ValueError("initial-window probability must be in [0, 1]")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    manifest_path = Path(args.teacher_manifest)
    manifest, sample_map = load_teacher_dataset(manifest_path)
    if args.eval_modulus is not None:
        if args.train_indices is not None or args.eval_indices is not None:
            raise ValueError("modulus split cannot be mixed with explicit indices")
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
        eval_index_set = set(eval_indices)
        train_indices = tuple(
            index for index in sorted(sample_map) if index not in eval_index_set
        )
    else:
        if args.train_indices is None or args.eval_indices is None:
            raise ValueError("explicit train and evaluation indices are required")
        train_indices = parse_int_list(args.train_indices, name="train indices")
        eval_indices = parse_int_list(args.eval_indices, name="eval indices")
    if set(train_indices) & set(eval_indices):
        raise ValueError("training and evaluation requests overlap")
    train_samples = [sample_map[index] for index in train_indices]
    eval_samples = [sample_map[index] for index in eval_indices]
    if args.selection_mode == "priority" and int(
        manifest.get("priority_page_size", -1)
    ) != args.page_size:
        raise ValueError("priority page size differs from the teacher data")
    target_model = str(manifest["target_model"])
    if model_fingerprint(target_model) != manifest["target_fingerprint"]:
        raise ValueError("target model fingerprint differs from teacher data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    embedding, target_config = load_target_embedding(target_model, device=device)
    eagle_checkpoint = Path(args.eagle_checkpoint)
    model = build_model(
        manifest=manifest,
        eagle_checkpoint=eagle_checkpoint,
        target_embedding=embedding,
        target_config=target_config,
        device=device,
    )
    configure_trainable_modules(
        model,
        train_fc=args.train_fc,
        train_midlayer=args.train_midlayer,
        adapter_scale_init=args.adapter_scale_init,
    )
    resumed_from = None
    if args.resume_adapter is not None:
        resume_path = Path(args.resume_adapter)
        resume = load_adapter_checkpoint(model, resume_path)
        if resume.get("target_fingerprint") != manifest["target_fingerprint"]:
            raise ValueError("resume checkpoint target fingerprint differs")
        resumed_from = {
            "path": str(resume_path.resolve()),
            "completed_steps": int(resume.get("completed_steps", 0)),
        }
    optimizer = torch.optim.AdamW(
        model.adapter_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    initial = evaluate(
        model,
        eval_samples,
        fractions,
        page_size=args.page_size,
        selection_seed=args.seed + 700_000,
        device=device,
        selection_mode=args.selection_mode,
        memory_mode="disabled",
        max_proposal_tokens=args.max_eval_tokens,
    )
    print(json.dumps({"event": "initial_eagle", "metrics": initial}), flush=True)

    history = []
    started = perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        sample_index = (step - 1) % len(train_samples)
        epoch_index = (step - 1) // len(train_samples)
        sample = train_samples[sample_index]
        negative_sample = pick_mismatched_sample(train_samples, sample_index)
        fraction = fractions[(sample_index + epoch_index) % len(fractions)]
        memory_keys, memory_values, actual_fraction, visible_tokens = select_memory(
            sample,
            page_size=args.page_size,
            requested_fraction=fraction,
            seed=args.seed + step * 7919 + sample.request_index,
            device=device,
            selection_mode=args.selection_mode,
        )
        negative_keys, negative_values, _, _ = select_memory(
            negative_sample,
            page_size=args.page_size,
            requested_fraction=fraction,
            seed=args.seed + step * 7919 + negative_sample.request_index,
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
        optimizer.zero_grad(set_to_none=True)
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
            loss, components = compressed_distillation_loss(
                model,
                logits,
                labels,
                teacher_topk_ids,
                teacher_topk_logprobs,
                prefix_decay=args.prefix_decay,
                hard_weight=args.hard_weight,
                soft_weight=args.soft_weight,
            )
            positive_score = compressed_target_score(
                model, logits, labels, prefix_decay=args.prefix_decay
            )
            negative_logits = model(
                input_ids,
                seed_hidden,
                negative_keys,
                negative_values,
                query_cos,
                query_sin,
                visible_fraction=actual_fraction,
                prompt_tokens=sample.prompt_tokens,
            )
            negative_score = compressed_target_score(
                model, negative_logits, labels, prefix_decay=args.prefix_decay
            )
            contrast_loss = F.relu(
                args.contrast_margin - positive_score + negative_score
            )
            with torch.no_grad():
                base_logits = model(
                    input_ids,
                    seed_hidden,
                    memory_keys,
                    memory_values,
                    query_cos,
                    query_sin,
                    visible_fraction=actual_fraction,
                    prompt_tokens=sample.prompt_tokens,
                    use_memory=False,
                )
                base_score = compressed_target_score(
                    model, base_logits, labels, prefix_decay=args.prefix_decay
                )
            base_contrast = F.relu(
                args.contrast_margin / 2 - positive_score + base_score
            )
            loss = (
                loss
                + args.contrast_weight * contrast_loss
                + args.base_contrast_weight * base_contrast
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        if step <= args.adapter_scale_warmup_steps:
            model.adapter_scale.grad = None
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.adapter_parameters(), args.gradient_clip
        )
        optimizer.step()
        row = {
            "step": step,
            "request_index": sample.request_index,
            "negative_request_index": negative_sample.request_index,
            "requested_fraction": fraction,
            "actual_fraction": actual_fraction,
            "visible_tokens": visible_tokens,
            "window_offset": window_offset,
            "window_tokens": int(input_ids.shape[1]),
            "loss": float(loss.detach().item()),
            "contrast_loss": float(contrast_loss.detach().item()),
            "base_contrast_loss": float(base_contrast.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
            "adapter_scale": float(torch.sigmoid(model.adapter_scale).detach().item()),
            **components,
        }
        history.append(row)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"event": "train", **row}), flush=True)
        if args.save_every > 0 and step % args.save_every == 0:
            snapshot = build_adapter_checkpoint(
                model=model,
                args=args,
                eagle_checkpoint=eagle_checkpoint,
                target_model=target_model,
                target_fingerprint=manifest["target_fingerprint"],
                manifest_path=manifest_path,
                manifest=manifest,
                train_indices=train_indices,
                eval_indices=eval_indices,
                completed_steps=step,
            )
            snapshot_path = output_dir / f"adapter_step_{step:08d}.pt"
            torch.save(snapshot, snapshot_path)
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "step": step,
                        "path": str(snapshot_path),
                    }
                ),
                flush=True,
            )

    evaluations = {}
    for memory_mode in ("exact", "zero", "shuffled", "disabled"):
        evaluations[memory_mode] = evaluate(
            model,
            eval_samples,
            fractions,
            page_size=args.page_size,
            selection_seed=args.seed + 700_000,
            device=device,
            selection_mode=args.selection_mode,
            memory_mode=memory_mode,
            max_proposal_tokens=args.max_eval_tokens,
        )
    elapsed = perf_counter() - started
    checkpoint = build_adapter_checkpoint(
        model=model,
        args=args,
        eagle_checkpoint=eagle_checkpoint,
        target_model=target_model,
        target_fingerprint=manifest["target_fingerprint"],
        manifest_path=manifest_path,
        manifest=manifest,
        train_indices=train_indices,
        eval_indices=eval_indices,
        completed_steps=args.steps,
    )
    torch.save(checkpoint, output_dir / "adapter.pt")
    result = {
        "schema_version": 1,
        "format": "sparsecache.sparse-kv-eagle-training.v1",
        "args": vars(args),
        "elapsed_seconds": elapsed,
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "initial_eagle": initial,
        "heldout": evaluations,
        "resumed_from": resumed_from,
        "history": history,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output_dir": str(output_dir),
                "elapsed_seconds": elapsed,
                "trainable_parameters": result["trainable_parameter_count"],
                "heldout": evaluations,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
