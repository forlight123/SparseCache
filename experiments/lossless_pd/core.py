"""Auditable sparse-view and exact-attention primitives for mechanism pilots.

An attention bound here assumes a FIXED, EXACT query. It is not a certificate
for a whole transformer or a query obtained from approximate previous layers.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from experiments.blockdraft.model import BlockKVDraft, build_packed_attention_mask
from experiments.kvshot.model import KVShotDraft


def page_order(tokens: int, page_size: int, mode: str, seed: int,
               scores: torch.Tensor | None = None) -> list[int]:
    if tokens <= 0 or page_size <= 0:
        raise ValueError("token count and page size must be positive")
    count = math.ceil(tokens / page_size)
    protected = list(dict.fromkeys([0, count - 1]))
    rest = [p for p in range(count) if p not in protected]
    if mode == "random":
        random.Random(seed).shuffle(rest)
    elif mode in {"priority", "reverse_priority"}:
        if scores is None or scores.numel() != count:
            raise ValueError("priority scheduling needs one P-side score per page")
        values = scores.detach().cpu().tolist()
        rest.sort(key=lambda p: (values[p], -p), reverse=mode == "priority")
    elif mode != "sequential":
        raise ValueError(f"unknown order {mode}")
    return protected + rest


def visible_positions(tokens: int, page_size: int, fraction: float,
                      order: list[int], device: torch.device) -> torch.Tensor:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    count = math.ceil(tokens / page_size)
    if sorted(order) != list(range(count)):
        raise ValueError("page order must be a permutation")
    n = min(count, max(min(count, 2), math.ceil(fraction * count)))
    selected = [t for p in order[:n]
                for t in range(p * page_size, min(tokens, (p + 1) * page_size))]
    return torch.tensor(sorted(selected), device=device, dtype=torch.long)


class ProgressiveBlock(nn.Module):
    """Physically compact KV inputs with original positions and a stage signal.

Every output position is an independent categorical proposal conditioned on
the known seed and this view. MASK positions never contain future labels.
There is no sampling/refinement/commit logic in this model component.
"""

    def __init__(self, base: BlockKVDraft):
        super().__init__()
        if base.config.use_seed_hidden:
            raise ValueError("pilot forbids hidden-state inputs from the Target")
        self.base = base
        self.stage = nn.Linear(2, base.config.hidden_size, bias=False)
        nn.init.zeros_(self.stage.weight)

    def sparse_hidden(self, seed, embedding, keys, values, positions,
                      prompt_length: int):
        if keys.shape != values.shape or keys.ndim != 5:
            raise ValueError("KV must have matching [B,L,H,T,D] shapes")
        if positions.ndim != 1 or positions.numel() != keys.shape[-2]:
            raise ValueError("one original position is required per physical KV token")
        if positions.numel() == 0 or bool((positions < 0).any()) or bool(
            (positions >= prompt_length).any()
        ):
            raise ValueError("missing/future/current seed KV must not be supplied")
        if positions.unique().numel() != positions.numel():
            raise ValueError("duplicate KV positions")
        batch = keys.shape[0]
        config = self.base.config
        if config.target_kv_fusion == "per_layer":
            memory_k, memory_v = keys, values
        else:
            memory_k, memory_v = self.base.project_target_memories(
                keys, values, position_ids=positions[None].expand(batch, -1)
            )
        current = embedding(seed)
        masks = self.base.mask_embedding.view(1, 1, -1).expand(
            batch, config.block_size - 1, -1
        )
        hidden = torch.cat((current, masks), dim=1)
        missing = 1 - positions.numel() / prompt_length
        stage_input = hidden.new_tensor([
            missing, missing * math.log1p(prompt_length) / math.log1p(131072)
        ])
        hidden = hidden + self.stage(stage_input).view(1, 1, -1)
        query_positions = torch.arange(
            prompt_length, prompt_length + config.block_size, device=keys.device
        )[None].expand(batch, -1)
        compact_lengths = torch.full((batch, 1), positions.numel(),
                                     device=keys.device, dtype=torch.long)
        mask = build_packed_attention_mask(
            compact_lengths, context_length=positions.numel(),
            block_size=config.block_size, causal_block=config.causal_block
        )
        for layer, key, value in zip(self.base.layers,
                                     memory_k.unbind(1), memory_v.unbind(1)):
            hidden = layer(hidden, key, value, query_positions, compact_lengths, mask)
        hidden = self.base.final_norm(hidden)
        return hidden[:, :-1] if config.shift_labels else hidden[:, 1:]

    def forward(self, seed, embedding, lm_head, keys, values, positions,
                prompt_length: int, previous_tokens: torch.Tensor | None = None):
        """Teacher-forced logits; causal correction sees only preceding tokens."""

        base_logits, correction = self.forward_components(
            seed, embedding, lm_head, keys, values, positions, prompt_length,
            previous_tokens=previous_tokens,
        )
        return base_logits + correction

    def forward_components(self, seed, embedding, lm_head, keys, values, positions,
                           prompt_length: int,
                           previous_tokens: torch.Tensor | None = None):
        """Expose frozen base and causal residual logits for constrained training."""

        selected = self.sparse_hidden(
            seed, embedding, keys, values, positions, prompt_length
        )
        if self.base.correction_gru is None:
            base_logits = lm_head(selected)
            return base_logits, torch.zeros_like(base_logits)
        if (
            previous_tokens is None
            or previous_tokens.ndim != 2
            or previous_tokens.shape[0] != selected.shape[0]
            or not 0 < previous_tokens.shape[1] <= selected.shape[1]
        ):
            raise ValueError(
                "causal correction needs a nonempty [seed, y0, ...] prefix"
            )
        selected = selected[:, :previous_tokens.shape[1]]
        base_logits = lm_head(selected)
        if self.base.config.correction_mode != "vocab":
            raise ValueError("rerank correction uses teacher_forced_rerank")
        recurrent, _ = self.base.correction_gru(embedding(previous_tokens))
        correction = self.base.correction_head(torch.cat((selected, recurrent), dim=-1))
        return base_logits, correction

    def teacher_forced_rerank(self, seed, embedding, lm_head, keys, values,
                              positions, prompt_length: int,
                              previous_tokens: torch.Tensor):
        """Return fixed base candidates and causal residual scores."""

        if self.base.correction_gru is None or self.base.config.correction_mode != "rerank":
            raise ValueError("checkpoint does not contain a rerank correction head")
        selected = self.sparse_hidden(
            seed, embedding, keys, values, positions, prompt_length
        )
        if (
            previous_tokens.ndim != 2
            or previous_tokens.shape[0] != selected.shape[0]
            or not 0 < previous_tokens.shape[1] <= selected.shape[1]
        ):
            raise ValueError("rerank needs a nonempty [seed, y0, ...] prefix")
        selected = selected[:, :previous_tokens.shape[1]]
        base_logits = lm_head(selected)
        base_scores, candidate_ids = base_logits.topk(
            self.base.config.correction_topk, dim=-1
        )
        recurrent, _ = self.base.correction_gru(embedding(previous_tokens))
        delta_hidden = self.base.correction_head(
            torch.cat((selected, recurrent), dim=-1)
        )
        candidate_weights = F.embedding(candidate_ids, lm_head.weight)
        correction_scores = torch.einsum(
            "bqh,bqkh->bqk", delta_hidden, candidate_weights
        )
        return candidate_ids, base_scores, correction_scores

    @torch.no_grad()
    def propose(self, seed, embedding, lm_head, keys, values, positions,
                prompt_length: int, length: int | None = None):
        """Autoregressive causal head over one parallel sparse-KV block pass."""

        selected = self.sparse_hidden(
            seed, embedding, keys, values, positions, prompt_length
        )
        length = selected.shape[1] if length is None else length
        if not 0 < length <= selected.shape[1]:
            raise ValueError("proposal length exceeds the configured block")
        base_logits = lm_head(selected[:, :length])
        if self.base.correction_gru is None:
            return base_logits.argmax(-1)
        proposals = []
        recurrent, state = self.base.correction_gru(embedding(seed))
        if self.base.config.correction_mode == "rerank":
            base_scores, candidate_ids = base_logits.topk(
                self.base.config.correction_topk, dim=-1
            )
        for position in range(length):
            if position:
                recurrent, state = self.base.correction_gru(
                    embedding(proposals[-1]), state
                )
            correction = self.base.correction_head(torch.cat((
                selected[:, position:position + 1], recurrent
            ), dim=-1))
            if self.base.config.correction_mode == "rerank":
                weights = F.embedding(candidate_ids[:, position:position + 1],
                                      lm_head.weight)
                residual = torch.einsum("bqh,bqkh->bqk", correction, weights)
                choice = (base_scores[:, position:position + 1] + residual).argmax(-1)
                proposals.append(
                    candidate_ids[:, position:position + 1]
                    .gather(-1, choice.unsqueeze(-1)).squeeze(-1)
                )
            else:
                proposals.append(
                    (base_logits[:, position:position + 1] + correction).argmax(-1)
                )
        return torch.cat(proposals, dim=1)


def kvshot_pd_propose(model: KVShotDraft, seed, embedding, keys, values,
                     positions, prompt_length: int, length: int):
    """Adapt the imported AR reference to the same P-side seed protocol.

Prompt KV excludes the seed. Every seed/draft representation is generated by
the drafter, never borrowed from a target forward that D could not yet run.
"""
    prefixes = [layer.project_prefix(keys, values, positions[None])
                for layer in model.layers]
    memories = [None] * len(model.layers)
    token, proposals = seed, []
    for step in range(length):
        logits, memories = model.step(
            token, embedding, prefixes, memories, prompt_length + step,
            prefix_has_current_token=False,
        )
        token = model.draft_to_target[logits[:, -1].argmax(-1), None]
        proposals.append(token)
    return torch.cat(proposals, dim=1)


def accepted_prefix(proposal: list[int], reference: list[int], stop_ids=()) -> int:
    n = 0
    for proposed, expected in zip(proposal, reference):
        if proposed != expected:
            break
        n += 1
        if expected in stop_ids:
            break
    return n


def expand_gqa(x: torch.Tensor, heads: int) -> torch.Tensor:
    if heads % x.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    return x.repeat_interleave(heads // x.shape[1], dim=1)


def attention_state(query, key, value):
    """Stable sufficient statistics for one nonempty KV tile, in FP32."""
    key, value = expand_gqa(key, query.shape[1]), expand_gqa(value, query.shape[1])
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    scores = scores / math.sqrt(query.shape[-1])
    maximum = scores.amax(-1, keepdim=True)
    weights = (scores - maximum).exp()
    return maximum, weights.sum(-1, keepdim=True), torch.matmul(weights, value.float())


def merge_attention(left, right):
    if left is None:
        return right
    m1, z1, n1 = left
    m2, z2, n2 = right
    maximum = torch.maximum(m1, m2)
    a, b = (m1 - maximum).exp(), (m2 - maximum).exp()
    return maximum, a * z1 + b * z2, a * n1 + b * n2


def attention_value(state):
    return state[2] / state[1]


def fixed_query_bound(query, key, value, positions, page_size: int):
    """Conservative center/radius page bound for a fixed exact query.

This is a mathematical FP64 diagnostic, not outward-rounded certification.
It includes all KV metadata at P. No omission is hidden from the bound.
"""
    q = query.double()
    k, v = expand_gqa(key.double(), q.shape[1]), expand_gqa(value.double(), q.shape[1])
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    visible = torch.zeros(k.shape[-2], dtype=torch.bool, device=k.device)
    visible[positions] = True
    log_z_visible = torch.logsumexp(scores[..., visible], dim=-1)
    full = scores.softmax(-1) @ v
    sparse = scores[..., visible].softmax(-1) @ v[..., visible, :]
    bounds = []
    for start in range(0, k.shape[-2], page_size):
        end = min(start + page_size, k.shape[-2])
        missing = ~visible[start:end]
        if not bool(missing.any()):
            continue
        page = k[..., start:end, :]
        center = page.mean(-2, keepdim=True)
        radius = (page - center).norm(dim=-1).amax(-1, keepdim=True)
        upper = (q * center).sum(-1) + q.norm(dim=-1) * radius
        upper = upper / math.sqrt(q.shape[-1]) + math.log(int(missing.sum()))
        bounds.append(upper)
    if bounds:
        log_z_upper = torch.logsumexp(torch.stack(bounds), dim=0)
        rho_upper = torch.sigmoid(log_z_upper - log_z_visible)
        rho_actual = scores.softmax(-1)[..., ~visible].sum(-1)
    else:
        rho_upper = torch.zeros_like(log_z_visible)
        rho_actual = torch.zeros_like(log_z_visible)
    vmax = v.norm(dim=-1).amax(-1, keepdim=True)
    bound = 2 * vmax * rho_upper
    error = (full - sparse).norm(dim=-1)
    return {
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
        "mean_bound": float(bound.mean()),
        "mean_missing_mass": float(rho_actual.mean()),
        "mean_missing_mass_upper": float(rho_upper.mean()),
        "bound_violations_fp64_tolerance_1e_9": int((error > bound + 1e-9).sum()),
        "mass_violations_fp64_tolerance_1e_9": int((rho_actual > rho_upper + 1e-9).sum()),
        "mean_bound_over_output_norm": float((bound / full.norm(dim=-1).clamp_min(1e-12)).mean()),
    }
