# SPDX-License-Identifier: Apache-2.0
"""RoPE relocation helpers for reusable cached keys."""

from __future__ import annotations

import torch


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the rotate-half convention used by Llama RoPE."""
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequence_dim: int = -2,
) -> torch.Tensor:
    """Apply a RoPE rotation to a tensor."""
    cos, sin = _broadcast_rope(tensor, cos, sin, sequence_dim)
    return tensor * cos + rotate_half(tensor) * sin


def remove_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequence_dim: int = -2,
) -> torch.Tensor:
    """Remove a previously applied orthogonal RoPE rotation."""
    cos, sin = _broadcast_rope(tensor, cos, sin, sequence_dim)
    return tensor * cos - rotate_half(tensor) * sin


def model_rope_cos_sin(
    model,
    positions: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Obtain the model's exact RoPE cosine and sine schedule."""
    backbone = model.model
    positions = positions.to(device=device, dtype=torch.long)
    position_ids = positions.unsqueeze(0) if positions.ndim == 1 else positions
    dummy = torch.empty(
        (*position_ids.shape, int(backbone.config.hidden_size)),
        device=device,
        dtype=dtype,
    )
    cos, sin = backbone.rotary_emb(dummy, position_ids)
    if positions.ndim == 1:
        cos, sin = cos[0], sin[0]
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def relocate_key(
    model,
    key: torch.Tensor,
    *,
    source_start: int,
    target_start: int,
) -> torch.Tensor:
    """Move a post-RoPE key tensor between absolute position intervals."""
    if source_start == target_start:
        return key
    tokens = key.shape[2]
    source = torch.arange(source_start, source_start + tokens)
    target = torch.arange(target_start, target_start + tokens)
    source_cos, source_sin = model_rope_cos_sin(
        model,
        source,
        device="cpu",
        dtype=torch.float32,
    )
    target_cos, target_sin = model_rope_cos_sin(
        model,
        target,
        device="cpu",
        dtype=torch.float32,
    )
    canonical = remove_rope(
        key.float(), source_cos, source_sin, sequence_dim=2
    )
    return apply_rope(
        canonical, target_cos, target_sin, sequence_dim=2
    ).to(key.dtype)


def _broadcast_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sequence_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_dim %= tensor.ndim
    if cos.shape != sin.shape:
        raise ValueError("cos and sin shapes differ")
    if cos.ndim == 1:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.ndim not in (2, 3):
        raise ValueError("RoPE tensors must have two or three dimensions")
    if cos.shape[-2] != tensor.shape[sequence_dim]:
        raise ValueError("RoPE token dimension does not match tensor")
    if cos.shape[-1] != tensor.shape[-1]:
        raise ValueError("RoPE head dimension does not match tensor")
    shape = [1] * tensor.ndim
    if cos.ndim == 3:
        if cos.shape[0] != tensor.shape[0]:
            raise ValueError("RoPE batch dimension does not match tensor")
        shape[0] = cos.shape[0]
        shape[sequence_dim] = cos.shape[1]
        shape[-1] = cos.shape[2]
    else:
        shape[sequence_dim] = cos.shape[0]
        shape[-1] = cos.shape[1]
    return cos.reshape(shape), sin.reshape(shape)
