# SPDX-License-Identifier: Apache-2.0
"""Portable BF16 CPU KV files and runtime cache composition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .rope import relocate_key

LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def model_fingerprint(model_path: str | Path) -> str:
    """Return a stable hash of the model configuration and weight index."""
    root = Path(model_path)
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_to_legacy(cache: Any) -> LegacyCache:
    """Convert a Transformers cache object into per-layer K/V tuples."""
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache object: {type(cache)!r}")


def save_cache(path: str | Path, cache: LegacyCache) -> int:
    """Save a CPU cache with safetensors and return its logical byte size."""
    tensors = {}
    logical_bytes = 0
    for layer_index, (key, value) in enumerate(cache):
        key = key.detach().to("cpu").contiguous()
        value = value.detach().to("cpu").contiguous()
        tensors[f"layer_{layer_index:03d}.key"] = key
        tensors[f"layer_{layer_index:03d}.value"] = value
        logical_bytes += (key.numel() + value.numel()) * key.element_size()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, temporary)
    temporary.replace(path)
    return logical_bytes


def load_cache(path: str | Path, expected_layers: int | None = None) -> LegacyCache:
    """Load a safetensors K/V file into pageable CPU tensors."""
    tensors = load_file(str(path), device="cpu")
    layer_indices = sorted(
        int(name.split(".")[0].split("_")[1])
        for name in tensors
        if name.endswith(".key")
    )
    if expected_layers is not None and len(layer_indices) != expected_layers:
        raise ValueError(
            f"cache has {len(layer_indices)} layers, expected {expected_layers}"
        )
    return tuple(
        (
            tensors[f"layer_{index:03d}.key"],
            tensors[f"layer_{index:03d}.value"],
        )
        for index in layer_indices
    )


def slice_cache(cache: Any, start: int, end: int) -> LegacyCache:
    """Copy a token interval from every cache layer into CPU memory."""
    return tuple(
        (
            key[:, :, start:end].detach().to("cpu").contiguous(),
            value[:, :, start:end].detach().to("cpu").contiguous(),
        )
        for key, value in cache_to_legacy(cache)
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a SparseCache manifest."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "sparsecache.chunk-kv.v1":
        raise ValueError("unsupported SparseCache manifest format")
    manifest["_root"] = str(manifest_path.parent.resolve())
    return manifest


@torch.inference_mode()
def compose_cache(model, manifest: dict[str, Any]) -> tuple[LegacyCache, dict[str, Any]]:
    """Load, RoPE-relocate, and concatenate system and document caches."""
    root = Path(manifest["_root"])
    layers = int(manifest["model"]["layers"])
    system = load_cache(root / manifest["system"]["file"], layers)
    key_parts = [[key] for key, _ in system]
    value_parts = [[value] for _, value in system]
    cursor = int(manifest["system"]["tokens"])
    ranges = []
    relocated_tokens = 0

    for chunk in manifest["chunks"]:
        chunk_cache = load_cache(root / chunk["file"], layers)
        source_start = int(chunk["source_start"])
        tokens = int(chunk["tokens"])
        for layer_index, (key, value) in enumerate(chunk_cache):
            moved_key = relocate_key(
                model,
                key,
                source_start=source_start,
                target_start=cursor,
            )
            key_parts[layer_index].append(moved_key)
            value_parts[layer_index].append(value)
        if source_start != cursor:
            relocated_tokens += tokens
        ranges.append(
            {
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "start": cursor,
                "end": cursor + tokens,
            }
        )
        cursor += tokens

    composed = tuple(
        (
            torch.cat(key_parts[index], dim=2),
            torch.cat(value_parts[index], dim=2),
        )
        for index in range(layers)
    )
    return composed, {
        "system_tokens": int(manifest["system"]["tokens"]),
        "document_tokens": cursor - int(manifest["system"]["tokens"]),
        "prefix_tokens": cursor,
        "relocated_tokens": relocated_tokens,
        "chunk_ranges": ranges,
    }


def cache_bytes(cache: LegacyCache) -> int:
    """Return the logical number of bytes occupied by K and V tensors."""
    return sum(
        (key.numel() + value.numel()) * key.element_size()
        for key, value in cache
    )
