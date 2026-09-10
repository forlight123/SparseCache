"""Opt-in shape-invariant FlashInfer verification for vLLM.

Normal speculative verification sends a multi-token causal block through a
prefill kernel.  Its floating-point reduction tree can differ from ordinary
one-token decoding, so an accepted block can leave non-canonical KV state.

This prototype instead exposes every speculative position as an independent
decode query.  Query ``j`` references the same physical paged cache but sees
only the canonical prefix ending at ``base + j``.  FlashInfer's fixed split-KV
mode then keeps each row's reduction partition independent of batch shape.

The patch is intentionally restricted to pure, uniform speculative-decode
batches.  Other batches retain vLLM's normal path.  The current end-to-end gate
uses ``--max-num-seqs 1 --enforce-eager``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

_INSTALLED = False


@dataclass(frozen=True)
class ExpandedDecodeMetadata:
    """CPU metadata for treating a speculative block as batched qlen=1."""

    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    last_page_len: tuple[int, ...]
    visible_lengths: tuple[int, ...]


def expand_decode_page_metadata(
    seq_lens: tuple[int, ...],
    query_lens: tuple[int, ...],
    block_tables: tuple[tuple[int, ...], ...],
    page_size: int,
) -> ExpandedDecodeMetadata:
    """Expand physical page tables into one logical sequence per query row."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if not seq_lens or len(seq_lens) != len(query_lens):
        raise ValueError("sequence and query lengths must be non-empty and aligned")
    if len(block_tables) != len(seq_lens):
        raise ValueError("one block table is required per request")

    indptr = [0]
    indices: list[int] = []
    last_page_len: list[int] = []
    visible_lengths: list[int] = []
    for seq_len, query_len, pages in zip(
        seq_lens, query_lens, block_tables, strict=True
    ):
        if query_len <= 0 or seq_len < query_len:
            raise ValueError("each request needs 0 < query_len <= seq_len")
        base = seq_len - query_len
        for offset in range(1, query_len + 1):
            visible = base + offset
            page_count = (visible + page_size - 1) // page_size
            if page_count > len(pages):
                raise ValueError("block table does not cover the visible KV prefix")
            indices.extend(pages[:page_count])
            indptr.append(len(indices))
            last_page_len.append((visible - 1) % page_size + 1)
            visible_lengths.append(visible)
    return ExpandedDecodeMetadata(
        indptr=tuple(indptr),
        indices=tuple(indices),
        last_page_len=tuple(last_page_len),
        visible_lengths=tuple(visible_lengths),
    )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def install() -> None:
    """Install the vLLM 0.23 FlashInfer metadata patch once per process."""

    global _INSTALLED
    if _INSTALLED:
        return

    import torch
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    required = (
        "_compute_flashinfer_kv_metadata",
        "_init_reorder_batch_threshold",
        "build",
        "paged_kv_indices",
    )
    missing = [name for name in required[:3] if not hasattr(FlashInferMetadataBuilder, name)]
    if missing:
        raise RuntimeError(f"unsupported vLLM FlashInfer builder: missing {missing}")

    fixed_split_pages = _positive_int_env(
        "SPARSECACHE_SHAPE_FIXED_SPLIT_PAGES", 64
    )
    original_init = FlashInferMetadataBuilder.__init__
    original_build = FlashInferMetadataBuilder.build
    original_compute = FlashInferMetadataBuilder._compute_flashinfer_kv_metadata

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        speculative = self.vllm_config.speculative_config
        spec_tokens = (
            int(speculative.num_speculative_tokens)
            if speculative is not None
            and speculative.num_speculative_tokens is not None
            else 0
        )
        parallel_factor = (
            2 if speculative is not None and speculative.parallel_drafting else 1
        )
        max_query_tokens = 1 + parallel_factor * spec_tokens
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.decode_fixed_split_size = fixed_split_pages
        self.disable_split_kv = False
        self._sparsecache_max_query_tokens = max_query_tokens
        self._sparsecache_expand_decode = False
        self._sparsecache_query_lens = ()

        # Native FlashInfer normally sizes these buffers by request count.  The
        # shape-invariant path needs one logical page table per query token.
        if max_query_tokens > 1:
            indptr_capacity = self.paged_kv_indptr.gpu.numel()
            request_capacity = indptr_capacity - 1
            index_capacity = self.paged_kv_indices.gpu.numel()
            self.paged_kv_indptr = self._make_buffer(
                request_capacity * max_query_tokens + 1
            )
            self.paged_kv_indptr_cpu_buffer = torch.zeros_like(
                self.paged_kv_indptr.cpu, pin_memory=self.pin_memory
            )
            self.paged_kv_last_page_len = self._make_buffer(
                request_capacity * max_query_tokens
            )
            self.paged_kv_indices = self._make_buffer(
                index_capacity * max_query_tokens
            )

    def patched_build(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ):
        query_lens_tensor = (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        query_lens = tuple(int(value) for value in query_lens_tensor.tolist())
        threshold = int(self.reorder_batch_threshold or 1)
        pure_short_batch = bool(query_lens) and max(query_lens) <= threshold
        has_spec_block = any(length > 1 for length in query_lens)
        all_uniform = len(set(query_lens)) == 1
        self._sparsecache_expand_decode = (
            common_prefix_len == 0
            and pure_short_batch
            and has_spec_block
            and all_uniform
        )
        self._sparsecache_query_lens = query_lens
        if has_spec_block and pure_short_batch and not self._sparsecache_expand_decode:
            raise RuntimeError(
                "shape-invariant verifier requires a pure uniform decode batch"
            )
        try:
            return original_build(
                self, common_prefix_len, common_attn_metadata, fast_build
            )
        finally:
            self._sparsecache_expand_decode = False
            self._sparsecache_query_lens = ()

    def patched_compute(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: Any,
        num_reqs: int,
        page_size: int,
    ):
        if not self._sparsecache_expand_decode:
            return original_compute(
                self,
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )

        query_lens = tuple(int(value) for value in self._sparsecache_query_lens)
        # Uniformity is checked in ``patched_build``.  Repeating the GPU block
        # table and delegating to vLLM's existing Triton copy kernel avoids a
        # page-table D2H synchronization on the verifier critical path.
        query_len = query_lens[0]
        expanded_seq_lens = np.concatenate(
            [
                np.arange(
                    int(seq_len) - current_query_len + 1,
                    int(seq_len) + 1,
                    dtype=np.int32,
                )
                for seq_len, current_query_len in zip(
                    seq_lens_np[:num_reqs], query_lens, strict=True
                )
            ]
        )
        logical_sequences = int(expanded_seq_lens.size)
        if logical_sequences > self.paged_kv_last_page_len.gpu.numel():
            raise RuntimeError("shape-invariant decode metadata exceeds buffer capacity")
        expanded_num_blocks = (
            expanded_seq_lens + (page_size - 1)
        ) // page_size
        required_indices = int(expanded_num_blocks.sum())
        if required_indices > self.paged_kv_indices.gpu.numel():
            raise RuntimeError("shape-invariant page indices exceed buffer capacity")
        expanded_block_tables = block_table_tensor[:num_reqs].repeat_interleave(
            query_len, dim=0
        )
        return original_compute(
            self,
            expanded_num_blocks,
            expanded_seq_lens,
            expanded_block_tables,
            logical_sequences,
            page_size,
        )

    FlashInferMetadataBuilder.__init__ = patched_init
    FlashInferMetadataBuilder.build = patched_build
    FlashInferMetadataBuilder._compute_flashinfer_kv_metadata = patched_compute
    _INSTALLED = True
    print(
        "SparseCache shape-invariant FlashInfer verifier enabled "
        f"(fixed_split_pages={fixed_split_pages})",
        flush=True,
    )
