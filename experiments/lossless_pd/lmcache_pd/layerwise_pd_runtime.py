"""Opt-in layer-major LMCache P/D transport for lossless verification.

The stock LMCache revision used by this repository supports layerwise local
storage and ordinary P/D transfer independently, but does not connect the two
paths.  This research hook joins them without modifying the adjacent LMCache
checkout:

* every wire object is one ``(target layer, token chunk)``;
* sparse Anchor objects are sent first and are reused by the exact Target;
* all remaining objects are sent once, in Target layer order;
* one-sided NIXL completion is published to D before a layer is made readable;
* a D request may promise the transferred prefix to the scheduler, while the
  worker blocks at the normal per-layer connector hook.

The hook is deliberately fail-closed and only activates when
``SPARSECACHE_LAYERWISE_PD_PATCH=1`` and LMCache ``use_layerwise`` is enabled.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.lossless_pd.lmcache_pd.anchor_runtime import (
    _claim_seed,
    _send_anchor_notification,
    anchor_indices,
)
from experiments.lossless_pd.lmcache_pd.priority_sidecar import (
    load_priority_sidecar,
    token_digest,
)

_INSTALLED = False
_TRACE_LOCK = threading.Lock()
_REQUEST_SPECS: dict[str, Any] = {}


@dataclass(frozen=True)
class TransferPhase:
    """One serialized wire batch with homogeneous layer/chunk layout."""

    ordinal: int
    kind: str
    layer: int
    chunks: tuple[int, ...]
    layer_complete: bool = False
    is_last: bool = False


def empty_layerwise_retrieve(tokens, num_layers: int):
    """Emit LMCache's layerwise protocol for an already-vLLM-cached prefix."""

    import torch

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
    # LMCache's vLLM adapter primes twice, advances once per layer, and reads
    # the final mask after the device synchronization marker.
    yield torch.sum(ret_mask)
    for _ in range(1, num_layers):
        yield None
    yield None
    yield ret_mask


def layerwise_retrieve_with_empty_fallback(
    original, engine, tokens, mask=None, **kwargs
):
    """Complete the upstream generator when its storage lookup is empty."""

    import torch

    if mask is not None and not bool(mask.any().item()):
        yield from empty_layerwise_retrieve(tokens, int(engine.num_layers))
        return
    try:
        yield from original(engine, tokens, mask=mask, **kwargs)
    except UnboundLocalError as error:
        # LMCache 0.5.4rc5 reaches this point only after emitting every layer
        # marker for an empty storage lookup.  Supply the missing final mask;
        # do not hide unrelated failures from the upstream generator.
        if "mem_obj_consumer" not in str(error):
            raise
        yield torch.zeros(len(tokens), dtype=torch.bool, device="cpu")


def _patch_empty_layerwise_retrieve() -> None:
    """Handle requests fully covered by vLLM prefix caching.

    LMCache 0.5.4rc5's layerwise generator unconditionally synchronizes an
    undefined ``mem_obj_consumer`` when its retrieval mask contains no work.
    This state is expected during canonical repair because the just-finished
    speculative request leaves the immutable prompt resident in vLLM.
    """

    from lmcache.v1.cache_engine import LMCacheEngine

    original = LMCacheEngine.retrieve_layer

    def retrieve_with_empty_prefix(self, tokens, mask=None, **kwargs):
        yield from layerwise_retrieve_with_empty_fallback(
            original, self, tokens, mask=mask, **kwargs
        )

    LMCacheEngine.retrieve_layer = retrieve_with_empty_prefix


def build_transfer_schedule(
    num_layers: int,
    num_chunks: int,
    draft_layers: Sequence[int],
    selected_chunks: Sequence[int],
) -> tuple[TransferPhase, ...]:
    """Build an Anchor-first, byte-exact schedule.

    Every ``(layer, chunk)`` pair occurs exactly once.  Anchor pairs therefore
    become part of the authoritative cache instead of being retransmitted in a
    second full-cache object.
    """

    if num_layers <= 0 or num_chunks <= 0:
        raise ValueError("num_layers and num_chunks must be positive")
    draft = tuple(int(layer) for layer in draft_layers)
    selected = tuple(int(chunk) for chunk in selected_chunks)
    if not draft or len(set(draft)) != len(draft):
        raise ValueError("draft_layers must be nonempty and distinct")
    if min(draft) < 0 or max(draft) >= num_layers:
        raise ValueError("draft layer lies outside the Target")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_chunks must be nonempty and distinct")
    if min(selected) < 0 or max(selected) >= num_chunks:
        raise ValueError("selected chunk lies outside the request")

    phases: list[TransferPhase] = []
    ordinal = 0
    for layer in draft:
        phases.append(
            TransferPhase(
                ordinal,
                "anchor",
                layer,
                selected,
                layer_complete=len(selected) == num_chunks,
            )
        )
        ordinal += 1

    selected_set = set(selected)
    all_chunks = tuple(range(num_chunks))
    for layer in range(num_layers):
        chunks = (
            tuple(chunk for chunk in all_chunks if chunk not in selected_set)
            if layer in draft
            else all_chunks
        )
        if chunks:
            # For a draft layer this residual phase completes the Anchor
            # objects already written for the same layer.  For all other
            # layers it contains the entire authoritative layer.
            phases.append(
                TransferPhase(
                    ordinal,
                    "target",
                    layer,
                    chunks,
                    layer_complete=True,
                )
            )
            ordinal += 1

    phases[-1] = replace(phases[-1], is_last=True)
    assert_schedule_exact(phases, num_layers=num_layers, num_chunks=num_chunks)
    return tuple(phases)


def assert_schedule_exact(
    phases: Iterable[TransferPhase], *, num_layers: int, num_chunks: int
) -> None:
    """Raise when a schedule drops or retransmits an authoritative KV object."""

    pairs = [(phase.layer, chunk) for phase in phases for chunk in phase.chunks]
    expected = {
        (layer, chunk) for layer in range(num_layers) for chunk in range(num_chunks)
    }
    if len(pairs) != len(set(pairs)):
        raise ValueError("transfer schedule retransmits a layer/chunk object")
    if set(pairs) != expected:
        missing = sorted(expected - set(pairs))
        extra = sorted(set(pairs) - expected)
        raise ValueError(
            f"non-exact transfer schedule: missing={missing}, extra={extra}"
        )


def schedule_byte_accounting(
    phases: Sequence[TransferPhase], chunk_bytes: Sequence[int]
) -> dict[str, int | float]:
    """Return wire-byte accounting for a single-layer chunk layout."""

    if not chunk_bytes or min(chunk_bytes) <= 0:
        raise ValueError("chunk_bytes must contain positive sizes")
    wire = 0
    anchor = 0
    seen: set[tuple[int, int]] = set()
    for phase in phases:
        for chunk in phase.chunks:
            if chunk >= len(chunk_bytes):
                raise ValueError("phase references a missing chunk size")
            pair = (phase.layer, chunk)
            if pair in seen:
                raise ValueError("byte accounting found a retransmission")
            seen.add(pair)
            size = int(chunk_bytes[chunk])
            wire += size
            if phase.kind == "anchor":
                anchor += size
    authoritative = len({phase.layer for phase in phases}) * sum(chunk_bytes)
    return {
        "wire_bytes": wire,
        "authoritative_bytes": authoritative,
        "anchor_bytes": anchor,
        "retransmitted_bytes": wire - authoritative,
        "wire_ratio": wire / authoritative,
    }


def parse_progressive_request(request: Any) -> tuple[str, int] | None:
    """Extract the fail-closed scheduler promise from a vLLM request."""

    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return None
    spec = params.get("sparsecache_progressive")
    if not isinstance(spec, dict):
        return None
    request_id = str(spec.get("request_id", ""))
    prompt_tokens = spec.get("prompt_tokens")
    if (
        not request_id
        or isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
    ):
        raise ValueError("invalid sparsecache_progressive request")
    if prompt_tokens <= 0:
        raise ValueError("progressive prompt_tokens must be positive")
    return request_id, prompt_tokens


def _append_trace(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK, destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class _StoreStep:
    request_id: str
    transfer_spec: Any
    num_layers: int
    prompt_tokens: int
    offset: int
    layer: int
    priority_chunks: tuple[int, ...] | None = None


@dataclass
class _QueuedLayer:
    keys: list[Any]
    objects: list[Any]
    location: str | None


_ACTIVE_STORE: ContextVar[_StoreStep | None] = ContextVar(
    "sparsecache_layer_store", default=None
)
_QUEUED: dict[str, dict[int, _QueuedLayer]] = {}
_QUEUE_LOCK = threading.Lock()


def _clone_phase_spec(
    base: Any,
    *,
    phase: TransferPhase,
    total_objects: int,
    phase_count: int,
    request_id: str,
    seed_record: dict[str, Any] | None,
) -> Any:
    result = replace(base) if hasattr(base, "__dataclass_fields__") else base
    if result is base:
        import copy

        result = copy.copy(base)
    result.is_last_prefill = phase.is_last
    result.total_chunks = total_objects
    result._sparsecache_phase = phase.kind
    result._sparsecache_phase_ordinal = phase.ordinal
    result._sparsecache_phase_count = phase_count
    result._sparsecache_layer_id = phase.layer
    result._sparsecache_layer_complete = phase.layer_complete
    result._sparsecache_request_id = request_id
    result._sparsecache_seed_record = seed_record
    return result


def _object_tokens(obj: Any) -> int:
    shape = tuple(int(value) for value in obj.get_shape())
    # LMCache's layerwise CUDA connector uses KV_T2D = [T, 2, D].
    if len(shape) == 3 and shape[1] == 2:
        return shape[0]
    return int(obj.get_num_tokens())


def _flush_request(
    manager: Any,
    step: _StoreStep,
    queued: dict[int, _QueuedLayer],
    original_batched_put: Any,
    *,
    draft_layers: tuple[int, ...],
    fraction: float,
    mode: str,
    trace_path: str,
) -> None:
    if set(queued) != set(range(step.num_layers)):
        raise RuntimeError("cannot flush an incomplete layerwise Target cache")
    chunk_counts = {len(item.objects) for item in queued.values()}
    if len(chunk_counts) != 1:
        raise RuntimeError("Target layers disagree on LMCache chunk count")
    num_chunks = chunk_counts.pop()
    if step.priority_chunks is None:
        selected = tuple(anchor_indices(num_chunks, fraction, mode))
        selection_source = mode
    else:
        if sorted(step.priority_chunks) != list(range(num_chunks)):
            raise RuntimeError(
                "priority sidecar does not match the live LMCache chunk count"
            )
        anchor_count = min(
            num_chunks, max(min(num_chunks, 2), math.ceil(fraction * num_chunks))
        )
        selected = step.priority_chunks[:anchor_count]
        selection_source = "query_priority_sidecar"
    schedule = build_transfer_schedule(
        step.num_layers, num_chunks, draft_layers, selected
    )
    seed_record = _claim_seed()
    chunk_tokens = [_object_tokens(obj) for obj in queued[0].objects]
    starts: list[int] = []
    ends: list[int] = []
    cursor = step.offset
    for tokens in chunk_tokens:
        starts.append(cursor)
        cursor += tokens
        ends.append(cursor)
    token_ranges = list(zip(starts, ends, strict=True))
    total_objects = step.num_layers * num_chunks

    anchor_keys: list[str] = []
    anchor_bytes: list[int] = []
    anchor_layers: list[int] = []
    anchor_object_chunks: list[int] = []
    for phase in schedule:
        layer = queued[phase.layer]
        if phase.kind == "anchor":
            for chunk in phase.chunks:
                key = layer.keys[chunk]
                anchor_keys.append(
                    key.to_string() if hasattr(key, "to_string") else str(key)
                )
                anchor_bytes.append(int(layer.objects[chunk].get_size()))
                anchor_layers.append(phase.layer)
                anchor_object_chunks.append(chunk)

    manifest = {
        "request_id": step.request_id,
        "pd_request_id": str(getattr(step.transfer_spec, "req_id", "")),
        "keys": anchor_keys,
        "key_bytes": anchor_bytes,
        "object_layers": anchor_layers,
        "object_chunk_indices": anchor_object_chunks,
        "chunk_indices": list(selected),
        "token_ranges": [list(token_ranges[index]) for index in selected],
        "prompt_tokens": step.prompt_tokens,
        "seed_record": seed_record,
        "anchor_last_ordinal": len(draft_layers) - 1,
    }

    accounting = schedule_byte_accounting(
        schedule,
        [int(obj.get_size()) for obj in queued[0].objects],
    )
    _append_trace(
        trace_path,
        {
            "event": "layerwise_schedule",
            "request_id": step.request_id,
            "pd_request_id": manifest["pd_request_id"],
            "num_layers": step.num_layers,
            "num_chunks": num_chunks,
            "draft_layers": list(draft_layers),
            "anchor_chunks": list(selected),
            "anchor_selection": selection_source,
            "phases": len(schedule),
            **accounting,
        },
    )

    for phase in schedule:
        layer = queued[phase.layer]
        keys = [layer.keys[index] for index in phase.chunks]
        objects = [layer.objects[index] for index in phase.chunks]
        spec = _clone_phase_spec(
            step.transfer_spec,
            phase=phase,
            total_objects=total_objects,
            phase_count=len(schedule),
            request_id=step.request_id,
            seed_record=seed_record,
        )
        spec._sparsecache_chunk_indices = tuple(phase.chunks)
        spec._sparsecache_token_ranges = tuple(
            token_ranges[index] for index in phase.chunks
        )
        spec._sparsecache_prompt_tokens = step.prompt_tokens
        if phase.ordinal == manifest["anchor_last_ordinal"]:
            spec._sparsecache_anchor_manifest = manifest
        original_batched_put(
            manager,
            keys,
            objects,
            transfer_spec=spec,
            location=layer.location,
        )


def _patch_memory_format() -> None:
    from lmcache.v1.memory_management import MemoryFormat

    original = MemoryFormat.token_dim

    def corrected_token_dim(self):
        if self == MemoryFormat.KV_T2D:
            return 0
        return original(self)

    MemoryFormat.token_dim = corrected_token_dim


def _patch_layer_cache_key_parser() -> None:
    """Teach the async P/D backend to deserialize layerwise cache keys."""

    from lmcache.utils import CacheEngineKey, LayerCacheEngineKey

    original = CacheEngineKey.from_string

    def parse_base_or_layer(value: str):
        parts = value.strip().split("@")
        if len(parts) >= 6 and parts[5].isdigit():
            return LayerCacheEngineKey.from_string(value)
        return original(value)

    CacheEngineKey.from_string = staticmethod(parse_base_or_layer)


def _patch_pd_allocator() -> None:
    import torch
    from lmcache import torch_dev
    from lmcache.integration.vllm.utils import get_size_bytes
    from lmcache.v1.memory_allocators.paged_cpu_gpu_memory_allocator import (
        PagedCpuGpuMemoryAllocator,
    )
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.storage_backend.pd_backend_async import PDBackendAsync

    original = PDBackendAsync.initialize_allocator

    def initialize_layerwise(self, config, metadata):
        if not getattr(config, "use_layerwise", False):
            return original(self, config, metadata)
        if metadata.use_mla:
            raise ValueError("SparseCache layerwise P/D does not support MLA")
        shape = torch.Size(
            [
                int(metadata.kv_shape[2]),
                int(metadata.kv_shape[1]),
                int(metadata.kv_shape[3]) * int(metadata.kv_shape[4]),
            ]
        )
        if shape[1] != 2:
            raise ValueError("expected a K/V axis of size two")
        if self.corrected_device != "cpu":
            torch_dev.set_device(self.corrected_device)
        allocator = PagedCpuGpuMemoryAllocator()
        init_fn = (
            allocator.init_cpu_memory_allocator
            if self.corrected_device == "cpu"
            else allocator.init_gpu_memory_allocator
        )
        shapes = [shape]
        dtypes = [metadata.kv_dtype]
        object_bytes = get_size_bytes(shapes, dtypes)
        original_bytes = int(config.pd_buffer_size)
        aligned_bytes = original_bytes // object_bytes * object_bytes
        if aligned_bytes == 0 and original_bytes > 0:
            raise ValueError("pd_buffer_size is smaller than one layer/chunk object")
        self._chunk_size_bytes = object_bytes
        self._aligned_buffer_size = aligned_bytes
        self._chunk_token_size = shape[0]
        self._sparsecache_num_layers = int(metadata.kv_shape[0])
        capacity_tokens = (
            aligned_bytes // object_bytes // self._sparsecache_num_layers * shape[0]
        )
        if (
            config.pd_max_prefill_len > 0
            and capacity_tokens < config.pd_max_prefill_len
        ):
            raise ValueError(
                "layerwise PD buffer is too small: "
                f"capacity_tokens={capacity_tokens}, "
                f"pd_max_prefill_len={config.pd_max_prefill_len}"
            )
        init_fn(
            aligned_bytes,
            shapes,
            dtypes,
            MemoryFormat.KV_T2D,
        )
        return allocator

    PDBackendAsync.initialize_allocator = initialize_layerwise


def _patch_layerwise_connector_format() -> None:
    """Repair LMCache's direct P/D layerwise connector initialization.

    LMCache only discovers ``engine_kv_format`` inside the intermediate-buffer
    branch.  P/D deliberately disables that buffer, although its direct device
    transfer still requires the format.  Discover it once from vLLM's registered
    cache without changing the adjacent LMCache checkout.
    """

    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.gpu_connectors import (
        VLLMPagedMemLayerwiseGPUConnector,
    )
    from lmcache.v1.gpu_connector.utils import (
        assert_is_vllm_mla_or_flash_attn_or_flash_infer,
        normalize_kv_and_discover_format,
    )

    original = VLLMPagedMemLayerwiseGPUConnector._lazy_initialize_buffer

    def discover_for_direct_pd(self, kv_caches):
        original(self, kv_caches)
        if hasattr(self, "engine_kv_format"):
            return
        self.engine_kv_format, self.kvcaches = normalize_kv_and_discover_format(
            self.kvcaches,
            EngineType.VLLM,
            layout_hints=self.layout_hints,
        )
        assert_is_vllm_mla_or_flash_attn_or_flash_infer(self.engine_kv_format)

    VLLMPagedMemLayerwiseGPUConnector._lazy_initialize_buffer = discover_for_direct_pd


def _patch_layerwise_direct_scatter() -> None:
    """Coalesce direct layerwise GPU gather/scatter into one kernel per layer.

    LMCache's no-intermediate-buffer layerwise H2D path currently supplies the
    concatenated mapping for the whole prefix to every individual chunk.  It is
    accidentally correct for one chunk and indexes past the source object for
    longer prompts.  Calling the corrected operation once per small object is
    also prohibitively expensive: a 64-token layout creates thousands of CUDA
    launches for one request.

    P/D staging objects already live on the GPU.  Pack/unpack them on the
    connector stream and invoke LMCache's paged-cache transfer exactly once per
    Target layer.  This changes neither the wire objects nor their schedule;
    it only removes object-granularity CUDA extension launches.  Set
    ``SPARSECACHE_COALESCE_LAYER_IO=0`` to retain the corrected per-object H2D
    path as an ablation.
    """

    import torch
    from lmcache import device_ops, lmcache_native
    from lmcache.v1.gpu_connector.gpu_connectors import (
        VLLMPagedMemLayerwiseGPUConnector,
    )
    from lmcache.v1.memory_management import MemoryFormat

    original_to_gpu = VLLMPagedMemLayerwiseGPUConnector.batched_to_gpu
    original_from_gpu = VLLMPagedMemLayerwiseGPUConnector.batched_from_gpu

    def validate_slices(starts, ends, memory_objs):
        if not (len(starts) == len(ends) == len(memory_objs)):
            raise ValueError("layerwise starts/ends/objects length mismatch")
        lengths = []
        for start, end, memory_obj in zip(starts, ends, memory_objs, strict=True):
            length = int(end) - int(start)
            if length <= 0:
                raise ValueError("layerwise token ranges must be nonempty")
            if memory_obj.tensor is None or int(memory_obj.tensor.shape[0]) < length:
                raise ValueError("layerwise object is shorter than its token range")
            lengths.append(length)
        return lengths

    def check_format(self, memory_obj):
        expected_format = (
            MemoryFormat.KV_MLA_FMT if self.use_mla else MemoryFormat.KV_T2D
        )
        if memory_obj.metadata.fmt != expected_format:
            raise ValueError(f"expected {expected_format}, got {memory_obj.metadata.fmt}")

    def scatter_direct(self, starts, ends, **kwargs):
        if self.use_gpu:
            yield from original_to_gpu(self, starts, ends, **kwargs)
            return

        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise RuntimeError("vLLM KV cache pointers were not initialized")
        if "slot_mapping" not in kwargs or "sync" not in kwargs:
            raise ValueError("slot_mapping and sync are required")
        slot_mapping = kwargs["slot_mapping"]
        sync = bool(kwargs["sync"])
        self._lazy_initialize_buffer(self.kvcaches)
        current_stream = torch.cuda.current_stream()
        coalesce = os.environ.get("SPARSECACHE_COALESCE_LAYER_IO", "1") != "0"

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            with torch.cuda.stream(self.load_stream):
                lengths = validate_slices(starts, ends, memory_objs_layer)
                for memory_obj in memory_objs_layer:
                    check_format(self, memory_obj)
                if coalesce:
                    sources = [
                        memory_obj.tensor[:length]
                        for memory_obj, length in zip(
                            memory_objs_layer, lengths, strict=True
                        )
                    ]
                    packed = sources[0] if len(sources) == 1 else torch.cat(sources)
                    mapping_parts = [
                        slot_mapping[start:end]
                        for start, end in zip(starts, ends, strict=True)
                    ]
                    packed_mapping = (
                        mapping_parts[0]
                        if len(mapping_parts) == 1
                        else torch.cat(mapping_parts)
                    )
                    device_ops.single_layer_kv_transfer(
                        packed,
                        self.kvcaches[layer_id],
                        packed_mapping,
                        lmcache_native.TransferDirection.H2D,
                        self.engine_kv_format,
                        token_major=True,
                    )
                else:
                    for start, end, memory_obj, length in zip(
                        starts, ends, memory_objs_layer, lengths, strict=True
                    ):
                        device_ops.single_layer_kv_transfer(
                            memory_obj.tensor[:length],
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            lmcache_native.TransferDirection.H2D,
                            self.engine_kv_format,
                            token_major=True,
                        )
        yield
        if sync:
            current_stream.wait_stream(self.load_stream)
        yield

    def gather_direct(self, memory_objs, starts, ends, **kwargs):
        if self.use_gpu or os.environ.get("SPARSECACHE_COALESCE_LAYER_IO", "1") == "0":
            yield from original_from_gpu(self, memory_objs, starts, ends, **kwargs)
            return

        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise RuntimeError("vLLM KV cache pointers were not initialized")
        if "slot_mapping" not in kwargs or "sync" not in kwargs:
            raise ValueError("slot_mapping and sync are required")
        slot_mapping = kwargs["slot_mapping"]
        sync = bool(kwargs["sync"])
        self._lazy_initialize_buffer(self.kvcaches)
        current_stream = torch.cuda.current_stream()
        mapping_parts = [
            slot_mapping[start:end]
            for start, end in zip(starts, ends, strict=True)
        ]
        packed_mapping = (
            mapping_parts[0] if len(mapping_parts) == 1 else torch.cat(mapping_parts)
        )
        packed = torch.empty(
            self.get_shape(len(packed_mapping)),
            dtype=self.dtype,
            device=self.device,
        )

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            lengths = validate_slices(starts, ends, memory_objs_layer)
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                device_ops.single_layer_kv_transfer(
                    packed,
                    self.kvcaches[layer_id],
                    packed_mapping,
                    lmcache_native.TransferDirection.D2H,
                    self.engine_kv_format,
                    token_major=True,
                )
                cursor = 0
                for memory_obj, length in zip(
                    memory_objs_layer, lengths, strict=True
                ):
                    memory_obj.tensor[:length].copy_(
                        packed[cursor : cursor + length], non_blocking=True
                    )
                    check_format(self, memory_obj)
                    cursor += length
            yield
            if sync:
                self.store_stream.synchronize()
        yield

    VLLMPagedMemLayerwiseGPUConnector.batched_to_gpu = scatter_direct
    VLLMPagedMemLayerwiseGPUConnector.batched_from_gpu = gather_direct


def _patch_pd_receiver_get() -> None:
    from lmcache.v1.storage_backend.pd_backend_async import PDBackendAsync

    async def contains_written_prefix(self, lookup_id, keys, pin=False):
        del lookup_id
        ready = getattr(self, "_sparsecache_ready_keys", set())
        count = 0
        with self.data_lock:
            for key in keys:
                obj = self.data.get(key)
                if obj is None or key.to_string() not in ready:
                    break
                if pin:
                    obj.pin()
                count += 1
        return count

    async def batched_get_when_written(
        self, lookup_id: str, keys: list[Any], transfer_spec: Any = None
    ) -> list[Any]:
        del lookup_id, transfer_spec
        timeout = float(os.environ.get("SPARSECACHE_LAYER_WAIT_TIMEOUT_SEC", "30"))
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            ready = getattr(self, "_sparsecache_ready_keys", set())
            with self.data_lock:
                complete = all(key.to_string() in ready for key in keys)
                if complete:
                    objects = [self.data.get(key) for key in keys]
                    if all(obj is not None for obj in objects):
                        # Transfer the backend's owning references to the
                        # layerwise retrieval caller.  LMCache's layerwise
                        # path decrements the caller reference and unpins each
                        # object after the H2D scatter, but unlike the ordinary
                        # path it never invokes remove_after_retrieve.  Keeping
                        # these entries in ``data`` therefore leaks one whole
                        # Target per request and progressively slows NIXL.
                        for key, obj in zip(keys, objects, strict=True):
                            assert obj is not None
                            obj.ref_count_up()
                            self.data.pop(key)
                            ready.discard(key.to_string())
                            obj.ref_count_down()
                        return objects
            if asyncio.get_running_loop().time() >= deadline:
                missing = [
                    key.to_string() for key in keys if key.to_string() not in ready
                ]
                raise TimeoutError(
                    f"Target layer did not become NIXL-complete: {missing[:4]}"
                )
            await asyncio.sleep(0.0005)

    PDBackendAsync.batched_async_contains = contains_written_prefix
    PDBackendAsync.batched_get_non_blocking = batched_get_when_written


def _patch_layerwise_store(
    *,
    draft_layers: tuple[int, ...],
    fraction: float,
    mode: str,
    trace_path: str,
    priority_by_digest: dict[str, tuple[int, ...]] | None,
) -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.storage_backend.storage_manager import StorageManager

    original_save = LMCacheConnectorV1Impl.save_kv_layer
    original_store = LMCacheEngine.store_layer
    original_put = StorageManager.batched_put

    def save_with_spec(self, *args, **kwargs):
        metadata = self._parent._get_connector_metadata()
        for request in getattr(metadata, "requests", ()):
            if request.disagg_spec is not None:
                _REQUEST_SPECS[str(request.req_id)] = request.disagg_spec
        return original_save(self, *args, **kwargs)

    def store_with_spec(self, tokens=None, mask=None, **kwargs):
        base = original_store(self, tokens=tokens, mask=mask, **kwargs)
        request_id = str(kwargs.get("req_id", ""))
        transfer_spec = _REQUEST_SPECS.get(request_id)
        if transfer_spec is None:
            return base
        prompt_tokens = len(tokens) if tokens is not None else 0
        offset = int(kwargs.get("offset", 0))
        priority_chunks = None
        if priority_by_digest is not None:
            if tokens is None:
                raise RuntimeError("query-priority transfer requires prompt tokens")
            digest = token_digest(tokens)
            priority_chunks = priority_by_digest.get(digest)
            if priority_chunks is None:
                raise RuntimeError(
                    "configured query-priority sidecar has no exact prompt match"
                )

        def drive():
            step_index = 0
            try:
                while True:
                    layer = step_index - 1
                    context = _StoreStep(
                        request_id,
                        transfer_spec,
                        int(self.num_layers),
                        prompt_tokens,
                        offset,
                        layer,
                        priority_chunks,
                    )
                    token = _ACTIVE_STORE.set(context)
                    try:
                        value = next(base)
                    except StopIteration:
                        return
                    finally:
                        _ACTIVE_STORE.reset(token)
                    step_index += 1
                    yield value
            finally:
                _REQUEST_SPECS.pop(request_id, None)

        return drive()

    def queue_or_put(
        self,
        keys,
        memory_objs,
        transfer_spec=None,
        location=None,
    ):
        step = _ACTIVE_STORE.get()
        if step is None or transfer_spec is not None or step.layer < 0:
            return original_put(
                self,
                keys,
                memory_objs,
                transfer_spec=transfer_spec,
                location=location,
            )
        if step.layer >= step.num_layers:
            raise RuntimeError("layerwise store produced too many layers")
        with _QUEUE_LOCK:
            queued = _QUEUED.setdefault(step.request_id, {})
            if step.layer in queued:
                raise RuntimeError("layerwise store queued one Target layer twice")
            queued[step.layer] = _QueuedLayer(list(keys), list(memory_objs), location)
            if len(queued) != step.num_layers:
                return None
            _QUEUED.pop(step.request_id)
        _flush_request(
            self,
            step,
            queued,
            original_put,
            draft_layers=draft_layers,
            fraction=fraction,
            mode=mode,
            trace_path=trace_path,
        )
        return None

    LMCacheConnectorV1Impl.save_kv_layer = save_with_spec
    LMCacheEngine.store_layer = store_with_spec
    StorageManager.batched_put = queue_or_put


def _patch_ordered_sender(trace_path: str, notify_endpoint: str) -> None:
    from lmcache.v1.storage_backend.pd_backend_async import PDBackendAsync

    original_abort = PDBackendAsync._abort_request

    async def mark_failed_before_abort(self, req_id, *args, **kwargs):
        failed = getattr(self, "_sparsecache_failed_requests", None)
        if failed is None:
            failed = self._sparsecache_failed_requests = set()
        failed.add(str(req_id))
        return await original_abort(self, req_id, *args, **kwargs)

    PDBackendAsync._abort_request = mark_failed_before_abort
    original = PDBackendAsync._async_transfer_task

    async def ordered(self, *args, **kwargs):
        spec = kwargs.get("transfer_spec")
        if spec is None and len(args) >= 5:
            spec = args[4]
        ordinal = getattr(spec, "_sparsecache_phase_ordinal", None)
        if ordinal is None:
            return await original(self, *args, **kwargs)
        req_id = str(getattr(spec, "req_id", ""))
        states = getattr(self, "_sparsecache_order_states", None)
        if states is None:
            states = self._sparsecache_order_states = {}
        state = states.get(req_id)
        if state is None:
            state = states[req_id] = {
                "condition": asyncio.Condition(),
                "next": 0,
            }
        condition = state["condition"]
        async with condition:
            await condition.wait_for(lambda: state["next"] == ordinal)
        failed = getattr(self, "_sparsecache_failed_requests", set())
        if req_id in failed:
            objects = kwargs.get("memory_objs")
            if objects is None and len(args) >= 2:
                objects = args[1]
            for obj in objects or ():
                obj.ref_count_down()
            self._notify_staging_freed()
            result = None
            started_ns = time.perf_counter_ns()
        else:
            started_ns = time.perf_counter_ns()
        try:
            if req_id not in failed:
                result = await original(self, *args, **kwargs)
            finished_ns = time.perf_counter_ns()
            if req_id in getattr(self, "_sparsecache_failed_requests", set()):
                return result
            keys = kwargs.get("keys")
            if keys is None and args:
                keys = args[0]
            objects = kwargs.get("memory_objs")
            if objects is None and len(args) >= 2:
                objects = args[1]
            key_strings = [
                key.to_string() if hasattr(key, "to_string") else str(key)
                for key in (keys or ())
            ]
            row = {
                "event": "layer_ready",
                "request_id": str(getattr(spec, "_sparsecache_request_id", "")),
                "pd_request_id": req_id,
                "phase": str(getattr(spec, "_sparsecache_phase", "")),
                "layer": int(getattr(spec, "_sparsecache_layer_id", -1)),
                "layer_complete": bool(
                    getattr(spec, "_sparsecache_layer_complete", False)
                ),
                "ordinal": int(ordinal),
                "keys": key_strings,
                "bytes": sum(int(obj.get_size()) for obj in (objects or ())),
                "chunk_indices": list(getattr(spec, "_sparsecache_chunk_indices", ())),
                "token_ranges": [
                    list(item)
                    for item in getattr(spec, "_sparsecache_token_ranges", ())
                ],
                "prompt_tokens": int(getattr(spec, "_sparsecache_prompt_tokens", 0)),
                "started_ns": started_ns,
                "finished_ns": finished_ns,
                "write_ms": (finished_ns - started_ns) / 1e6,
                "complete": True,
            }
            _append_trace(trace_path, row)
            if notify_endpoint:
                _send_anchor_notification(notify_endpoint, row)

            manifest = getattr(spec, "_sparsecache_anchor_manifest", None)
            if manifest is not None:
                anchor_row = {
                    "event": "nixl_write",
                    "phase": "anchor",
                    **manifest,
                    "bytes": sum(manifest["key_bytes"]),
                    "resident_bytes": sum(manifest["key_bytes"]),
                    "finished_ns": finished_ns,
                }
                _append_trace(trace_path, anchor_row)
                if notify_endpoint:
                    _send_anchor_notification(notify_endpoint, anchor_row)
            return result
        finally:
            async with condition:
                state["next"] += 1
                condition.notify_all()
                if state["next"] >= int(
                    getattr(spec, "_sparsecache_phase_count", math.inf)
                ):
                    states.pop(req_id, None)
                    getattr(self, "_sparsecache_failed_requests", set()).discard(req_id)

    PDBackendAsync._async_transfer_task = ordered


def _patch_scheduler_promise() -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorV1Impl,
        LoadSpec,
    )

    original = LMCacheConnectorV1Impl.get_num_new_matched_tokens

    def progressive_match(self, request, num_computed_tokens):
        progressive = parse_progressive_request(request)
        if progressive is None:
            return original(self, request, num_computed_tokens)
        _, prompt_tokens = progressive
        if prompt_tokens > int(request.num_tokens):
            raise ValueError("promised progressive prefix exceeds the D prompt")
        req_id = request.request_id
        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=int(num_computed_tokens),
            lmcache_cached_tokens=prompt_tokens,
            can_load=False,
        )
        need = prompt_tokens - int(num_computed_tokens)
        if prompt_tokens == int(request.num_tokens):
            need -= 1
        return max(0, need)

    LMCacheConnectorV1Impl.get_num_new_matched_tokens = progressive_match


def install() -> None:
    """Install the experimental layer-major P/D protocol once per process."""

    global _INSTALLED
    if _INSTALLED:
        return
    fraction = float(os.environ.get("SPARSECACHE_ANCHOR_FRACTION", "0.1"))
    mode = os.environ.get("SPARSECACHE_ANCHOR_MODE", "protected_uniform")
    draft_layers = tuple(
        int(item)
        for item in os.environ.get("SPARSECACHE_DRAFT_LAYERS", "1,9,17,25,33").split(
            ","
        )
        if item.strip()
    )
    trace_path = os.environ.get("SPARSECACHE_LAYERWISE_TRACE", "")
    notify_endpoint = os.environ.get("SPARSECACHE_ANCHOR_NOTIFY", "")
    priority_path = os.environ.get("SPARSECACHE_PRIORITY_SIDECAR", "")
    priority_by_digest = (
        load_priority_sidecar(Path(priority_path).resolve()) if priority_path else None
    )
    anchor_indices(1, fraction, mode)
    if not draft_layers:
        raise ValueError("SPARSECACHE_DRAFT_LAYERS must be nonempty")

    _patch_memory_format()
    _patch_layer_cache_key_parser()
    _patch_pd_allocator()
    _patch_layerwise_connector_format()
    _patch_layerwise_direct_scatter()
    _patch_pd_receiver_get()
    _patch_empty_layerwise_retrieve()
    _patch_layerwise_store(
        draft_layers=draft_layers,
        fraction=fraction,
        mode=mode,
        trace_path=trace_path,
        priority_by_digest=priority_by_digest,
    )
    _patch_ordered_sender(trace_path, notify_endpoint)
    _patch_scheduler_promise()
    _INSTALLED = True
