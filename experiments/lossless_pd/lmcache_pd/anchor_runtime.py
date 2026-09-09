"""Opt-in progressive gather/transfer instrumentation for LMCache.

This module is loaded through ``sitecustomize`` in the prefiller process.  It
does not change the bytes delivered to the decoder.  The basic mode partitions
one already-gathered NIXL batch into anchor and residual writes.  With
``SPARSECACHE_GATHER_FIRST=1`` it also moves the partition ahead of LMCache's
GPU gather: anchor chunks are gathered and submitted immediately, then residual
chunks are gathered and submitted.  The first implementation selects whole
LMCache token chunks, so every selected chunk contains all target layers and
can later be reused by the exact verifier without duplicate transfer.
"""

from __future__ import annotations

import contextvars
import copy
import json
import math
import os
import socket
import threading
import time
from collections import deque
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Sequence


_REQUEST_ID = contextvars.ContextVar("sparsecache_request_id", default="")
_PD_REQUEST_ID = contextvars.ContextVar("sparsecache_pd_request_id", default="")
_TRANSFER_PHASE = contextvars.ContextVar("sparsecache_transfer_phase", default="")
_SEED_RECORD = contextvars.ContextVar("sparsecache_seed_record", default=None)
_TRANSFER_KEYS = contextvars.ContextVar("sparsecache_transfer_keys", default=())
_TRANSFER_KEY_BYTES = contextvars.ContextVar(
    "sparsecache_transfer_key_bytes", default=()
)
_TRANSFER_INDICES = contextvars.ContextVar("sparsecache_transfer_indices", default=())
_TRACE_LOCK = threading.Lock()
_SEED_LOCK = threading.Lock()
_PENDING_SEEDS: deque[dict] = deque()
_INSTALLED = False


def anchor_indices(num_chunks: int, fraction: float, mode: str) -> list[int]:
    if num_chunks <= 0:
        return []
    if not 0 < fraction <= 1:
        raise ValueError("anchor fraction must be in (0, 1]")
    count = min(num_chunks, max(1, math.ceil(num_chunks * fraction)))
    if mode == "prefix":
        return list(range(count))
    if mode != "uniform":
        raise ValueError(f"unsupported anchor selection mode: {mode}")
    # Midpoints of equal-width bins are unique whenever count <= num_chunks.
    return [
        min(num_chunks - 1, int((index + 0.5) * num_chunks / count))
        for index in range(count)
    ]


def _append_trace(path: str, row: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK, destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def publish_seed_batch(sampled_token_ids: Sequence[Sequence[int]]) -> None:
    """Queue P-side target seeds in the request order used by LMCache store."""

    sampled_ns = time.perf_counter_ns()
    records = [
        {"seed_token_id": int(tokens[0]), "seed_sampled_ns": sampled_ns}
        for tokens in sampled_token_ids
        if tokens
    ]
    with _SEED_LOCK:
        _PENDING_SEEDS.extend(records)


def _claim_seed() -> dict | None:
    with _SEED_LOCK:
        return _PENDING_SEEDS.popleft() if _PENDING_SEEDS else None


def _send_anchor_notification(endpoint: str, row: dict) -> None:
    """Send a best-effort UDP control message after remote Anchor completion."""

    if not endpoint:
        return
    if not endpoint.startswith("udp://") or ":" not in endpoint[6:]:
        raise ValueError("anchor notify endpoint must be udp://HOST:PORT")
    host, port = endpoint[6:].rsplit(":", 1)
    payload = json.dumps(row, separators=(",", ":")).encode()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
            channel.sendto(payload, (host, int(port)))
    except OSError:
        # The control plane is advisory. A missing listener must never turn an
        # exact full-KV transfer into a failed request.
        return


def partition_indices(
    num_chunks: int, fraction: float, mode: str
) -> tuple[list[int], list[int]]:
    """Return stable anchor/residual indexes whose union is the full batch."""

    anchors = anchor_indices(num_chunks, fraction, mode)
    selected = set(anchors)
    return anchors, [index for index in range(num_chunks) if index not in selected]


def _subset(values: Sequence[Any], indexes: Sequence[int]) -> list[Any]:
    return [values[index] for index in indexes]


def _phase_spec(
    transfer_spec: Any,
    *,
    phase: str,
    is_last: bool,
    total: int,
    request_id: str = "",
    seed_record: dict | None = None,
    indices: Sequence[int] = (),
):
    """Clone a mutable LMCache DisaggSpec before asynchronous submission."""

    if is_dataclass(transfer_spec):
        result = replace(
            transfer_spec,
            is_last_prefill=is_last,
            total_chunks=getattr(transfer_spec, "total_chunks", 0) or total,
        )
    else:
        result = copy.copy(transfer_spec)
        result.is_last_prefill = is_last
        if not getattr(result, "total_chunks", 0):
            result.total_chunks = total
    # DisaggSpec is not slotted in the tested LMCache revision.  Keeping the
    # label on the cloned object avoids a shared cross-thread phase map.
    result._sparsecache_phase = phase
    result._sparsecache_request_id = request_id
    result._sparsecache_seed_record = seed_record
    result._sparsecache_indices = tuple(int(index) for index in indices)
    return result


def _install_gather_first_store(
    *, fraction: float, mode: str, trace_path: str
) -> None:
    """Patch LMCacheEngine.store to submit anchor chunks before residual gather."""

    import torch

    from lmcache.v1.cache_engine import LMCacheEngine

    original_store = LMCacheEngine.store

    @torch.inference_mode()
    def gather_first_store(
        self,
        tokens=None,
        hashes=None,
        offsets=None,
        mask=None,
        **kwargs,
    ):
        # This research hook is intentionally restricted to the normal P/D
        # token path.  Preserve upstream behavior for uncommon features whose
        # event semantics are outside this experiment.
        transfer_spec = kwargs.get("transfer_spec")
        if (
            transfer_spec is None
            or self.kv_events_enabled
            or hashes is not None
            or offsets is not None
            or not self.is_healthy()
            or self._is_passive()
            or self.is_frozen()
        ):
            return original_store(
                self,
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                mask=mask,
                **kwargs,
            )

        assert tokens is not None
        assert self.gpu_connector is not None
        assert self.storage_manager is not None
        num_to_store_tokens = (
            int(torch.sum(mask).item()) if mask is not None else len(tokens)
        )
        req_id = self._get_req_id(kwargs)
        seed_record = _claim_seed()
        store_started_ns = time.perf_counter_ns()
        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: list[int] = []
        ends: list[int] = []
        keys: list[Any] = []
        memory_objs: list[Any] = []
        total_bytes = 0
        total_tokens = 0
        request_configs = kwargs.get("request_configs")

        with store_stats.profile_process_tokens():
            for start, end, key in self.token_database.process_tokens(
                tokens,
                None,
                None,
                mask,
                request_configs=request_configs,
            ):
                num_tokens = end - start
                memory_obj = self.storage_manager.allocate(
                    self.metadata.get_shapes(num_tokens),
                    self.metadata.get_dtypes(),
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if memory_obj is None:
                    break
                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                total_bytes += memory_obj.get_size()
                total_tokens += num_tokens

        if not memory_objs:
            return None

        anchor_ids, residual_ids = partition_indices(
            len(memory_objs), fraction, mode
        )
        phases = [("anchor", anchor_ids)]
        if residual_ids:
            phases.append(("residual", residual_ids))

        phase_rows = []
        for phase_number, (phase, indexes) in enumerate(phases):
            phase_started_ns = time.perf_counter_ns()
            phase_objects = _subset(memory_objs, indexes)
            with store_stats.profile_from_gpu():
                self.gpu_connector.batched_from_gpu(
                    phase_objects,
                    _subset(starts, indexes),
                    _subset(ends, indexes),
                    **kwargs,
                )
            gather_done_ns = time.perf_counter_ns()
            is_last = phase_number == len(phases) - 1 and bool(
                getattr(transfer_spec, "is_last_prefill", False)
            )
            spec = _phase_spec(
                transfer_spec,
                phase=phase,
                is_last=is_last,
                total=len(memory_objs),
                request_id=req_id,
                seed_record=seed_record,
                indices=indexes,
            )
            with store_stats.profile_put():
                self.storage_manager.batched_put(
                    _subset(keys, indexes),
                    phase_objects,
                    transfer_spec=spec,
                    location=self.store_location,
                )
            submitted_ns = time.perf_counter_ns()
            phase_rows.append(
                {
                    "phase": phase,
                    "chunks": len(indexes),
                    "indices": indexes,
                    "bytes": sum(obj.get_size() for obj in phase_objects),
                    "started_ns": phase_started_ns,
                    "gather_done_ns": gather_done_ns,
                    "submitted_ns": submitted_ns,
                    "gather_ms": (gather_done_ns - phase_started_ns) / 1e6,
                    "submit_ms": (submitted_ns - gather_done_ns) / 1e6,
                }
            )

        self.stats_monitor.on_store_finished(store_stats, total_tokens)
        store_returned_ns = time.perf_counter_ns()
        if trace_path:
            _append_trace(
                trace_path,
                {
                    "event": "gather_submit",
                    "request_id": req_id,
                    "selection": mode,
                    "anchor_fraction_requested": fraction,
                    "total_chunks": len(memory_objs),
                    "total_bytes": total_bytes,
                    "store_started_ns": store_started_ns,
                    "store_returned_ns": store_returned_ns,
                    "store_ms": (store_returned_ns - store_started_ns) / 1e6,
                    "seed_record": seed_record,
                    "phases": phase_rows,
                },
            )
        return None

    LMCacheEngine.store = gather_first_store


def install() -> None:
    """Install opt-in progressive transfer without modifying LMCache sources."""

    global _INSTALLED
    if _INSTALLED:
        return
    fraction = float(os.environ["SPARSECACHE_ANCHOR_FRACTION"])
    mode = os.environ.get("SPARSECACHE_ANCHOR_MODE", "uniform")
    trace_path = os.environ.get("SPARSECACHE_ANCHOR_TRACE", "")
    notify_endpoint = os.environ.get("SPARSECACHE_ANCHOR_NOTIFY", "")
    gather_first = os.environ.get("SPARSECACHE_GATHER_FIRST") == "1"
    anchor_indices(1, fraction, mode)  # validate before monkey-patching

    from lmcache.v1.storage_backend.pd_backend_async import PDBackendAsync
    from lmcache.v1.transfer_channel.nixl_channel import NixlChannel

    original_task = PDBackendAsync._async_transfer_task
    original_write = NixlChannel.async_batched_write

    async def task_with_request_context(self, *args, **kwargs):
        transfer_spec = kwargs.get("transfer_spec")
        if transfer_spec is None and len(args) >= 5:
            transfer_spec = args[4]
        keys = kwargs.get("keys")
        if keys is None and args:
            keys = args[0]
        memory_objs = kwargs.get("memory_objs")
        if memory_objs is None and len(args) >= 2:
            memory_objs = args[1]
        key_strings = tuple(
            key.to_string() if hasattr(key, "to_string") else str(key)
            for key in (keys or ())
        )
        key_bytes = tuple(int(obj.get_size()) for obj in (memory_objs or ()))
        pd_request_id = getattr(transfer_spec, "req_id", "")
        request_id = getattr(
            transfer_spec,
            "_sparsecache_request_id",
            pd_request_id,
        )
        phase = getattr(transfer_spec, "_sparsecache_phase", "")
        seed_record = getattr(transfer_spec, "_sparsecache_seed_record", None)
        indices = getattr(transfer_spec, "_sparsecache_indices", ())
        request_token = _REQUEST_ID.set(request_id)
        pd_request_token = _PD_REQUEST_ID.set(pd_request_id)
        phase_token = _TRANSFER_PHASE.set(phase)
        seed_token = _SEED_RECORD.set(seed_record)
        keys_token = _TRANSFER_KEYS.set(key_strings)
        key_bytes_token = _TRANSFER_KEY_BYTES.set(key_bytes)
        indices_token = _TRANSFER_INDICES.set(indices)
        try:
            return await original_task(self, *args, **kwargs)
        finally:
            _TRANSFER_INDICES.reset(indices_token)
            _TRANSFER_KEY_BYTES.reset(key_bytes_token)
            _TRANSFER_KEYS.reset(keys_token)
            _SEED_RECORD.reset(seed_token)
            _TRANSFER_PHASE.reset(phase_token)
            _PD_REQUEST_ID.reset(pd_request_token)
            _REQUEST_ID.reset(request_token)

    async def anchor_first_write(self, objects, transfer_spec=None):
        if gather_first:
            started_ns = time.perf_counter_ns()
            result = await original_write(self, objects, transfer_spec)
            finished_ns = time.perf_counter_ns()
            total_bytes = sum(obj.get_size() for obj in objects)
            row = {
                "event": "nixl_write",
                "request_id": _REQUEST_ID.get(),
                "pd_request_id": _PD_REQUEST_ID.get(),
                "phase": _TRANSFER_PHASE.get(),
                "chunks": len(objects),
                "bytes": total_bytes,
                "keys": list(_TRANSFER_KEYS.get()),
                "chunk_indices": list(_TRANSFER_INDICES.get()),
                # ``bytes`` counts actual NIXL writes. ``resident_bytes`` also
                # includes deduplicated keys that were already on the receiver
                # and is therefore the correct mailbox-readiness invariant.
                "resident_bytes": sum(_TRANSFER_KEY_BYTES.get()),
                "remote_indexes": list(
                    (transfer_spec or {}).get("remote_indexes", ())
                ),
                "started_ns": started_ns,
                "finished_ns": finished_ns,
                "write_ms": (finished_ns - started_ns) / 1e6,
                "seed_record": _SEED_RECORD.get(),
            }
            if trace_path:
                _append_trace(trace_path, row)
            if row["phase"] == "anchor":
                _send_anchor_notification(notify_endpoint, row)
            return result
        if transfer_spec is None or len(objects) <= 1:
            return await original_write(self, objects, transfer_spec)
        selected = set(anchor_indices(len(objects), fraction, mode))
        remote_indexes = transfer_spec["remote_indexes"]
        anchors = [obj for index, obj in enumerate(objects) if index in selected]
        anchor_remote = [
            remote for index, remote in enumerate(remote_indexes) if index in selected
        ]
        residual = [obj for index, obj in enumerate(objects) if index not in selected]
        residual_remote = [
            remote for index, remote in enumerate(remote_indexes) if index not in selected
        ]
        started = time.perf_counter()
        await original_write(
            self,
            anchors,
            {**transfer_spec, "remote_indexes": anchor_remote},
        )
        anchor_done = time.perf_counter()
        if residual:
            await original_write(
                self,
                residual,
                {**transfer_spec, "remote_indexes": residual_remote},
            )
        finished = time.perf_counter()
        if trace_path:
            anchor_bytes = sum(obj.get_size() for obj in anchors)
            total_bytes = anchor_bytes + sum(obj.get_size() for obj in residual)
            _append_trace(
                trace_path,
                {
                    "event": "nixl_split_write",
                    "request_id": _REQUEST_ID.get(),
                    "chunks": len(objects),
                    "anchor_chunks": len(anchors),
                    "anchor_indices": sorted(selected),
                    "anchor_bytes": anchor_bytes,
                    "total_bytes": total_bytes,
                    "anchor_fraction_bytes": anchor_bytes / total_bytes,
                    "anchor_write_ms": (anchor_done - started) * 1000,
                    "residual_write_ms": (finished - anchor_done) * 1000,
                    "total_write_ms": (finished - started) * 1000,
                    "selection": mode,
                },
            )
        return len(objects)

    PDBackendAsync._async_transfer_task = task_with_request_context
    NixlChannel.async_batched_write = anchor_first_write
    if gather_first:
        _install_gather_first_store(
            fraction=fraction,
            mode=mode,
            trace_path=trace_path,
        )
    _INSTALLED = True
