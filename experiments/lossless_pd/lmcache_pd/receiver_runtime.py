"""Decoder-side AnchorReady mailbox for the real LMCache P/D path.

The sender emits a small UDP notification only after NIXL reports remote-write
completion.  This receiver hook resolves the advertised cache keys against the
decoder's actual ``PDBackendAsync.data`` map and records the GPU objects that
are available to a future sparse-KV drafter.  It does not alter LMCache's final
FullReady notification, retrieval, or target-model output.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRACE_LOCK = threading.Lock()
_INSTALLED = False


def parse_layers(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        raise ValueError("draft layers must be distinct non-negative integers")
    return layers


@dataclass
class ClaimedLayerViews:
    """Pinned anchor owners plus zero-copy per-layer tensor views."""

    objects: list[Any]
    chunk_indices: tuple[int, ...]
    token_ranges: tuple[tuple[int, int], ...]
    prompt_tokens: int
    layers: tuple[int, ...]
    views: dict[int, list[Any]]
    owners_by_layer: dict[int, list[Any]] | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        for obj in self.objects:
            obj.ref_count_down()
        self._released = True


def parse_udp_endpoint(endpoint: str) -> tuple[str, int]:
    if not endpoint.startswith("udp://") or ":" not in endpoint[6:]:
        raise ValueError("anchor listen endpoint must be udp://HOST:PORT")
    host, port = endpoint[6:].rsplit(":", 1)
    if not host or not port:
        raise ValueError("anchor listen endpoint must be udp://HOST:PORT")
    parsed_port = int(port)
    if not 0 < parsed_port < 65536:
        raise ValueError("anchor listen port must be in [1, 65535]")
    return host, parsed_port


def _append_trace(path: str, row: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK, destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def inspect_anchor_message(
    backend: Any,
    message: dict[str, Any],
    *,
    received_ns: int | None = None,
) -> dict[str, Any]:
    """Resolve one AnchorReady notification to decoder-resident KV objects."""

    if message.get("event") != "nixl_write" or message.get("phase") != "anchor":
        raise ValueError("expected an anchor nixl_write notification")
    if received_ns is None:
        received_ns = time.perf_counter_ns()
    requested_keys = list(message.get("keys") or ())
    with backend.data_lock:
        resident = {key.to_string(): obj for key, obj in backend.data.items()}
        objects = [(key, resident[key]) for key in requested_keys if key in resident]
    missing = [key for key in requested_keys if key not in resident]

    object_rows = []
    logical_bytes = 0
    physical_bytes = 0
    for key, obj in objects:
        size = int(obj.get_size())
        logical_bytes += size
        physical_size = int(obj.get_physical_size())
        physical_bytes += physical_size
        object_rows.append(
            {
                "key": key,
                "shape": list(obj.get_shape()),
                "dtype": str(obj.get_dtype()),
                "format": str(obj.get_memory_format()),
                "logical_bytes": size,
                "physical_bytes": physical_size,
                "address": int(obj.meta.address),
                "data_ptr": int(obj.data_ptr),
                "ref_count": int(obj.get_ref_count()),
            }
        )

    pd_request_id = str(message.get("pd_request_id", ""))
    tracked = list(getattr(backend, "_req_allocated_keys", {}).get(pd_request_id, ()))
    sender_finished_ns = int(message.get("finished_ns", 0))
    written_bytes = int(message.get("bytes", 0))
    expected_resident_bytes = int(message.get("resident_bytes", written_bytes))
    resolved_ns = time.perf_counter_ns()
    return {
        "event": "receiver_anchor_ready",
        "request_id": str(message.get("request_id", "")),
        "pd_request_id": pd_request_id,
        "sender_finished_ns": sender_finished_ns,
        "receiver_received_ns": received_ns,
        "receiver_resolved_ns": resolved_ns,
        "control_plane_ms": (
            (received_ns - sender_finished_ns) / 1e6 if sender_finished_ns else None
        ),
        "lookup_ms": (resolved_ns - received_ns) / 1e6,
        "expected_keys": len(requested_keys),
        "found_keys": len(objects),
        "missing_keys": missing,
        "written_bytes": written_bytes,
        "expected_resident_bytes": expected_resident_bytes,
        "resolved_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "pd_tracked_keys": len(tracked),
        "device": str(getattr(backend, "corrected_device", "")),
        "seed_record": message.get("seed_record"),
        "remote_indexes": list(message.get("remote_indexes") or ()),
        "chunk_indices": list(message.get("chunk_indices") or ()),
        "object_layers": list(message.get("object_layers") or ()),
        "object_chunk_indices": list(message.get("object_chunk_indices") or ()),
        "token_ranges": [
            [int(start), int(end)] for start, end in (message.get("token_ranges") or ())
        ],
        "prompt_tokens": int(message.get("prompt_tokens", 0)),
        "objects": object_rows,
        "complete": (
            bool(requested_keys)
            and len(objects) == len(requested_keys)
            and not missing
            and logical_bytes == expected_resident_bytes
        ),
    }


def inspect_layer_ready_message(
    backend: Any,
    message: dict[str, Any],
    *,
    received_ns: int | None = None,
) -> dict[str, Any]:
    """Validate one sender-side NIXL completion before making keys readable."""

    if message.get("event") != "layer_ready":
        raise ValueError("expected a layer_ready notification")
    layer = message.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("layer_ready notification has an invalid layer")
    if received_ns is None:
        received_ns = time.perf_counter_ns()
    requested_keys = [str(key) for key in message.get("keys") or ()]
    with backend.data_lock:
        resident = {key.to_string(): obj for key, obj in backend.data.items()}
        objects = [resident.get(key) for key in requested_keys]
    complete = bool(requested_keys) and all(obj is not None for obj in objects)
    resolved_bytes = sum(
        int(obj.get_size()) for obj in objects if obj is not None
    )
    expected_bytes = int(message.get("bytes", 0))
    return {
        **message,
        "event": "receiver_layer_ready",
        "receiver_received_ns": received_ns,
        "resolved_bytes": resolved_bytes,
        "missing_keys": [
            key for key, obj in zip(requested_keys, objects, strict=True) if obj is None
        ],
        "complete": complete and resolved_bytes == expected_bytes,
    }


class AnchorMailbox:
    """Thread-safe receiver readiness map with an explicit future claim API."""

    def __init__(
        self,
        backend: Any,
        endpoint: str,
        trace_path: str = "",
        lookup_timeout_ms: float = 50.0,
        draft_layers: tuple[int, ...] = (),
    ) -> None:
        self.backend = backend
        self.trace_path = trace_path
        self.lookup_timeout_ms = lookup_timeout_ms
        self.draft_layers = draft_layers
        from experiments.lossless_pd.lmcache_pd.online_drafter import (
            get_online_drafter,
        )

        self.online_drafter = get_online_drafter()
        self._condition = threading.Condition()
        self._records: dict[str, dict[str, Any]] = {}
        self._layer_records: dict[tuple[str, int], dict[str, Any]] = {}
        self.backend._sparsecache_ready_keys = set()
        self._running = True
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.2)
        self._socket.bind(parse_udp_endpoint(endpoint))
        self._thread = threading.Thread(
            target=self._listen,
            daemon=True,
            name="sparsecache-anchor-mailbox",
        )
        self._thread.start()

    def _listen(self) -> None:
        while self._running:
            try:
                payload, _ = self._socket.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            received_ns = time.perf_counter_ns()
            try:
                message = json.loads(payload)
                if message.get("event") == "layer_ready":
                    row = inspect_layer_ready_message(
                        self.backend, message, received_ns=received_ns
                    )
                    if row["complete"]:
                        with self._condition:
                            self.backend._sparsecache_ready_keys.update(row["keys"])
                            if row.get("layer_complete", False):
                                for identifier in (
                                    row.get("request_id"),
                                    row.get("pd_request_id"),
                                ):
                                    if identifier:
                                        self._layer_records[
                                            (str(identifier), int(row["layer"]))
                                        ] = row
                            self._condition.notify_all()
                    if self.trace_path:
                        _append_trace(self.trace_path, row)
                    continue
                deadline = time.perf_counter() + self.lookup_timeout_ms / 1000
                while True:
                    row = inspect_anchor_message(
                        self.backend, message, received_ns=received_ns
                    )
                    if row["complete"] or time.perf_counter() >= deadline:
                        break
                    time.sleep(0.001)
                if (
                    row["complete"]
                    and self.draft_layers
                    and self.online_drafter is None
                ):
                    view_started_ns = time.perf_counter_ns()
                    claim = self._claim_record_layer_views(row, self.draft_layers)
                    if claim is not None:
                        try:
                            view_bytes = sum(
                                view.numel() * view.element_size()
                                for views in claim.views.values()
                                for view in views
                            )
                            aliases = all(
                                view.untyped_storage().data_ptr()
                                == owner.tensor.untyped_storage().data_ptr()
                                for layer, views in claim.views.items()
                                for view, owner in zip(
                                    views,
                                    (claim.owners_by_layer or {})[layer],
                                    strict=True,
                                )
                            )
                        finally:
                            claim.release()
                        view_finished_ns = time.perf_counter_ns()
                        row["layer_views"] = {
                            "layers": list(claim.layers),
                            "chunks": len(claim.objects),
                            "views": sum(len(views) for views in claim.views.values()),
                            "logical_bytes": view_bytes,
                            "all_storage_aliases": aliases,
                            # Includes key resolution, ref-count pinning, tensor
                            # view construction, alias validation, and release.
                            "inspection_ms": (view_finished_ns - view_started_ns) / 1e6,
                        }
                self.publish(row)
                if row["complete"] and self.online_drafter is not None:
                    with self._condition:
                        self.backend._sparsecache_ready_keys.update(
                            str(key) for key in message.get("keys") or ()
                        )
                        self._condition.notify_all()
                    self.online_drafter.submit(self, row)
            # UDP is an external control plane. A malformed notification must
            # be traced without terminating the mailbox for subsequent work.
            except Exception as error:  # noqa: BLE001
                if self.trace_path:
                    _append_trace(
                        self.trace_path,
                        {
                            "event": "receiver_anchor_error",
                            "receiver_received_ns": received_ns,
                            "error": repr(error),
                        },
                    )

    def publish(self, row: dict[str, Any]) -> None:
        with self._condition:
            for identifier in (row.get("request_id"), row.get("pd_request_id")):
                if identifier:
                    self._records[str(identifier)] = row
            self._condition.notify_all()
        if self.trace_path:
            _append_trace(self.trace_path, row)

    def wait(
        self, request_id: str, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """Wait for an AnchorReady record without pinning or consuming its KV."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while request_id not in self._records:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._records[request_id]

    def claim_objects(self, request_id: str) -> list[Any]:
        """Pin and return the live decoder KV objects for a sparse draft task."""

        with self._condition:
            record = self._records.get(request_id)
        if record is None or not record["complete"]:
            return []
        return self._claim_record_objects(record)

    def wait_layer(
        self,
        request_id: str,
        layer: int,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Wait until all residual keys needed by one Target layer are written."""

        key = (request_id, layer)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while key not in self._layer_records:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._layer_records[key]

    def _claim_record_objects(self, record: dict[str, Any]) -> list[Any]:
        keys = [entry["key"] for entry in record["objects"]]
        with self.backend.data_lock:
            resident = {key.to_string(): obj for key, obj in self.backend.data.items()}
            claimed = [resident[key] for key in keys if key in resident]
            if len(claimed) == len(keys):
                for obj in claimed:
                    obj.ref_count_up()
        if len(claimed) != len(keys):
            return []
        return claimed

    def _claim_record_layer_views(
        self, record: dict[str, Any], layers: tuple[int, ...]
    ) -> ClaimedLayerViews | None:
        objects = self._claim_record_objects(record)
        if not objects:
            return None
        try:
            object_layers = tuple(int(value) for value in record.get("object_layers", ()))
            object_chunks = tuple(
                int(value) for value in record.get("object_chunk_indices", ())
            )
            if object_layers or object_chunks:
                if len(object_layers) != len(objects) or len(object_chunks) != len(objects):
                    raise ValueError("layerwise Anchor manifest is not object-aligned")
                selected_chunks = tuple(int(value) for value in record["chunk_indices"])
                views = {layer: [] for layer in layers}
                owners_by_layer = {layer: [] for layer in layers}
                by_pair = {
                    (layer, chunk): obj
                    for layer, chunk, obj in zip(
                        object_layers, object_chunks, objects, strict=True
                    )
                }
                for layer in layers:
                    for chunk in selected_chunks:
                        obj = by_pair.get((layer, chunk))
                        if obj is None:
                            raise ValueError(
                                f"Anchor is missing layer={layer}, chunk={chunk}"
                            )
                        tensor = obj.tensor
                        if tensor is None or tensor.ndim != 3 or tensor.shape[1] != 2:
                            raise ValueError(
                                "layerwise Anchor must expose KV_T2D [T,2,D]"
                            )
                        # permute/unsqueeze are metadata-only views.  Preserve the
                        # historical [2,1,T,D] drafter contract without a gather.
                        views[layer].append(tensor.permute(1, 0, 2).unsqueeze(1))
                        owners_by_layer[layer].append(obj)
                return ClaimedLayerViews(
                    objects,
                    selected_chunks,
                    tuple(
                        (int(start), int(end))
                        for start, end in record.get("token_ranges", ())
                    ),
                    int(record.get("prompt_tokens", 0)),
                    layers,
                    views,
                    owners_by_layer,
                )

            views = {layer: [] for layer in layers}
            for obj in objects:
                tensor = obj.tensor
                if tensor is None or tensor.ndim != 4:
                    raise ValueError("anchor object must expose a rank-4 KV tensor")
                if layers and max(layers) >= tensor.shape[1]:
                    raise ValueError("draft layer is outside the target KV layout")
                for layer in layers:
                    # Basic slicing preserves the LMCache allocation as storage;
                    # no gather/copy is performed on the decoder critical path.
                    views[layer].append(tensor[:, layer : layer + 1, :, :])
            chunk_indices = tuple(record.get("chunk_indices") or range(len(objects)))
            token_ranges = tuple(
                (int(start), int(end)) for start, end in record.get("token_ranges", ())
            )
            return ClaimedLayerViews(
                objects,
                chunk_indices,
                token_ranges,
                int(record.get("prompt_tokens", 0)),
                layers,
                views,
                {layer: list(objects) for layer in layers},
            )
        except BaseException:
            self.release_objects(objects)
            raise

    def claim_layer_views(
        self, request_id: str, layers: tuple[int, ...]
    ) -> ClaimedLayerViews | None:
        """Pin anchor owners and expose ordered, zero-copy target-layer views."""

        with self._condition:
            record = self._records.get(request_id)
        if record is None or not record["complete"]:
            return None
        return self._claim_record_layer_views(record, layers)

    @staticmethod
    def release_objects(objects: list[Any]) -> None:
        for obj in objects:
            obj.ref_count_down()

    def close(self) -> None:
        self._running = False
        self._socket.close()
        self._thread.join(timeout=1)
        if self.online_drafter is not None:
            self.online_drafter.close()


def install_receiver() -> None:
    """Attach one mailbox to each LMCache decoder receiver backend."""

    global _INSTALLED
    if _INSTALLED:
        return
    endpoint = os.environ["SPARSECACHE_ANCHOR_LISTEN"]
    trace_path = os.environ.get("SPARSECACHE_RECEIVER_TRACE", "")
    lookup_timeout_ms = float(
        os.environ.get("SPARSECACHE_RECEIVER_LOOKUP_TIMEOUT_MS", "50")
    )
    draft_layers = parse_layers(os.environ.get("SPARSECACHE_DRAFT_LAYERS", ""))
    parse_udp_endpoint(endpoint)

    from lmcache.v1.storage_backend.pd_backend_async import PDBackendAsync

    original_init = PDBackendAsync.__init__
    original_close = PDBackendAsync.close

    def init_with_mailbox(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.pd_config.role == "receiver":
            self._sparsecache_anchor_mailbox = AnchorMailbox(
                self,
                endpoint,
                trace_path,
                lookup_timeout_ms,
                draft_layers,
            )

    def close_with_mailbox(self):
        mailbox = getattr(self, "_sparsecache_anchor_mailbox", None)
        if mailbox is not None:
            mailbox.close()
        return original_close(self)

    PDBackendAsync.__init__ = init_with_mailbox
    PDBackendAsync.close = close_with_mailbox
    _INSTALLED = True
