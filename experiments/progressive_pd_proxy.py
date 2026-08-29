# SPDX-License-Identifier: Apache-2.0
"""One-prefiller/one-decoder proxy for SparseCache-PD experiments.

The prefiller produces and stores the exact prompt KV plus one exact greedy
seed token.  After the store-completion telemetry event, the decoder retrieves
that KV.  In ``progressive`` mode, the seed is passed to vLLM's internal
progressive sparse-draft state machine; in ``baseline`` mode, the decoder waits
for the complete retrieve and regenerates the same greedy seed normally.

This proxy deliberately supports one request at a time.  The current
completion snapshot is process-global, and concurrency is a later system gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

LOGGER = logging.getLogger("sparsecache.progressive_pd_proxy")


@dataclass(frozen=True)
class ProxyConfig:
    host: str
    port: int
    telemetry_port: int
    prefiller_url: str
    decoder_url: str
    default_mode: str
    default_visibility_mode: str
    max_draft_tokens: int
    start_fraction: float
    require_greedy: bool


CONFIG: ProxyConfig | None = None
MAIN_LOOP: asyncio.AbstractEventLoop | None = None
PENDING: dict[str, asyncio.Event] = {}
PENDING_LOCK = threading.Lock()
REQUEST_LOCK = asyncio.Lock()
PREFILL_CACHE: dict[str, "PrefillCacheEntry"] = {}


@dataclass(frozen=True)
class PrefillCacheEntry:
    """One producer result reused by every arm of a paired execution."""

    fingerprint: str
    seed_token_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--telemetry-port", type=int, default=5768)
    parser.add_argument("--prefiller-url", default="http://127.0.0.1:8100")
    parser.add_argument("--decoder-url", default="http://127.0.0.1:8200")
    parser.add_argument(
        "--default-mode", choices=("baseline", "progressive"), default="progressive"
    )
    parser.add_argument(
        "--default-visibility-mode",
        choices=("continuous", "fixed_s1"),
        default="continuous",
    )
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--start-fraction", type=float, default=0.05)
    parser.add_argument(
        "--require-greedy", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ProxyConfig:
    if args.max_draft_tokens < 1:
        raise ValueError("max_draft_tokens must be positive")
    if not 0.0 < args.start_fraction < 1.0:
        raise ValueError("start_fraction must lie in (0, 1)")
    return ProxyConfig(
        host=args.host,
        port=args.port,
        telemetry_port=args.telemetry_port,
        prefiller_url=args.prefiller_url.rstrip("/"),
        decoder_url=args.decoder_url.rstrip("/"),
        default_mode=args.default_mode,
        default_visibility_mode=args.default_visibility_mode,
        max_draft_tokens=args.max_draft_tokens,
        start_fraction=args.start_fraction,
        require_greedy=args.require_greedy,
    )


def _headers(request_id: str) -> dict[str, str]:
    return {"Content-Type": "application/json", "X-Request-Id": request_id}


def _prefill_fingerprint(endpoint: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"endpoint": endpoint, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _extract_seed_token(response_payload: dict[str, Any]) -> int:
    try:
        token_ids = response_payload["choices"][0]["token_ids"]
        seed = token_ids[0]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "prefiller response did not contain choices[0].token_ids[0]; "
            "the prefiller must support return_token_ids"
        ) from error
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("prefiller seed token ID is not an integer")
    return seed


def _prepare_requests(
    request_payload: dict[str, Any], config: ProxyConfig, seed_token_id: int | None
) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = dict(request_payload)
    mode = payload.pop("sparsecache_mode", config.default_mode)
    visibility_mode = payload.pop(
        "sparsecache_visibility_mode", config.default_visibility_mode
    )
    gamma = payload.pop("sparsecache_max_draft_tokens", config.max_draft_tokens)
    start_fraction = payload.pop("sparsecache_start_fraction", config.start_fraction)
    priority_chunks = payload.pop("sparsecache_priority_chunks", None)
    prefill_group = payload.pop("sparsecache_prefill_group", None)
    if mode not in {"baseline", "progressive"}:
        raise ValueError("sparsecache_mode must be baseline or progressive")
    if visibility_mode not in {"continuous", "fixed_s1"}:
        raise ValueError("sparsecache_visibility_mode must be continuous or fixed_s1")
    if not isinstance(gamma, int) or isinstance(gamma, bool) or gamma < 1:
        raise ValueError("sparsecache_max_draft_tokens must be a positive integer")
    if (
        not isinstance(start_fraction, int | float)
        or not 0.0 < float(start_fraction) < 1.0
    ):
        raise ValueError("sparsecache_start_fraction must lie in (0, 1)")
    if priority_chunks is not None and (
        not isinstance(priority_chunks, list)
        or not priority_chunks
        or any(
            not isinstance(chunk_id, int) or isinstance(chunk_id, bool) or chunk_id < 0
            for chunk_id in priority_chunks
        )
        or len(priority_chunks) != len(set(priority_chunks))
    ):
        raise ValueError(
            "sparsecache_priority_chunks must be unique non-negative integers"
        )
    if prefill_group is not None and (
        not isinstance(prefill_group, str) or not prefill_group.strip()
    ):
        raise ValueError("sparsecache_prefill_group must be a non-empty string")
    temperature = payload.get("temperature", 1.0)
    if config.require_greedy and float(temperature) != 0.0:
        raise ValueError("the correctness gate requires temperature=0")

    prefiller = dict(payload)
    prefiller["max_tokens"] = 1
    if "max_completion_tokens" in prefiller:
        prefiller["max_completion_tokens"] = 1
    prefiller["stream"] = False
    prefiller.pop("stream_options", None)
    prefiller["return_token_ids"] = True

    decoder = dict(payload)
    transfer_params = dict(decoder.get("kv_transfer_params") or {})
    if "require_full_remote_kv" in transfer_params:
        raise ValueError("kv_transfer_params.require_full_remote_kv is proxy-owned")
    transfer_params["require_full_remote_kv"] = True
    if mode == "progressive":
        if seed_token_id is None:
            raise ValueError("progressive mode requires the exact prefiller seed")
        if "progressive_sparse_draft" in transfer_params:
            raise ValueError(
                "kv_transfer_params.progressive_sparse_draft is proxy-owned"
            )
        transfer_params["progressive_sparse_draft"] = {
            "seed_token_id": seed_token_id,
            "max_draft_tokens": gamma,
            "start_fraction": float(start_fraction),
            "visibility_mode": visibility_mode,
        }
        if priority_chunks is not None:
            transfer_params["progressive_priority_chunks"] = priority_chunks
        decoder["kv_transfer_params"] = transfer_params
    elif seed_token_id is not None:
        if "producer_seed_token_id" in transfer_params:
            raise ValueError("kv_transfer_params.producer_seed_token_id is proxy-owned")
        transfer_params["producer_seed_token_id"] = seed_token_id
        decoder["kv_transfer_params"] = transfer_params
    return prefiller, decoder, mode


def _create_pending(request_id: str) -> asyncio.Event:
    event = asyncio.Event()
    with PENDING_LOCK:
        PENDING[request_id] = event
    return event


def _notify(raw_request_id: str) -> bool:
    with PENDING_LOCK:
        request_id = next(
            (
                candidate
                for candidate in PENDING
                if raw_request_id == candidate
                or raw_request_id.startswith(
                    (f"chatcmpl-{candidate}", f"cmpl-{candidate}")
                )
            ),
            None,
        )
        event = PENDING.get(request_id) if request_id is not None else None
    if event is None or MAIN_LOOP is None:
        return False
    MAIN_LOOP.call_soon_threadsafe(event.set)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MAIN_LOOP
    assert CONFIG is not None
    MAIN_LOOP = asyncio.get_running_loop()
    # These endpoints are launcher-owned loopback services. Inheriting a host
    # ALL_PROXY/HTTP_PROXY can route 127.0.0.1 through an unavailable SOCKS
    # transport and makes an otherwise self-contained deployment fail.
    app.state.prefiller = httpx.AsyncClient(
        base_url=CONFIG.prefiller_url, timeout=None, trust_env=False
    )
    app.state.decoder = httpx.AsyncClient(
        base_url=CONFIG.decoder_url, timeout=None, trust_env=False
    )
    yield
    await app.state.prefiller.aclose()
    await app.state.decoder.aclose()


APP = FastAPI(lifespan=lifespan)
TELEMETRY_APP = FastAPI()


async def _forward(endpoint: str, request: Request):
    assert CONFIG is not None
    await REQUEST_LOCK.acquire()
    release_after_stream = False
    try:
        received_at = time.perf_counter()
        try:
            request_payload = await request.json()
            if not isinstance(request_payload, dict):
                raise TypeError("request body must be a JSON object")
            # Validate proxy-owned controls before spending a prefill.
            prefiller_payload, _, mode = _prepare_requests(
                request_payload, CONFIG, seed_token_id=0
            )
            prefill_group = request_payload.get("sparsecache_prefill_group")
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        request_id = uuid.uuid4().hex[:16]
        prefill_fingerprint = _prefill_fingerprint(endpoint, prefiller_payload)
        cached = PREFILL_CACHE.get(prefill_group) if prefill_group is not None else None
        if cached is not None and cached.fingerprint != prefill_fingerprint:
            raise HTTPException(
                status_code=409,
                detail="sparsecache_prefill_group was reused for a different prompt",
            )
        prefill_started = time.perf_counter()
        if cached is not None:
            seed_token_id = cached.seed_token_id
            prefill_responded = prefill_started
            store_ready = prefill_started
            prefill_reused = True
        else:
            store_event = _create_pending(request_id)
            try:
                prefill_response = await APP.state.prefiller.post(
                    endpoint,
                    json=prefiller_payload,
                    headers=_headers(request_id),
                )
                prefill_response.raise_for_status()
                prefill_responded = time.perf_counter()
                seed_token_id = _extract_seed_token(prefill_response.json())
                await store_event.wait()
                store_ready = time.perf_counter()
            except (httpx.HTTPError, RuntimeError, TypeError) as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
            finally:
                with PENDING_LOCK:
                    PENDING.pop(request_id, None)
            prefill_reused = False
            if prefill_group is not None:
                PREFILL_CACHE[prefill_group] = PrefillCacheEntry(
                    fingerprint=prefill_fingerprint,
                    seed_token_id=seed_token_id,
                )

        try:
            _, decoder_payload, mode = _prepare_requests(
                request_payload, CONFIG, seed_token_id=seed_token_id
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        decode_started = time.perf_counter()
        common_headers = {
            "X-SparseCache-Mode": mode,
            "X-SparseCache-Request-Id": request_id,
            "X-SparseCache-Seed-Token": str(seed_token_id),
            "X-SparseCache-Prefill-Reused": str(prefill_reused).lower(),
            "X-SparseCache-Prefill-Group": prefill_group or "",
            "Server-Timing": (
                f"prefill;dur={(prefill_responded - prefill_started) * 1000:.3f}, "
                f"store_wait;dur={(store_ready - prefill_responded) * 1000:.3f}"
            ),
        }
        LOGGER.info(
            "request=%s mode=%s prefill_reused=%s prefill_ms=%.3f "
            "store_wait_ms=%.3f seed=%d",
            request_id,
            mode,
            prefill_reused,
            (prefill_responded - prefill_started) * 1000,
            (store_ready - prefill_responded) * 1000,
            seed_token_id,
        )

        if decoder_payload.get("stream", False):

            async def stream():
                first = True
                async with APP.state.decoder.stream(
                    "POST",
                    endpoint,
                    json=decoder_payload,
                    headers=_headers(request_id),
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        if first:
                            LOGGER.info(
                                "request=%s mode=%s decode_first_byte_ms=%.3f "
                                "e2e_first_byte_ms=%.3f",
                                request_id,
                                mode,
                                (time.perf_counter() - decode_started) * 1000,
                                (time.perf_counter() - received_at) * 1000,
                            )
                            first = False
                        yield chunk

            release_after_stream = True
            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers=common_headers,
                background=BackgroundTask(REQUEST_LOCK.release),
            )

        try:
            response = await APP.state.decoder.post(
                endpoint,
                json=decoder_payload,
                headers=_headers(request_id),
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        completed_at = time.perf_counter()
        LOGGER.info(
            "request=%s mode=%s decode_e2e_ms=%.3f total_ms=%.3f",
            request_id,
            mode,
            (completed_at - decode_started) * 1000,
            (completed_at - received_at) * 1000,
        )
        common_headers["Server-Timing"] += (
            f", decode;dur={(completed_at - decode_started) * 1000:.3f}"
        )
        return JSONResponse(
            response.json(), status_code=response.status_code, headers=common_headers
        )
    finally:
        if not release_after_stream:
            REQUEST_LOCK.release()


@APP.post("/v1/completions")
async def completions(request: Request):
    return await _forward("/v1/completions", request)


@APP.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward("/v1/chat/completions", request)


@APP.get("/v1/models")
async def models():
    response = await APP.state.prefiller.get("/v1/models")
    return JSONResponse(response.json(), status_code=response.status_code)


@TELEMETRY_APP.post("/api/v1/telemetry")
async def telemetry(request: Request):
    payload = await request.json()
    request_ids = payload.get("request_ids_set", [])
    notified = sum(_notify(str(request_id)) for request_id in request_ids)
    return {"status": "ok", "notified": notified, "total": len(request_ids)}


def _run_telemetry(host: str, port: int) -> None:
    uvicorn.run(TELEMETRY_APP, host=host, port=port, log_level="info")


def main() -> None:
    global CONFIG
    logging.basicConfig(level=logging.INFO)
    CONFIG = build_config(parse_args())
    thread = threading.Thread(
        target=_run_telemetry,
        args=(CONFIG.host, CONFIG.telemetry_port),
        daemon=True,
    )
    thread.start()
    uvicorn.run(APP, host=CONFIG.host, port=CONFIG.port, log_level="info")


if __name__ == "__main__":
    main()
