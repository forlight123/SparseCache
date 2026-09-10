"""CPU proxy that dispatches D before FullReady without exposing draft tokens.

P still supplies the first authoritative Target token.  D is submitted only
after its local sparse proposal is sealed and layers 0..k are NIXL-complete.
The response contains P's exact token followed by ordinary vLLM output; custom
proposals never cross the client boundary before Target verification.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from experiments.lossless_pd.lmcache_pd.direct_token_proxy import (
    _TokenizeResponse,
    is_token_id_prompt,
)


def external_draft_request(
    template: dict[str, Any],
    prompt_ids: list[int],
    *,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Build one deterministic sidecar request without Target transfer fields."""

    request = {
        key: value
        for key, value in template.items()
        if key
        not in {
            "kv_transfer_params",
            "stream_options",
            "return_token_ids",
        }
    }
    request.update(
        {
            "model": model,
            "prompt": list(prompt_ids),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "return_token_ids": True,
            "ignore_eos": True,
        }
    )
    return request


def external_draft_payload(
    *,
    pd_request_id: str,
    prompt_tokens: int,
    seed_token_id: int,
    proposals: list[int],
    started_ns: int,
    finished_ns: int,
    model_ms: float,
) -> dict[str, Any]:
    if not proposals:
        raise ValueError("external draft returned no proposal tokens")
    return {
        "event": "external_draft",
        "request_id": pd_request_id,
        "pd_request_id": pd_request_id,
        "prompt_tokens": prompt_tokens,
        "seed_token_id": seed_token_id,
        "proposals": proposals,
        "draft_started_ns": started_ns,
        "draft_finished_ns": finished_ns,
        "model_ms": model_ms,
        "wall_ms": (finished_ns - started_ns) / 1e6,
        "source": "external_small_model",
    }


def external_token_ids(response: dict[str, Any]) -> list[int]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError("external drafter requires one completion")
    tokens = choices[0].get("token_ids")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
    ):
        raise RuntimeError("external drafter did not return integer token IDs")
    return tokens


def matched_external_suffix(
    prefill_tokens: list[int], *, seed_token_id: int, max_tokens: int
) -> list[int] | None:
    """Reuse the speculative branch iff its root equals the exact P seed."""

    if max_tokens <= 0:
        raise ValueError("external suffix horizon must be positive")
    if not prefill_tokens or prefill_tokens[0] != seed_token_id:
        return None
    suffix = prefill_tokens[1 : 1 + max_tokens]
    return suffix or None


def parse_udp_endpoint(endpoint: str) -> tuple[str, int]:
    if not endpoint.startswith("udp://") or ":" not in endpoint[6:]:
        raise ValueError("draft endpoint must be udp://HOST:PORT")
    host, raw_port = endpoint[6:].rsplit(":", 1)
    port = int(raw_port)
    if not host or not 0 < port < 65536:
        raise ValueError("invalid draft-ready endpoint")
    return host, port


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def append_trace(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


class DraftReadyMailbox:
    """Small fail-closed UDP control mailbox keyed by both request IDs."""

    def __init__(self, endpoint: str) -> None:
        self._condition = threading.Condition()
        self._records: dict[str, dict[str, Any]] = {}
        self._running = True
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.2)
        self._socket.bind(parse_udp_endpoint(endpoint))
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self) -> None:
        while self._running:
            try:
                payload, _ = self._socket.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                row = json.loads(payload)
                if row.get("event") != "draft_ready":
                    continue
                with self._condition:
                    for identifier in (row.get("request_id"), row.get("pd_request_id")):
                        if identifier:
                            self._records[str(identifier)] = row
                    self._condition.notify_all()
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

    def wait(self, request_id: str, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while request_id not in self._records:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._records.pop(request_id)

    def close(self) -> None:
        self._running = False
        self._socket.close()
        self._thread.join(timeout=1)


def prepare_decoder_request(
    prefill_request: dict[str, Any],
    prefill_output: dict[str, Any],
    *,
    original_max_tokens: int,
    pd_request_id: str,
    prompt_tokens: int,
) -> tuple[dict[str, Any], int]:
    choices = prefill_output.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError("layer-ready P/D requires one P completion")
    exact_ids = choices[0].get("token_ids")
    if (
        not isinstance(exact_ids, list)
        or len(exact_ids) != 1
        or isinstance(exact_ids[0], bool)
        or not isinstance(exact_ids[0], int)
    ):
        raise RuntimeError("P did not return exactly one authoritative token ID")
    request = dict(prefill_request)
    request["prompt"] = list(prefill_request["prompt"]) + exact_ids
    request["max_tokens"] = original_max_tokens - 1
    request["stream"] = True
    request["return_token_ids"] = True
    request["kv_transfer_params"] = {
        "sparsecache_progressive": {
            "request_id": pd_request_id,
            "prompt_tokens": prompt_tokens,
        }
    }
    return request, exact_ids[0]


def _remove_upstream_completion_route(upstream: Any) -> None:
    upstream.app.router.routes[:] = [
        route
        for route in upstream.app.router.routes
        if not (
            getattr(route, "path", None) == "/v1/completions"
            and "POST" in getattr(route, "methods", set())
        )
    ]


def _install_cancellable_zmq_proxy(upstream: Any) -> None:
    """Give the LMCache example proxy a bounded shutdown poll interval.

    The upstream coroutine awaits ``socket.recv()`` indefinitely and only
    checks its shutdown flag after a message arrives.  Fresh-server sandwich
    arms would therefore leave the control port and process alive.  This local
    equivalent preserves its message handling while polling every 200 ms.
    """

    async def polling_zmq_pull_server() -> None:
        channel = upstream.zmq_ctx.socket(upstream.zmq.PULL)
        proxy_url = f"{upstream.global_args.proxy_host}:{upstream.global_args.proxy_port}"
        try:
            channel.bind(f"tcp://{proxy_url}")
        except upstream.zmq.ZMQError:
            upstream.logger.exception(
                "ZMQ proxy server failed to bind on %s", proxy_url
            )
            return
        upstream.logger.info("ZMQ proxy server started on %s", proxy_url)
        try:
            while upstream.run_proxy:
                try:
                    message_bytes = await asyncio.wait_for(
                        channel.recv(), timeout=0.2
                    )
                except TimeoutError:
                    continue
                except upstream.zmq.ZMQError as error:
                    if error.errno in (upstream.zmq.ETERM, upstream.zmq.ENOTSOCK):
                        break
                    upstream.logger.warning("ZMQ recv error: %s", error)
                    await asyncio.sleep(0.05)
                    continue
                try:
                    message = upstream.msgspec.msgpack.decode(
                        message_bytes, type=upstream.PDMsg
                    )
                except upstream.msgspec.DecodeError as error:
                    upstream.logger.warning("ZMQ received non-PD message: %s", error)
                    continue
                except Exception:
                    upstream.logger.exception("ZMQ message decode failed")
                    continue
                if not isinstance(message, upstream.ProxyNotif):
                    continue
                upstream.app.state.finished_reqs[message.req_id] += 1
        finally:
            channel.close()
            upstream.logger.info("ZMQ PULL server stopped.")

    upstream.zmq_pull_server = polling_zmq_pull_server


def main() -> None:
    try:
        from examples.disagg_prefill import disagg_proxy_server as upstream
        from fastapi import Request
        from fastapi.responses import StreamingResponse
    except ImportError as error:
        raise RuntimeError(
            "run with the adjacent LMCache environment and repository on PYTHONPATH"
        ) from error

    globals()["Request"] = Request
    endpoint = os.environ.get("SPARSECACHE_DRAFT_LISTEN", "udp://127.0.0.1:17610")
    timeout = float(os.environ.get("SPARSECACHE_DRAFT_READY_TIMEOUT_SEC", "30"))
    early_dispatch = parse_bool(
        os.environ.get("SPARSECACHE_EARLY_DISPATCH", "true")
    )
    trace_path = os.environ.get("SPARSECACHE_PROXY_TRACE", "")
    external_url = os.environ.get("SPARSECACHE_EXTERNAL_DRAFT_URL", "").rstrip("/")
    external_model = os.environ.get(
        "SPARSECACHE_EXTERNAL_DRAFT_MODEL", "/data/models/qwen/Qwen3-4B"
    )
    external_tokens = int(os.environ.get("SPARSECACHE_EXTERNAL_DRAFT_TOKENS", "8"))
    external_notify = os.environ.get("SPARSECACHE_EXTERNAL_DRAFT_NOTIFY", "")
    if external_tokens <= 0:
        raise ValueError("external draft horizon must be positive")
    if bool(external_url) != bool(external_notify):
        raise ValueError("external draft URL and notify endpoint must be set together")
    if external_notify:
        parse_udp_endpoint(external_notify)
    mailbox = DraftReadyMailbox(endpoint)
    original_send = upstream.send_request_to_service
    external_client = None
    if external_url:
        import httpx

        external_client = httpx.AsyncClient(
            base_url=external_url,
            timeout=float(os.environ.get("SPARSECACHE_EXTERNAL_DRAFT_TIMEOUT_SEC", "30")),
            trust_env=False,
        )

    # LMCache's current example proxy stores AsyncClient inside ClientInfo but
    # calls ``ClientInfo.aclose()`` during shutdown.  Supply that forwarding
    # method locally so repeated fresh-server sandwich arms terminate cleanly.
    if not hasattr(upstream.ClientInfo, "aclose"):

        async def close_client_info(value) -> None:
            await value.client.aclose()

        upstream.ClientInfo.aclose = close_client_info

    # The sidecar client is first used on uvicorn's loop and must be closed on
    # that same loop.  Wrapping the existing LMCache lifespan avoids attempting
    # to close an httpcore transport from a new asyncio.run() loop afterwards.
    original_lifespan = upstream.app.router.lifespan_context

    @asynccontextmanager
    async def sparsecache_lifespan(app):
        try:
            async with original_lifespan(app):
                yield
        finally:
            if external_client is not None:
                await external_client.aclose()

    upstream.app.router.lifespan_context = sparsecache_lifespan

    async def send_external(request_data: dict[str, Any]) -> dict[str, Any]:
        if external_client is None:
            raise RuntimeError("external draft client is disabled")
        response = await external_client.post("/v1/completions", json=request_data)
        response.raise_for_status()
        return response.json()

    async def send_external_timed(
        request_data: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        response = await send_external(request_data)
        return response, time.perf_counter_ns()

    async def preserve_token_ids(client, endpoint: str, request: dict):
        prompt = request.get("prompt")
        if endpoint == "/tokenize" and is_token_id_prompt(prompt):
            return _TokenizeResponse(list(prompt))
        return await original_send(client, endpoint, request)

    upstream.send_request_to_service = preserve_token_ids
    _install_cancellable_zmq_proxy(upstream)
    _remove_upstream_completion_route(upstream)
    upstream.counter = time.time_ns()

    @upstream.app.post("/v1/completions")
    async def handle_layer_ready(request: Request):
        upstream.counter += 1
        pd_request_id = str(upstream.counter)
        slots = 0
        acquired = False
        full_ready_task: asyncio.Task | None = None
        external_prefill_task: asyncio.Task | None = None
        try:
            request_data = await request.json()
            request_received_ns = time.perf_counter_ns()
            tokenizer, prefiller, decoder = upstream.pick_up_clients(request)
            tokenized = await upstream.send_request_to_service(
                tokenizer.client, "/tokenize", {"prompt": request_data["prompt"]}
            )
            prompt_ids = list(tokenized.json()["tokens"])
            original_max_tokens = int(request_data["max_tokens"])
            if original_max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            external_template = dict(request_data)
            external_started_ns = time.perf_counter_ns()
            if external_client is not None:
                # Generate one seed branch in parallel with P.  If its root
                # equals P's exact seed the suffix is already complete; a root
                # mismatch falls back to one prefix-cached conditional call.
                external_prefill_task = asyncio.create_task(
                    send_external_timed(
                        external_draft_request(
                            external_template,
                            prompt_ids,
                            model=external_model,
                            max_tokens=external_tokens + 1,
                        )
                    )
                )
            request_data["prompt"] = prompt_ids
            request_data["max_tokens"] = 1
            request_data["return_token_ids"] = True
            request_data["stream"] = False
            stream_options = request_data.pop("stream_options", None)

            slots = math.ceil(len(prompt_ids) / upstream.global_args.chunk_size)
            if upstream.pd_buffer_semaphore is not None:
                await upstream.pd_buffer_semaphore.acquire(slots)
                acquired = True

            disagg_spec = {
                "req_id": pd_request_id,
                "receiver_host": decoder.host,
                "receiver_init_port": decoder.init_port,
                "receiver_alloc_port": decoder.alloc_port,
            }
            request_data["kv_transfer_params"] = {
                "ret_first_tok": True,
                "disagg_spec": disagg_spec,
            }
            prefill_response = await upstream.send_request_to_service(
                prefiller.client, "/v1/completions", request_data
            )
            prefill_returned_ns = time.perf_counter_ns()
            prefill_output = prefill_response.json()
            decoder_request, first_token = prepare_decoder_request(
                request_data,
                prefill_output,
                original_max_tokens=original_max_tokens,
                pd_request_id=pd_request_id,
                prompt_tokens=len(prompt_ids),
            )
            if stream_options is not None:
                decoder_request["stream_options"] = stream_options
            num_tp_rank = len(decoder.init_port or [])
            full_ready_task = asyncio.create_task(
                upstream.wait_decode_kv_ready(pd_request_id, num_tp_rank)
            )

            async def produce_external_draft() -> bool:
                if external_prefill_task is None:
                    return False
                try:
                    prefill_response, concurrent_finished_ns = (
                        await external_prefill_task
                    )
                    prefill_tokens = external_token_ids(prefill_response)
                    proposals = matched_external_suffix(
                        prefill_tokens,
                        seed_token_id=first_token,
                        max_tokens=external_tokens,
                    )
                    seed_branch_hit = proposals is not None
                    concurrent_ms = (
                        concurrent_finished_ns - external_started_ns
                    ) / 1e6
                    fallback_ms = 0.0
                    if proposals is None:
                        fallback_started_ns = time.perf_counter_ns()
                        response, finished_ns = await send_external_timed(
                            external_draft_request(
                                external_template,
                                [*prompt_ids, first_token],
                                model=external_model,
                                max_tokens=external_tokens,
                            )
                        )
                        proposals = external_token_ids(response)[:external_tokens]
                        fallback_ms = (finished_ns - fallback_started_ns) / 1e6
                    else:
                        finished_ns = concurrent_finished_ns
                    payload = external_draft_payload(
                        pd_request_id=pd_request_id,
                        prompt_tokens=len(prompt_ids),
                        seed_token_id=first_token,
                        proposals=proposals,
                        started_ns=external_started_ns,
                        finished_ns=finished_ns,
                        model_ms=concurrent_ms + fallback_ms,
                    )
                    from experiments.lossless_pd.lmcache_pd.anchor_runtime import (
                        _send_anchor_notification,
                    )

                    _send_anchor_notification(external_notify, payload)
                    append_trace(
                        trace_path,
                        {
                            **payload,
                            "event": "external_draft_submitted",
                            "prefill_overlap_ms": (
                                min(prefill_returned_ns, concurrent_finished_ns)
                                - external_started_ns
                            )
                            / 1e6,
                            "concurrent_branch_ms": concurrent_ms,
                            "fallback_branch_ms": fallback_ms,
                            "seed_branch_hit": seed_branch_hit,
                        },
                    )
                    return True
                except Exception as error:  # noqa: BLE001
                    append_trace(
                        trace_path,
                        {
                            "event": "external_draft_error",
                            "pd_request_id": pd_request_id,
                            "error": repr(error),
                            "finished_ns": time.perf_counter_ns(),
                        },
                    )
                    return False

            external_draft_task = (
                asyncio.create_task(produce_external_draft())
                if external_prefill_task is not None
                else None
            )

            async def generate_stream():
                nonlocal acquired
                draft_ready = None
                try:
                    head = {
                        "id": prefill_output["id"],
                        "object": "text_completion",
                        "created": prefill_output["created"],
                        "model": prefill_output["model"],
                        "choices": [
                            {
                                "index": 0,
                                "text": prefill_output["choices"][0]["text"],
                                "logprobs": None,
                                "finish_reason": None,
                                "stop_reason": None,
                                "token_ids": [first_token],
                            }
                        ],
                        "usage": None,
                    }
                    yield (
                        "data: "
                        + json.dumps(head, separators=(",", ":"))
                        + "\n\n"
                    ).encode()

                    external_submitted = None
                    if external_draft_task is not None:
                        external_submitted = await external_draft_task
                    if early_dispatch and external_submitted is not False:
                        draft_ready = await asyncio.to_thread(
                            mailbox.wait, pd_request_id, timeout
                        )
                    if not early_dispatch or draft_ready is None:
                        # Full-ready is both the control condition and the safe
                        # fallback when a sparse draft or layer gate times out.
                        await full_ready_task
                        decoder_request.pop("kv_transfer_params", None)
                    dispatch_ns = time.perf_counter_ns()
                    append_trace(
                        trace_path,
                        {
                            "event": "decoder_dispatch",
                            "pd_request_id": pd_request_id,
                            "prompt_tokens": len(prompt_ids),
                            "early_dispatch": early_dispatch,
                            "draft_ready": draft_ready is not None,
                            "full_ready_at_dispatch": full_ready_task.done(),
                            "request_received_ns": request_received_ns,
                            "prefill_returned_ns": prefill_returned_ns,
                            "dispatch_ns": dispatch_ns,
                        },
                    )
                    async for chunk in upstream.stream_service_response(
                        decoder.client, "/v1/completions", decoder_request
                    ):
                        yield chunk
                    if not full_ready_task.done():
                        await full_ready_task
                    append_trace(
                        trace_path,
                        {
                            "event": "decoder_complete",
                            "pd_request_id": pd_request_id,
                            "finished_ns": time.perf_counter_ns(),
                        },
                    )
                finally:
                    if upstream.pd_buffer_semaphore is not None and acquired:
                        await upstream.pd_buffer_semaphore.release(slots)
                        acquired = False

            return StreamingResponse(generate_stream(), media_type="application/json")
        except Exception:
            if external_prefill_task is not None and not external_prefill_task.done():
                external_prefill_task.cancel()
            if full_ready_task is not None and not full_ready_task.done():
                full_ready_task.cancel()
            if upstream.pd_buffer_semaphore is not None and acquired:
                await upstream.pd_buffer_semaphore.release(slots)
            raise

    upstream.global_args = upstream.parse_args()

    import uvicorn

    try:
        uvicorn.run(
            upstream.app,
            host=upstream.global_args.host,
            port=upstream.global_args.port,
        )
    finally:
        mailbox.close()


if __name__ == "__main__":
    main()
