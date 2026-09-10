"""Experimental LMCache proxy for an exact P-side runahead baseline.

The upstream example asks P for exactly one token and then sends the remaining
request to D.  This wrapper asks P for a bounded exact block, returns that block
to the client, appends its token IDs to D's prompt, and lets D finish the
request.  The path is a baseline for progressive sparse drafting, not the
SparseCache treatment itself.

The implementation is intentionally fail-closed: P must return exact token IDs
through vLLM's ``return_token_ids`` extension.  Text retokenization is never
used for the handoff.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from experiments.lossless_pd.lmcache_pd.direct_token_proxy import (
    _TokenizeResponse,
    is_token_id_prompt,
)


def runahead_budget(max_tokens: int, requested: int) -> int:
    if max_tokens <= 0 or requested <= 0:
        raise ValueError("token budgets must be positive")
    return min(max_tokens, requested)


def prepare_decoder_request(
    prefill_request: dict[str, Any],
    prefill_output: dict[str, Any],
    *,
    original_max_tokens: int,
) -> tuple[dict[str, Any], list[int]]:
    choices = prefill_output.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError("P runahead requires exactly one completion choice")
    token_ids = choices[0].get("token_ids")
    if not token_ids or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in token_ids
    ):
        raise RuntimeError("P did not return exact generated token IDs")
    if len(token_ids) > original_max_tokens:
        raise RuntimeError("P returned more tokens than the client requested")

    request = dict(prefill_request)
    request["prompt"] = list(prefill_request["prompt"]) + list(token_ids)
    stopped = choices[0].get("finish_reason") == "stop"
    request["max_tokens"] = 0 if stopped else original_max_tokens - len(token_ids)
    request.pop("kv_transfer_params", None)
    # Keep exact IDs on the D response as well, so the benchmark can compare
    # the complete P+D trajectory without relying on decoded text equality.
    request["return_token_ids"] = True
    request["stream"] = True
    return request, list(token_ids)


def _remove_upstream_completion_route(upstream) -> None:
    upstream.app.router.routes[:] = [
        route
        for route in upstream.app.router.routes
        if not (
            getattr(route, "path", None) == "/v1/completions"
            and "POST" in getattr(route, "methods", set())
        )
    ]


def main() -> None:
    try:
        from examples.disagg_prefill import disagg_proxy_server as upstream
        from fastapi import Request
        from fastapi.responses import StreamingResponse
    except ImportError as error:
        raise RuntimeError(
            "run with the adjacent LMCache environment and repository on PYTHONPATH"
        ) from error

    # This module postpones annotations, while FastAPI resolves endpoint types
    # from module globals at route-registration time.  Keep the heavy optional
    # dependency local but publish the resolved class for that inspection.
    globals()["Request"] = Request

    requested = int(os.environ.get("SPARSECACHE_P_RUNAHEAD_TOKENS", "4"))
    if requested <= 0:
        raise ValueError("SPARSECACHE_P_RUNAHEAD_TOKENS must be positive")

    original_send = upstream.send_request_to_service

    async def preserve_token_ids(client, endpoint: str, request: dict):
        prompt = request.get("prompt")
        if endpoint == "/tokenize" and is_token_id_prompt(prompt):
            return _TokenizeResponse(list(prompt))
        return await original_send(client, endpoint, request)

    upstream.send_request_to_service = preserve_token_ids
    _remove_upstream_completion_route(upstream)
    # The upstream example resets its integer counter whenever the CPU proxy
    # restarts.  P/D engines can outlive that process, so a reset can collide
    # with an in-flight or retained transfer notification and wait forever.
    upstream.counter = time.time_ns()

    @upstream.app.post("/v1/completions")
    async def handle_runahead_completions(request: Request):
        upstream.counter += 1
        request_id = str(upstream.counter)
        slots = 0
        acquired = False
        try:
            request_data = await request.json()
            request_budget = int(
                request_data.pop("sparsecache_p_runahead_tokens", requested)
            )
            tokenizer, prefiller, decoder = upstream.pick_up_clients(request)
            tokenized_response = await upstream.send_request_to_service(
                tokenizer.client,
                "/tokenize",
                {"prompt": request_data["prompt"]},
            )
            prompt_ids = tokenized_response.json()["tokens"]
            original_max_tokens = int(request_data["max_tokens"])
            budget = runahead_budget(original_max_tokens, request_budget)
            request_data["prompt"] = prompt_ids
            request_data["max_tokens"] = budget
            request_data["return_token_ids"] = True

            slots = math.ceil(
                (len(prompt_ids) + budget) / upstream.global_args.chunk_size
            )
            if upstream.pd_buffer_semaphore is not None:
                await upstream.pd_buffer_semaphore.acquire(slots)
                acquired = True

            disagg_spec = {
                "req_id": request_id,
                "receiver_host": decoder.host,
                "receiver_init_port": decoder.init_port,
                "receiver_alloc_port": decoder.alloc_port,
            }
            request_data["kv_transfer_params"] = {
                "ret_first_tok": True,
                "disagg_spec": disagg_spec,
            }
            request_data["stream"] = False
            stream_options = request_data.pop("stream_options", None)
            started = time.time()
            prefill_response = await upstream.send_request_to_service(
                prefiller.client, "/v1/completions", request_data
            )
            prefill_output = prefill_response.json()
            upstream.stats_calculator.add(time.time() - started)
            decoder_request, generated_ids = prepare_decoder_request(
                request_data,
                prefill_output,
                original_max_tokens=original_max_tokens,
            )
            advertised_first = prefill_output.get("kv_transfer_params", {}).get(
                "first_tok"
            )
            if advertised_first != generated_ids[0]:
                raise RuntimeError("P token IDs disagree with KV-transfer metadata")
            if stream_options is not None:
                decoder_request["stream_options"] = stream_options
            num_tp_rank = len(decoder.init_port or [])

            async def generate_stream():
                nonlocal acquired
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
                            "finish_reason": (
                                prefill_output["choices"][0].get("finish_reason")
                                if decoder_request["max_tokens"] == 0
                                else None
                            ),
                            "stop_reason": prefill_output["choices"][0].get(
                                "stop_reason"
                            ),
                            "token_ids": generated_ids,
                        }
                    ],
                    "usage": None,
                    "sparsecache_p_runahead_tokens": len(generated_ids),
                }
                yield (
                    "data: " + json.dumps(head, separators=(",", ":")) + "\n\n"
                ).encode()
                try:
                    await upstream.wait_decode_kv_ready(request_id, num_tp_rank)
                finally:
                    if upstream.pd_buffer_semaphore is not None and acquired:
                        await upstream.pd_buffer_semaphore.release(slots)
                        acquired = False
                if decoder_request["max_tokens"] == 0:
                    yield b"data: [DONE]\n\n"
                    return
                async for chunk in upstream.stream_service_response(
                    decoder.client, "/v1/completions", decoder_request
                ):
                    yield chunk

            return StreamingResponse(generate_stream(), media_type="application/json")
        except Exception:
            if upstream.pd_buffer_semaphore is not None and acquired:
                await upstream.pd_buffer_semaphore.release(slots)
            raise

    upstream.global_args = upstream.parse_args()

    import uvicorn

    uvicorn.run(
        upstream.app,
        host=upstream.global_args.host,
        port=upstream.global_args.port,
    )


if __name__ == "__main__":
    main()
