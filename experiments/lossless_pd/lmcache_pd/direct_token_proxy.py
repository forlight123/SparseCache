"""Run LMCache's example P/D proxy without retokenizing integer prompts.

The upstream completion proxy always calls vLLM's ``/tokenize`` endpoint.
That endpoint accepts text but not an already-tokenized integer prompt, even
though ``/v1/completions`` itself accepts token IDs.  Frozen-teacher replay
needs byte-for-byte token identity, so this wrapper returns the supplied IDs
from the tokenization step and delegates every other operation to the upstream
proxy module.
"""

from __future__ import annotations

from typing import Any


def is_token_id_prompt(prompt: Any) -> bool:
    return (
        isinstance(prompt, list)
        and bool(prompt)
        and all(
            isinstance(token, int) and not isinstance(token, bool) for token in prompt
        )
    )


class _TokenizeResponse:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens

    def json(self) -> dict[str, list[int]]:
        return {"tokens": self.tokens}


def main() -> None:
    try:
        from examples.disagg_prefill import disagg_proxy_server as upstream
    except ImportError as error:
        raise RuntimeError(
            "add the adjacent LMCache repository root to PYTHONPATH"
        ) from error

    original = upstream.send_request_to_service

    async def preserve_token_ids(client, endpoint: str, request: dict):
        prompt = request.get("prompt")
        if endpoint == "/tokenize" and is_token_id_prompt(prompt):
            return _TokenizeResponse(list(prompt))
        return await original(client, endpoint, request)

    upstream.send_request_to_service = preserve_token_ids
    upstream.global_args = upstream.parse_args()

    import uvicorn

    uvicorn.run(
        upstream.app,
        host=upstream.global_args.host,
        port=upstream.global_args.port,
    )


if __name__ == "__main__":
    main()
