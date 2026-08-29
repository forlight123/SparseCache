import pytest

from experiments.build_longbench_pd_pack import (
    _render_longbench,
    align_context_to_chunk,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return messages[0]["content"]


def test_alignment_removes_only_context_and_preserves_both_ends() -> None:
    prefix = [100, 101, 102]
    context = list(range(20))
    suffix = [200, 201]

    prompt, cap_removed, alignment_removed, alignment_added = align_context_to_chunk(
        prefix,
        context,
        suffix,
        max_prompt_tokens=32,
        chunk_tokens=8,
    )

    assert len(prompt) == 24
    assert prompt[:3] == prefix
    assert prompt[-2:] == suffix
    assert cap_removed == 0
    assert alignment_removed == 1
    assert alignment_added == 0
    assert prompt[3:13] == list(range(10))
    assert prompt[13:-2] == list(range(11, 20))


def test_model_cap_and_chunk_alignment_are_accounted_separately() -> None:
    prompt, cap_removed, alignment_removed, alignment_added = align_context_to_chunk(
        [100, 101],
        list(range(100)),
        [200, 201],
        max_prompt_tokens=32,
        chunk_tokens=8,
    )

    assert len(prompt) == 32
    assert cap_removed == 72
    assert alignment_removed == 0
    assert alignment_added == 0


def test_short_context_is_padded_instead_of_removed() -> None:
    prompt, cap_removed, alignment_removed, alignment_added = align_context_to_chunk(
        list(range(100)),
        [300, 301],
        list(range(100, 200)),
        max_prompt_tokens=512,
        chunk_tokens=256,
        padding_token_id=999,
    )

    assert len(prompt) == 256
    assert prompt[100:102] == [300, 301]
    assert prompt[102:156] == [999] * 54
    assert cap_removed == 0
    assert alignment_removed == 0
    assert alignment_added == 54


def test_alignment_rejects_scaffolding_without_context_budget() -> None:
    with pytest.raises(ValueError, match="no context"):
        align_context_to_chunk(
            list(range(16)),
            [1, 2],
            list(range(16)),
            max_prompt_tokens=32,
            chunk_tokens=8,
        )


def test_qwen_style_thinking_is_explicitly_disabled() -> None:
    tokenizer = FakeTokenizer()
    rendered, placeholder = _render_longbench(
        tokenizer,
        "qmsum",
        {"input": "What happened?"},
        disable_thinking=True,
    )

    assert placeholder in rendered
    assert tokenizer.kwargs["enable_thinking"] is False
