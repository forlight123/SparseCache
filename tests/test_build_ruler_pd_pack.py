from experiments.build_ruler_pd_pack import _render_parts


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return f"PREFIX {messages[1]['content']} SUFFIX"

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())


def test_ruler_prompt_uses_model_chat_template_and_disables_thinking() -> None:
    tokenizer = FakeTokenizer()
    row = {
        "question": "Q?",
        "placements": {"evidence_first": [0]},
        "documents": [{"text": "evidence", "supporting": True}],
    }

    prefix, documents, suffix = _render_parts(
        tokenizer, row, "evidence_first", disable_thinking=True
    )

    assert prefix and suffix
    assert len(documents) == 1
    assert documents[0]["supporting"]
    assert documents[0]["source_document_index"] == 0
    assert tokenizer.template_kwargs["enable_thinking"] is False
