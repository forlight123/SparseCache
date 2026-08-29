import json

import pytest
import torch

try:
    from vllm.v1.spec_decode.eagle_proposal_trace import (
        publish_eagle_proposal_trace,
    )
except ModuleNotFoundError:
    pytest.skip(
        "the optional SparseCache vLLM working copy is absent",
        allow_module_level=True,
    )


def test_publish_eagle_proposal_trace_is_atomic_and_first_writer_wins(tmp_path):
    destination = tmp_path / "proposal.json"
    payload = publish_eagle_proposal_trace(
        destination=str(destination),
        method="eagle3",
        proposal_started_at_ns=1,
        seed_token_ids=torch.tensor([17], dtype=torch.int32),
        draft_token_ids=torch.tensor([[18, 19, 20]], dtype=torch.int32),
    )

    assert payload is not None
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored["state"] == "ready"
    assert stored["seed_token_id"] == 17
    assert stored["draft_token_ids"] == [18, 19, 20]
    assert stored["created_at_ns"] >= stored["proposal_started_at_ns"]
    assert not list(tmp_path.glob("*.tmp"))

    assert (
        publish_eagle_proposal_trace(
            destination=str(destination),
            method="eagle3",
            proposal_started_at_ns=2,
            seed_token_ids=torch.tensor([99]),
            draft_token_ids=torch.tensor([[100]]),
        )
        is None
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == stored


def test_publish_eagle_proposal_trace_is_disabled_for_other_methods(tmp_path):
    destination = tmp_path / "proposal.json"
    assert (
        publish_eagle_proposal_trace(
            destination=str(destination),
            method="eagle",
            proposal_started_at_ns=1,
            seed_token_ids=torch.tensor([1]),
            draft_token_ids=torch.tensor([[2]]),
        )
        is None
    )
    assert not destination.exists()
