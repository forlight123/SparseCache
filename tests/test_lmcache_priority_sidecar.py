import json

import pytest

from experiments.lossless_pd.lmcache_pd.priority_sidecar import (
    chunk_priority_from_page_scores,
    load_priority_sidecar,
    token_digest,
)


def test_chunk_priority_aggregates_pages_and_protects_edges():
    # Four 2-page chunks. Chunk 2 has the largest unprotected total.
    ranking = chunk_priority_from_page_scores(
        [0.1, 0.1, 0.4, 0.4, 2.0, 1.0, 0.2, 0.2],
        prompt_tokens=8,
        page_tokens=1,
        chunk_tokens=2,
    )
    assert ranking == (0, 3, 2, 1)


def test_token_digest_accepts_tensor_style_batch_shape():
    assert token_digest([1, 2, 3]) == token_digest([[1, 2, 3]])
    with pytest.raises(TypeError):
        token_digest([1, True])


def test_priority_sidecar_loader_is_strict(tmp_path):
    path = tmp_path / "priority.jsonl"
    digest = token_digest([1, 2, 3])
    path.write_text(
        json.dumps({"prompt_digest": digest, "priority_chunks": [0, 2, 1]}) + "\n"
    )
    assert load_priority_sidecar(path) == {digest: (0, 2, 1)}

    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt_digest": digest, "priority_chunks": [0, 2, 1]}),
                json.dumps({"prompt_digest": digest, "priority_chunks": [0, 2, 1]}),
            ]
        )
        + "\n"
    )
    assert load_priority_sidecar(path) == {digest: (0, 2, 1)}

    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt_digest": digest, "priority_chunks": [0, 2, 1]}),
                json.dumps({"prompt_digest": digest, "priority_chunks": [0, 1, 2]}),
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError):
        load_priority_sidecar(path)
