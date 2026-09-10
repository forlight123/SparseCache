import json

import pytest

from experiments.lossless_pd.lmcache_pd.layer_ready_proxy import (
    load_oracle_drafts,
    prompt_token_digest,
)


def test_oracle_drafts_are_keyed_by_exact_prompt(tmp_path):
    digest = prompt_token_digest([1, 23, 456])
    path = tmp_path / "oracle.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "rows": [
                    {
                        "prompt_sha256": digest,
                        "output_token_ids": [7, 8, 9],
                    }
                ],
            }
        )
    )
    assert load_oracle_drafts(path) == {digest: (7, 8, 9)}
    assert prompt_token_digest([1, 23, 456]) != prompt_token_digest([12, 3, 456])


def test_oracle_drafts_reject_duplicate_prompts(tmp_path):
    digest = prompt_token_digest([1])
    path = tmp_path / "oracle.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "rows": [
                    {"prompt_sha256": digest, "output_token_ids": [1, 2]},
                    {"prompt_sha256": digest, "output_token_ids": [1, 3]},
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate oracle"):
        load_oracle_drafts(path)
