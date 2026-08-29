# SPDX-License-Identifier: Apache-2.0

import json

from sparsecache.data import load_hotpotqa_case, load_prepared_case


def test_hotpot_adapter_preserves_ten_passages(tmp_path):
    source = tmp_path / "hotpot.jsonl"
    row = {
        "_id": "case",
        "input": "question",
        "answers": ["answer"],
        "context": "".join(
            f"Passage {index}:\nTitle {index}\nBody {index}.\n"
            for index in range(1, 11)
        ),
    }
    source.write_text(json.dumps(row) + "\n")
    case = load_hotpotqa_case(source, index=0, max_chunks=10)
    assert len(case.chunks) == 10
    assert case.chunks[0].title == "Title 1"
    assert case.chunks[-1].text == "Body 10."

    prepared = tmp_path / "case.json"
    prepared.write_text(json.dumps(case.as_dict()))
    assert load_prepared_case(prepared) == case
