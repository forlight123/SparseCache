import hashlib
import json
from pathlib import Path

import pytest

from experiments.audit_ruler_pd_packs import audit_matrix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cell(root: Path, length: int, placement: str, *, row_id: str) -> None:
    cell = root / str(length) / placement
    cell.mkdir(parents=True)
    request_path = cell / "requests.jsonl"
    metadata_path = cell / "metadata.jsonl"
    request = {
        "model": "test",
        "prompt": list(range(length)),
        "max_tokens": 32,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    }
    metadata = {
        "request_index": 0,
        "source_index": 0,
        "id": row_id,
        "dataset": "ruler_qa2",
        "format": "ruler",
        "placement": placement,
        "answers": ["answer"],
        "prompt_tokens": length,
        "document_fit_mode": "repeat_distractors",
        "supporting_documents": 2,
        "output_tokens": 32,
        "fixed_output_horizon": True,
    }
    request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "ruler_qa2",
        "placement": placement,
        "requests": 1,
        "prompt_tokens": length,
        "chunk_tokens": 256,
        "output_tokens": 32,
        "document_fit_modes": {"repeat_distractors": 1},
        "requests_sha256": _sha256(request_path),
        "metadata_sha256": _sha256(metadata_path),
    }
    (cell / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )


def test_audit_matrix_validates_paired_control_cells(tmp_path: Path) -> None:
    _write_cell(tmp_path, 256, "original", row_id="same")
    _write_cell(tmp_path, 256, "evidence_first", row_id="same")

    result = audit_matrix(
        tmp_path,
        lengths=(256,),
        placements=("original", "evidence_first"),
        requests_per_cell=1,
    )

    assert result["status"] == "passed"
    assert result["cell_count"] == 2
    assert result["total_requests"] == 2
    assert result["total_prompt_tokens"] == 512


def test_audit_matrix_rejects_unpaired_samples(tmp_path: Path) -> None:
    _write_cell(tmp_path, 256, "original", row_id="first")
    _write_cell(tmp_path, 256, "evidence_first", row_id="different")

    with pytest.raises(ValueError, match="identity/order mismatch"):
        audit_matrix(
            tmp_path,
            lengths=(256,),
            placements=("original", "evidence_first"),
            requests_per_cell=1,
        )
