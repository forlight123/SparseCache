import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lynx_baseline_cannot_be_mislabeled_as_exact_or_reproduced() -> None:
    matrix = json.loads(
        (ROOT / "configs/iclr_pd_experiment_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    policy = matrix["baseline_evidence_policy"]
    lynx = policy["lynx"]

    assert "quantspec_hierarchical_quantized_self_speculation" in matrix[
        "baselines"
    ]
    assert lynx["current_evidence_label"] == "authors_paper_only"
    assert not lynx["public_author_code_found"]
    assert not lynx["official_reproduction_allowed"]
    assert lynx["surrogate_must_not_be_named_reproduction"]
    assert lynx["completed_target"] == "hierarchical_int8_q_full"
    assert lynx["anchor_bits_per_coordinate"] == 4
    assert lynx["residual_bits_per_coordinate"] == 4
    assert "lynx_authors_paper" not in policy["endpoint_tables"]["exact_bf16"]
    assert "lynx_authors_paper" in policy["endpoint_tables"][
        "accuracy_latency_pareto"
    ]


def test_lynx_reproduction_gaps_and_protocol_are_materialized() -> None:
    matrix = json.loads(
        (ROOT / "configs/iclr_pd_experiment_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    lynx = matrix["baseline_evidence_policy"]["lynx"]
    missing = set(lynx["missing_for_faithful_local_reproduction"])

    assert "log_transform_alpha" in missing
    assert "packed_metadata_layout" in missing
    protocol = ROOT / lynx["protocol"]
    assert protocol.is_file()
    text = protocol.read_text(encoding="utf-8")
    assert "Lynx (authors, paper)" in text
    assert "split-INT8 matched-byte oracle" in text
    assert "Exact BF16 endpoint table" in text


def test_quantspec_is_a_compute_baseline_not_claimed_pd_reproduction() -> None:
    matrix = json.loads(
        (ROOT / "configs/iclr_pd_experiment_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    policy = matrix["baseline_evidence_policy"]
    quantspec = policy["quantspec"]

    assert quantspec["current_evidence_label"] == (
        "official_source_audited_reproduction_pending"
    )
    assert quantspec["public_author_code_found"]
    assert not quantspec["local_reproduction_complete"]
    assert not quantspec["progressive_pd_transfer_in_author_code"]
    assert quantspec["separate_environment_required"]
    assert not quantspec["repository_license_file_found"]
    assert not quantspec["vendoring_allowed"]
    assert "quantspec_hierarchical_quantized_self_speculation" not in policy[
        "endpoint_tables"
    ]["exact_bf16"]

    protocol = ROOT / quantspec["protocol"]
    assert protocol.is_file()
    text = protocol.read_text(encoding="utf-8")
    assert "official source audited; local reproduction pending" in text
    assert "compute-path comparator" in text
    assert "P/D transition comparator" in text
