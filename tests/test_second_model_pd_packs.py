import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "outputs/progressive_kv/pd_packs/qwen3_8b"
QUEUE = ROOT / "outputs/progressive_kv/live_queue_qwen3_8b_20260827.json"


def _load(relative_path: str) -> dict:
    with (PACK_ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_qwen3_8b_materialized_dataset_scale_is_frozen() -> None:
    longbench = _load("quality_audit.json")
    longbench_v2 = _load("longbench_v2_audit.json")
    ruler = _load("controlled/ruler_qa2_audit.json")

    assert (longbench["status"], longbench["dataset_count"]) == ("passed", 13)
    assert (longbench["total_requests"], longbench["total_prompt_tokens"]) == (
        3150,
        32232192,
    )
    assert (longbench_v2["status"], longbench_v2["dataset_count"]) == (
        "passed",
        1,
    )
    assert (
        longbench_v2["total_requests"],
        longbench_v2["total_prompt_tokens"],
    ) == (503, 15155712)
    assert (ruler["status"], ruler["lengths"], ruler["cell_count"]) == (
        "passed",
        [16384, 32768],
        10,
    )
    assert (ruler["total_requests"], ruler["total_prompt_tokens"]) == (
        2000,
        49152000,
    )


def test_qwen3_8b_packs_use_native_chat_without_thinking() -> None:
    manifests = [
        _load("quality/2wikimqa/manifest.json"),
        _load("quality/longbench_v2/manifest.json"),
        _load("controlled/ruler_qa2/16384/evidence_first/manifest.json"),
    ]

    assert all(item["tokenizer"] == "/data/models/qwen/Qwen3-8B" for item in manifests)
    assert all(item["served_model_name"] == "sparsecache-qwen3-8b" for item in manifests)
    assert all(item["thinking_mode"] == "disabled" for item in manifests)
    assert manifests[0]["prompt_mode"] == "official_chat"
    assert manifests[2]["prompt_mode"] == "tokenizer_chat_template"


def test_qwen3_8b_live_queue_matches_model_and_pack_limits() -> None:
    with QUEUE.open(encoding="utf-8") as handle:
        queue = json.load(handle)

    assert queue["counts"] == {
        "jobs": 46,
        "by_phase": {
            "core_mechanism": 12,
            "controlled_confirmation": 20,
            "natural_quality": 14,
        },
        "by_status": {
            "ready_when_two_gpus_are_exclusive": 12,
            "blocked_on_core_gate_and_method_freeze": 20,
            "template_blocked_on_method_freeze": 14,
        },
    }
    assert queue["protocol"]["model"] == {
        "name": "Qwen3-8B",
        "path": "/data/models/qwen/Qwen3-8B",
        "served_model_name": "sparsecache-qwen3-8b",
        "max_position_embeddings": 40960,
        "logical_bf16_kv_bytes_per_token": 147456,
    }
    core = [job for job in queue["jobs"] if job["phase"] == "core_mechanism"]
    assert {job["bandwidth_gbps"] for job in core} == {25, 50}
    assert {
        job["decoder_connector_extra_config"][
            "lmcache.mp.progressive_bytes_per_token"
        ]
        for job in core
    } == {147456}
    assert {
        job["decoder_environment"]["VLLM_PROGRESSIVE_KV_DOC_END_TOKEN"]
        for job in core
    } == {"40960"}
