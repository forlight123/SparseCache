import json
from pathlib import Path
from types import SimpleNamespace

from experiments.run_progressive_pd_live_job import (
    ROOT,
    _foreign_compute_processes,
    build_launch_spec,
    load_job,
    validate_pack_hashes,
)

QUEUE = ROOT / "outputs/progressive_kv/live_queue_llama31_8b_20260827.json"
SCHEDULING_QUEUE = (
    ROOT / "outputs/progressive_kv/live_scheduling_queue_llama31_8b_20260827.json"
)
JOB_ID = "core_ruler_65536_evidence_first_bw25_n100"


def test_live_job_launch_spec_uses_preregistered_sparse_pd_protocol() -> None:
    job, queue_hash = load_job(QUEUE, JOB_ID)
    observed = validate_pack_hashes(job)

    spec = build_launch_spec(
        job,
        model="/model",
        host="127.0.0.1",
        lmcache_port=5555,
        prefill_port=8100,
        decoder_port=8200,
        proxy_port=8000,
        telemetry_port=5768,
        prefill_gpu=0,
        decode_gpu=1,
        l1_size_gb=40,
        audit_input_trace=True,
    )

    assert len(queue_hash) == 64
    assert observed["requests"] == job["pack_hashes"]["requests_sha256"]
    decoder = spec["commands"]["decoder"]
    assert "PROGRESSIVE_KV" in decoder
    speculative = json.loads(decoder[decoder.index("--speculative-config") + 1])
    assert speculative["method"] == "custom_class"
    assert speculative["num_speculative_tokens"] == 32
    assert spec["environments"]["prefiller"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert spec["environments"]["decoder"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert spec["environments"]["lmcache"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert spec["commands"]["benchmark"][-2:] == [
        "--url",
        "http://127.0.0.1:8000",
    ]
    connector = json.loads(decoder[decoder.index("--kv-transfer-config") + 1])
    assert connector["kv_role"] == "kv_consumer"
    extra = connector["kv_connector_extra_config"]
    assert extra["lmcache.mp.progressive_link_gbps"] == 25
    assert extra["lmcache.mp.progressive_retrieve_priority"] == "uniform"
    assert extra["lmcache.mp.progressive_bytes_per_token"] == 131072
    assert extra["lmcache.mp.controlled_link_stats_path"].endswith("/link.jsonl")
    prefiller = spec["commands"]["prefiller"]
    producer = json.loads(prefiller[prefiller.index("--kv-transfer-config") + 1])
    assert producer["kv_role"] == "kv_producer"


def test_launch_spec_uses_queue_model_shape_for_qwen() -> None:
    job, _ = load_job(QUEUE, JOB_ID)
    job = {
        **job,
        "model": {
            "name": "Qwen3-8B",
            "path": "/qwen",
            "served_model_name": "sparsecache-qwen3-8b",
            "max_position_embeddings": 40960,
            "logical_bf16_kv_bytes_per_token": 147456,
        },
    }
    spec = build_launch_spec(
        job,
        model="/qwen",
        host="127.0.0.1",
        lmcache_port=15555,
        prefill_port=18100,
        decoder_port=18200,
        proxy_port=18000,
        telemetry_port=15768,
        prefill_gpu=0,
        decode_gpu=1,
        l1_size_gb=40,
    )

    decoder = spec["commands"]["decoder"]
    assert decoder[decoder.index("--served-model-name") + 1] == "sparsecache-qwen3-8b"
    assert decoder[decoder.index("--max-model-len") + 1] == "40960"
    assert spec["environments"]["decoder"][
        "VLLM_PROGRESSIVE_KV_DOC_END_TOKEN"
    ] == "40960"
    assert spec["environments"]["decoder"][
        "VLLM_PROGRESSIVE_KV_PROTECTED_PREFIX_TOKENS"
    ] == "256"
    assert spec["environments"]["decoder"][
        "VLLM_PROGRESSIVE_KV_PROTECTED_SUFFIX_TOKENS"
    ] == "512"


def test_queue_job_paths_are_workspace_relative() -> None:
    job, _ = load_job(QUEUE, JOB_ID)

    assert not Path(job["pack_dir"]).is_absolute()
    assert not Path(job["run_dir"]).is_absolute()


def test_input_audit_is_decoder_only_and_counts_progressive_arms() -> None:
    job, _ = load_job(QUEUE, JOB_ID)
    spec = build_launch_spec(
        job,
        model="/model",
        host="127.0.0.1",
        lmcache_port=35555,
        prefill_port=38100,
        decoder_port=38200,
        proxy_port=38000,
        telemetry_port=35768,
        prefill_gpu=0,
        decode_gpu=1,
        l1_size_gb=128,
        audit_input_trace=True,
    )

    trace_variable = "VLLM_PROGRESSIVE_PD_INPUT_STATS_PATH"
    assert trace_variable not in spec["environments"]["prefiller"]
    assert trace_variable not in spec["environments"]["lmcache"]
    assert spec["environments"]["decoder"][trace_variable].endswith(
        "/input_tokens.jsonl"
    )
    audit = spec["commands"]["input_audit"]
    expected_index = audit.index("--expected-requests") + 1
    expected = (job["requests"] + job["warmup_requests"]) * 2
    assert audit[expected_index] == str(expected)


def test_runtime_gpu_monitor_allows_only_service_descendants() -> None:
    states = [
        SimpleNamespace(
            index=0,
            compute_processes=(
                {"pid": 30, "process_name": "owned", "used_gpu_memory_mib": 1},
                {"pid": 40, "process_name": "foreign", "used_gpu_memory_mib": 2},
            ),
        )
    ]
    parents = {30: 20, 20: 10, 40: 2, 2: 1}

    foreign = _foreign_compute_processes(
        states,
        (0,),
        {10},
        parent_lookup=parents.get,
    )

    assert [process["pid"] for process in foreign] == [40]


def test_scheduling_job_validates_every_request_priority_sidecar() -> None:
    job_id = "scheduling_selection_ruler_65536_original_bw25_n100_offset0"
    job, _ = load_job(SCHEDULING_QUEUE, job_id)

    observed = validate_pack_hashes(job)
    spec = build_launch_spec(
        job,
        model="/model",
        host="127.0.0.1",
        lmcache_port=25555,
        prefill_port=28100,
        decoder_port=28200,
        proxy_port=28000,
        telemetry_port=25768,
        prefill_gpu=0,
        decode_gpu=1,
        l1_size_gb=40,
    )

    assert {key for key in observed if key.startswith("priority:")} == {
        "priority:sequential",
        "priority:uniform",
        "priority:random",
        "priority:bm25",
        "priority:oracle",
    }
    assert spec["commands"]["benchmark"].count("--priority-sidecar") == 5
    assert spec["commands"]["aggregate"].count("--priority-sidecar") == 5
    audit = spec["commands"]["input_audit"]
    expected_index = audit.index("--expected-requests") + 1
    expected = (job["requests"] + job["warmup_requests"]) * 2 * 5
    assert audit[expected_index] == str(expected)
