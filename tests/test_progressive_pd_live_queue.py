from pathlib import Path

from experiments.build_progressive_pd_live_queue import build_queue

PACK_ROOT = Path("outputs/progressive_kv/pd_packs/llama31_8b")


def test_live_queue_covers_core_controlled_and_quality_packs() -> None:
    queue = build_queue(
        PACK_ROOT,
        Path("outputs/test-live-matrix"),
        bandwidths=(25, 50),
        core_requests=100,
        confirmation_requests=100,
        warmup_requests=4,
        max_draft_tokens=8,
    )

    assert queue["counts"]["by_phase"] == {
        "core_mechanism": 16,
        "controlled_confirmation": 40,
        "natural_quality": 14,
    }
    assert queue["counts"]["jobs"] == 70
    assert len({job["id"] for job in queue["jobs"]}) == 70
    assert all(job["pack_hashes"]["requests_sha256"] for job in queue["jobs"])
    core = next(job for job in queue["jobs"] if job["phase"] == "core_mechanism")
    assert core["arms"] == ["baseline", "fixed_s1", "continuous"]
    assert core["pd_roles"] == {
        "prefiller": "kv_producer",
        "decoder": "kv_consumer",
    }
    assert core["validity_requirements"]["minimum_free_gpu_count"] == 2
    assert core["request_offset"] == 0
    assert core["model"]["name"] == "Llama-3.1-8B-Instruct"
    assert core["model"]["logical_bf16_kv_bytes_per_token"] == 131072
    assert core["model"]["eos_token_ids"] == [128001, 128008, 128009]
    assert core["aggregate_command"][
        core["aggregate_command"].index("--stop-token-ids") + 1
    ] == "128001,128008,128009"
    assert core["decoder_environment"][
        "VLLM_PROGRESSIVE_KV_PROTECTED_PREFIX_TOKENS"
    ] == "256"
    assert core["decoder_environment"][
        "VLLM_PROGRESSIVE_KV_PROTECTED_SUFFIX_TOKENS"
    ] == "512"
    assert core["decoder_connector_extra_config"][
        "lmcache.mp.progressive_protected_prefix_tokens"
    ] == 256
    assert core["decoder_connector_extra_config"][
        "lmcache.mp.progressive_protected_suffix_tokens"
    ] == 512
    fractions = [
        float(item)
        for item in core["decoder_connector_extra_config"][
            "lmcache.mp.progressive_retrieve_fractions"
        ].split(",")
    ]
    assert fractions == [step / 20 for step in range(1, 21)]
    assert (
        core["decoder_connector_extra_config"][
            "lmcache.mp.controlled_link_stats_path"
        ]
        == f"{core['run_dir']}/link.jsonl"
    )
    quality = next(job for job in queue["jobs"] if job["phase"] == "natural_quality")
    assert quality["bandwidth_gbps"] == "{FROZEN_BANDWIDTH_GBPS}"
    assert quality["fixed_output_tokens"] is None
    confirmation = next(
        job for job in queue["jobs"] if job["phase"] == "controlled_confirmation"
    )
    assert confirmation["requests"] == 100
    assert confirmation["request_offset"] == 100
