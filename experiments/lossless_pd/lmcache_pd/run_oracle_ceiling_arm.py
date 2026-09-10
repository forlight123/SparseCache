"""Run one fresh-server real-LMCache oracle-ceiling arm on two GPUs.

The oracle supplies a prerecorded same-stack Target trajectory at zero draft
cost. It is intentionally an optimistic verifier/control upper bound, not a
deployable method.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path("/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python")
VLLM = Path("/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/vllm")
MODEL = Path("/data/models/qwen/Qwen3-8B")
BUFFER_BYTES = 25_769_803_776


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": ":".join(
                [
                    str(ROOT),
                    str(ROOT / "experiments/lossless_pd/lmcache_pd/runtime_site"),
                    str(ROOT / "LMCache"),
                ]
            ),
            "ALL_PROXY": "",
            "all_proxy": "",
            "HTTP_PROXY": "",
            "http_proxy": "",
            "HTTPS_PROXY": "",
            "https_proxy": "",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return env


def assert_ports_free(ports: list[int]) -> None:
    for port in ports:
        with socket.socket() as channel:
            channel.settimeout(0.1)
            if channel.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"required port is occupied: {port}")


def wait_http(
    url: str,
    processes: list[subprocess.Popen],
    *,
    timeout: float,
    require_success: bool,
) -> None:
    deadline = time.monotonic() + timeout
    session = requests.Session()
    session.trust_env = False
    while time.monotonic() < deadline:
        failed = [process.pid for process in processes if process.poll() is not None]
        if failed:
            raise RuntimeError(f"service exited during startup: {failed}")
        try:
            response = session.get(url, timeout=0.5)
            if not require_success or response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"service did not become ready: {url}")


def stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def vllm_common(
    port: int,
    attention_backend: str | None = None,
    *,
    enable_prefix_caching: bool = False,
) -> list[str]:
    command = [
        str(VLLM),
        "serve",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8304",
        "--gpu-memory-utilization",
        "0.40",
        "--max-num-seqs",
        "1",
        "--no-async-scheduling",
        "--enforce-eager",
        "--disable-log-stats",
    ]
    command.append(
        "--enable-prefix-caching"
        if enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
    if attention_backend:
        command.extend(["--attention-backend", attention_backend])
    return command


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"fresh output directory required: {output_dir}")
    output_dir.mkdir(parents=True)
    assert_ports_free(
        [
            args.prefiller_port,
            args.decoder_port,
            args.proxy_port,
            17300,
            17400,
            17500,
            17600,
            17610,
            17620,
        ]
    )
    base = local_env()
    if args.shape_invariant_verifier:
        base.update(
            {
                "SPARSECACHE_SHAPE_INVARIANT_VERIFIER": "1",
                "SPARSECACHE_SHAPE_FIXED_SPLIT_PAGES": str(
                    args.shape_fixed_split_pages
                ),
            }
        )
    p_env = {
        **base,
        "CUDA_VISIBLE_DEVICES": str(args.prefill_gpu),
        "LMCACHE_CONFIG_FILE": str(
            ROOT / "experiments/lossless_pd/lmcache_pd/prefiller_layerwise_64.yaml"
        ),
        "SPARSECACHE_LAYERWISE_PD_PATCH": "1",
        "SPARSECACHE_ANCHOR_FRACTION": "0.1",
        "SPARSECACHE_ANCHOR_MODE": "protected_uniform",
        "SPARSECACHE_GATHER_FIRST": "1",
        "SPARSECACHE_DRAFT_LAYERS": "1,9,17,25,33",
        "SPARSECACHE_LAYERWISE_TRACE": str(output_dir / "sender.jsonl"),
        "SPARSECACHE_ANCHOR_NOTIFY": "udp://127.0.0.1:17600",
    }
    d_env = {
        **base,
        "CUDA_VISIBLE_DEVICES": str(args.decode_gpu),
        "LMCACHE_CONFIG_FILE": str(
            ROOT / "experiments/lossless_pd/lmcache_pd/decoder_layerwise_64.yaml"
        ),
        "SPARSECACHE_LAYERWISE_PD_PATCH": "1",
        "SPARSECACHE_RECEIVER_PATCH": "1",
        "SPARSECACHE_ANCHOR_LISTEN": "udp://127.0.0.1:17600",
        "SPARSECACHE_RECEIVER_TRACE": str(output_dir / "receiver.jsonl"),
        "SPARSECACHE_DRAFTER_KIND": "external",
        "SPARSECACHE_DRAFT_LAYERS": "1,9,17,25,33",
        "SPARSECACHE_DRAFT_TOKENS": str(args.draft_tokens),
        "SPARSECACHE_EXTERNAL_DRAFT_LISTEN": "udp://127.0.0.1:17620",
        "SPARSECACHE_DRAFT_NOTIFY": "udp://127.0.0.1:17610",
        "SPARSECACHE_VERIFY_NOTIFY": "udp://127.0.0.1:17610",
        "SPARSECACHE_ONLINE_DRAFT_MODE": args.mode,
        "SPARSECACHE_DRAFT_TRACE": str(output_dir / "draft.jsonl"),
    }
    proposer_p = (
        '{"method":"custom_class","model":"experiments.lossless_pd.lmcache_pd.'
        'seed_signal_proposer.SeedSignalProposer","num_speculative_tokens":1}'
    )
    proposer_d = (
        '{"method":"custom_class","model":"experiments.lossless_pd.lmcache_pd.'
        'online_drafter.OnlineSparseKVProposer","num_speculative_tokens":'
        f"{args.speculative_tokens}}}"
    )
    p_command = [
        *vllm_common(
            args.prefiller_port,
            args.attention_backend,
            enable_prefix_caching=args.enable_prefix_caching,
        ),
        "--speculative-config",
        proposer_p,
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}',
    ]
    d_command = [
        *vllm_common(
            args.decoder_port,
            args.attention_backend,
            enable_prefix_caching=args.enable_prefix_caching,
        ),
        "--speculative-config",
        proposer_d,
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer"}',
    ]
    processes: list[subprocess.Popen] = []
    handles = []
    try:
        for name, command, env in (
            ("prefiller", p_command, p_env),
            ("decoder", d_command, d_env),
        ):
            handle = (output_dir / f"{name}.log").open("wb")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    command,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        wait_http(
            f"http://127.0.0.1:{args.prefiller_port}/health",
            processes,
            timeout=args.startup_timeout,
            require_success=True,
        )
        wait_http(
            f"http://127.0.0.1:{args.decoder_port}/health",
            processes,
            timeout=args.startup_timeout,
            require_success=True,
        )
        proxy_env = {
            **base,
            "SPARSECACHE_ORACLE_DRAFT_JSON": str(args.oracle.resolve()),
            "SPARSECACHE_EXTERNAL_DRAFT_TOKENS": str(args.draft_tokens),
            "SPARSECACHE_EXTERNAL_DRAFT_NOTIFY": "udp://127.0.0.1:17620",
            "SPARSECACHE_DRAFT_LISTEN": "udp://127.0.0.1:17610",
            "SPARSECACHE_PROXY_TRACE": str(output_dir / "proxy.jsonl"),
            "SPARSECACHE_EARLY_DISPATCH": "true",
            "SPARSECACHE_CANONICAL_REPLAY_ON_REJECT": str(
                args.canonical_replay
            ).lower(),
            "SPARSECACHE_ISOLATE_PREFIX_CACHE": str(
                args.enable_prefix_caching
            ).lower(),
        }
        proxy_command = [
            str(PYTHON),
            "-m",
            "experiments.lossless_pd.lmcache_pd.layer_ready_proxy",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.proxy_port),
            "--prefiller-host",
            "127.0.0.1",
            "--prefiller-port",
            str(args.prefiller_port),
            "--decoder-host",
            "127.0.0.1",
            "--decoder-port",
            str(args.decoder_port),
            "--decoder-init-port",
            "17300",
            "--decoder-alloc-port",
            "17400",
            "--proxy-host",
            "127.0.0.1",
            "--proxy-port",
            "17500",
            "--model",
            str(MODEL),
            "--pd-buffer-size",
            str(BUFFER_BYTES),
            "--chunk-size",
            "64",
        ]
        handle = (output_dir / "proxy.log").open("wb")
        handles.append(handle)
        processes.append(
            subprocess.Popen(
                proxy_command,
                env=proxy_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )
        wait_http(
            f"http://127.0.0.1:{args.proxy_port}/health",
            processes,
            timeout=60,
            require_success=False,
        )
        benchmark_command = [
            str(PYTHON),
            "-m",
            "experiments.lossless_pd.lmcache_pd.benchmark_pd_packets",
            "--packets",
            str(args.packets.resolve()),
            "--output",
            str(output_dir / "result.json"),
            "--num-requests",
            str(args.num_requests),
            "--max-input-tokens",
            str(args.max_input_tokens),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--pd-url",
            f"http://127.0.0.1:{args.proxy_port}/v1/completions",
        ]
        if args.packet_indices:
            benchmark_command.extend(["--packet-indices", args.packet_indices])
        if args.logprobs > 0:
            benchmark_command.extend(["--logprobs", str(args.logprobs)])
        with (output_dir / "benchmark.log").open("wb") as handle:
            subprocess.run(
                benchmark_command,
                env=base,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        stop_processes(processes)
        for handle in handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--mode", choices=("observe", "inject"), required=True)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument(
        "--packet-indices",
        help="comma-separated exact packet indexes; overrides --num-requests selection",
    )
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--speculative-tokens", type=int, default=7)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--prefiller-port", type=int, default=18100)
    parser.add_argument("--decoder-port", type=int, default=18200)
    parser.add_argument("--proxy-port", type=int, default=19100)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--canonical-replay", action="store_true")
    parser.add_argument(
        "--shape-invariant-verifier",
        action="store_true",
        help=(
            "verify uniform speculative blocks as fixed-split batched qlen=1 "
            "FlashInfer queries"
        ),
    )
    parser.add_argument("--shape-fixed-split-pages", type=int, default=64)
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="reuse the completed speculative prefix during rare canonical replay",
    )
    parser.add_argument(
        "--attention-backend",
        choices=(
            "FLASH_ATTN",
            "FLASHINFER",
            "TRITON_ATTN",
        ),
        help="force the same vLLM attention backend on P and D",
    )
    args = parser.parse_args()
    if (
        min(
            args.num_requests,
            args.max_input_tokens,
            args.max_new_tokens,
            args.draft_tokens,
            args.speculative_tokens,
        )
        <= 0
    ):
        parser.error("request, token, and horizon values must be positive")
    if args.logprobs < 0:
        parser.error("logprobs must be non-negative")
    if args.shape_fixed_split_pages <= 0:
        parser.error("--shape-fixed-split-pages must be positive")
    if args.shape_invariant_verifier and args.attention_backend != "FLASHINFER":
        parser.error("shape-invariant verifier requires --attention-backend FLASHINFER")
    run(args)


if __name__ == "__main__":
    main()
