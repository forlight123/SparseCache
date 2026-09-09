"""Build and audit immutable P-side packets for paired drafter experiments.

One full Target execution owns the canonical greedy trajectory.  Subsequent
drafter arms consume the same prompt KV, page priority, seed, and reference
tokens instead of independently replaying a numerically fragile BF16 Target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from experiments.kvshot.model import stack_sampled_target_kv
from experiments.lossless_pd.pilot import (
    context_forward,
    load_requests,
    load_target,
    priority_scores,
)


LAYER_IDS = [1, 9, 17, 25, 33]


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_identity(path: Path) -> dict:
    files = {}
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        candidate = path / name
        if candidate.exists():
            files[name] = digest_file(candidate)
    return {"path": str(path.resolve()), "metadata_sha256": files}


@torch.no_grad()
def greedy_trace(target, cache, seed, length: int, stop_ids: set[int]):
    """Return the canonical suffix and top1-minus-top2 logit margins."""

    token = seed
    tokens, margins = [], []
    if int(token.item()) in stop_ids:
        return tokens, margins
    for _ in range(length):
        output = target.model(token, past_key_values=cache, use_cache=True)
        logits = target.lm_head(output.last_hidden_state[:, -1:]).float()
        top = logits.topk(2, dim=-1)
        token = top.indices[..., :1].squeeze(-1)
        tokens.append(int(token.item()))
        margins.append(float((top.values[..., 0] - top.values[..., 1]).item()))
        cache = output.past_key_values
        if tokens[-1] in stop_ids:
            break
    return tokens, margins


def packet_path(root: Path, index: int) -> Path:
    return root / "packets" / f"request_{index:05d}.pt"


def read_index(root: Path) -> dict:
    index = json.loads((root / "index.json").read_text())
    if index.get("status") != "completed":
        raise ValueError("packet index is not complete")
    return index


def load_packet(root: Path, entry: dict) -> dict:
    path = root / entry["packet"]
    if digest_file(path) != entry["sha256"]:
        raise RuntimeError(f"immutable packet digest mismatch: {path}")
    packet = torch.load(path, map_location="cpu", weights_only=True)
    if packet["record_id"] != entry["record_id"]:
        raise RuntimeError(f"packet/index record mismatch: {path}")
    return packet


def build(args) -> None:
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    (root / "packets").mkdir(exist_ok=True)
    index_path = root / "index.json"
    if index_path.exists() and not args.resume:
        raise ValueError("packet index exists; pass --resume or choose a fresh directory")

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    requests = load_requests(
        args.requests, args.request_offset + args.num_requests, args.max_context
    )[args.request_offset:]
    target = load_target(args.target)
    eos = target.generation_config.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or [])
    metadata = {
        "format_version": 1,
        "status": "running",
        "created_unix": time.time(),
        "target": target_identity(Path(args.target)),
        "requests": str(Path(args.requests).resolve()),
        "num_requests": args.num_requests,
        "request_offset": args.request_offset,
        "max_context": args.max_context,
        "draft_tokens": args.draft_tokens,
        "page_size": args.page_size,
        "layer_ids": LAYER_IDS,
        "seed": args.seed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "contract": (
            "one canonical full-BF16/full-KV greedy Target execution; prompt KV, "
            "priority, seed, and suffix are immutable across all drafter arms"
        ),
        "entries": [],
    }
    write_json(index_path, metadata)

    for request_index, request in enumerate(requests):
        destination = packet_path(root, request_index)
        if destination.exists() and args.resume:
            digest = digest_file(destination)
            packet = torch.load(destination, map_location="cpu", weights_only=True)
            if packet["record_id"] != request["record_id"]:
                raise RuntimeError(f"resume record mismatch at {destination}")
        else:
            ids = torch.tensor([request["tokens"]], device="cuda", dtype=torch.long)
            output, queries = context_forward(target, ids, LAYER_IDS)
            keys, values = stack_sampled_target_kv(output.past_key_values, LAYER_IDS)
            scores = priority_scores(keys, queries, LAYER_IDS, args.page_size)
            prompt_logits = target.lm_head(output.last_hidden_state[:, -1:]).float()
            prompt_top = prompt_logits.topk(2, dim=-1)
            seed = prompt_top.indices[..., :1].squeeze(-1)
            reference, margins = greedy_trace(
                target, output.past_key_values, seed, args.draft_tokens, stop_ids
            )
            packet = {
                "format_version": 1,
                "record_id": request["record_id"],
                "source": request["source"],
                "source_index": request["source_index"],
                "original_tokens": request["original_tokens"],
                "prompt_tokens": request["prompt_tokens"],
                "truncated": request["truncated"],
                "input_sha256": request["input_sha256"],
                "prompt_ids": ids.cpu(),
                "layer_ids": torch.tensor(LAYER_IDS, dtype=torch.long),
                "keys": keys.cpu().contiguous(),
                "values": values.cpu().contiguous(),
                "priority_scores": scores.cpu(),
                "seed": seed.cpu(),
                "seed_margin": float(
                    (prompt_top.values[..., 0] - prompt_top.values[..., 1]).item()
                ),
                "reference": torch.tensor(reference, dtype=torch.long),
                "reference_margins": torch.tensor(margins, dtype=torch.float32),
                "stop_ids": torch.tensor(sorted(stop_ids), dtype=torch.long),
                "page_size": args.page_size,
            }
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(packet, temporary)
            os.replace(temporary, destination)
            digest = digest_file(destination)
            del ids, output, queries, keys, values, scores, prompt_logits, prompt_top
            torch.cuda.empty_cache()
        entry = {
            "index": request_index,
            "record_id": request["record_id"],
            "packet": str(destination.relative_to(root)),
            "sha256": digest,
            "bytes": destination.stat().st_size,
            "prompt_tokens": int(packet["prompt_tokens"]),
            "reference_tokens": int(packet["reference"].numel()),
            "input_sha256": packet["input_sha256"],
        }
        metadata["entries"] = [e for e in metadata["entries"] if e["index"] != request_index]
        metadata["entries"].append(entry)
        metadata["entries"].sort(key=lambda e: e["index"])
        write_json(index_path, metadata)
        print(json.dumps({"event": "packet", "completed": request_index + 1,
                          "total": len(requests), "bytes": entry["bytes"]}), flush=True)

    metadata.update(status="completed", completed_unix=time.time(),
                    total_bytes=sum(e["bytes"] for e in metadata["entries"]))
    write_json(index_path, metadata)
    print(json.dumps({"event": "completed", "output": str(root),
                      "total_bytes": metadata["total_bytes"]}), flush=True)


def replay(args) -> None:
    """Replay canonical targets in a fresh process/GPU and locate drift."""

    root = Path(args.packets).resolve()
    index = read_index(root)
    target = load_target(args.target)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    rows = []
    for completed, entry in enumerate(index["entries"]):
        packet = load_packet(root, entry)
        ids = packet["prompt_ids"].to("cuda")
        output = target.model(ids, use_cache=True, return_dict=True)
        prompt_logits = target.lm_head(output.last_hidden_state[:, -1:]).float()
        top = prompt_logits.topk(2, dim=-1)
        seed = top.indices[..., :1].squeeze(-1)
        stop_ids = set(packet["stop_ids"].tolist())
        reference, margins = greedy_trace(
            target, output.past_key_values, seed, index["draft_tokens"], stop_ids
        )
        canonical = packet["reference"].tolist()
        divergence = next(
            (i for i, (left, right) in enumerate(zip(canonical, reference)) if left != right),
            None,
        )
        if divergence is None and len(canonical) != len(reference):
            divergence = min(len(canonical), len(reference))
        row = {
            "record_id": packet["record_id"],
            "seed_equal": int(seed.item()) == int(packet["seed"].item()),
            "reference_equal": reference == canonical,
            "first_divergence": divergence,
            "canonical": canonical,
            "replay": reference,
            "canonical_seed_margin": packet["seed_margin"],
            "replay_seed_margin": float((top.values[..., 0] - top.values[..., 1]).item()),
            "canonical_reference_margins": packet["reference_margins"].tolist(),
            "replay_reference_margins": margins,
        }
        rows.append(row)
        print(json.dumps({"event": "replay", "completed": completed + 1,
                          "total": len(index["entries"]),
                          "reference_equal": row["reference_equal"]}), flush=True)
    result = {
        "packets": str(root),
        "packet_index_sha256": digest_file(root / "index.json"),
        "target": target_identity(Path(args.target)),
        "gpu": torch.cuda.get_device_name(),
        "deterministic_algorithms": args.deterministic,
        "requests": len(rows),
        "seed_mismatches": sum(not r["seed_equal"] for r in rows),
        "reference_mismatches": sum(not r["reference_equal"] for r in rows),
        "rows": rows,
    }
    write_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    build_parser.add_argument("--requests", default=(
        "outputs/progressive_kv/pd_packs/qwen3_8b/quality/qmsum/requests.jsonl"
    ))
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--num-requests", type=int, default=64)
    build_parser.add_argument("--request-offset", type=int, default=0)
    build_parser.add_argument("--max-context", type=int, default=8192)
    build_parser.add_argument("--draft-tokens", type=int, default=15)
    build_parser.add_argument("--page-size", type=int, default=64)
    build_parser.add_argument("--seed", type=int, default=20260909)
    build_parser.add_argument("--resume", action="store_true")

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    replay_parser.add_argument("--packets", required=True)
    replay_parser.add_argument("--output", required=True)
    replay_parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        if (
            min(args.num_requests, args.max_context, args.draft_tokens, args.page_size) <= 0
            or args.request_offset < 0
        ):
            parser.error("all resource and shape limits must be positive")
        build(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
