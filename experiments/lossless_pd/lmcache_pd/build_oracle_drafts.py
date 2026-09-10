"""Build an exact-prompt oracle proposal artifact from a live Target run.

This artifact is an optimistic systems ceiling, never a deployable drafter.
The resulting proposals still pass through the ordinary full-KV verifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.lossless_pd.lmcache_pd.layer_ready_proxy import (
    prompt_token_digest,
)
from experiments.lossless_pd.reference_packets import load_packet, read_index


def build(benchmark: dict, packet_root: Path) -> dict:
    index = read_index(packet_root)
    entries = {int(entry["index"]): entry for entry in index["entries"]}
    rows = []
    for row in benchmark.get("rows", []):
        packet_index = int(row["packet_index"])
        entry = entries.get(packet_index)
        if entry is None:
            raise ValueError(f"packet index is absent: {packet_index}")
        if entry["record_id"] != row["record_id"]:
            raise ValueError(f"record mismatch at packet {packet_index}")
        packet = load_packet(packet_root, entry)
        prompt_ids = [
            int(token)
            for token in packet["prompt_ids"][
                0, : int(benchmark["max_input_tokens"])
            ].tolist()
        ]
        output_ids = row["pd"].get("token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) < 2
            or any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in output_ids
            )
        ):
            raise ValueError(
                f"live Target row has no usable token IDs: {row['record_id']}"
            )
        rows.append(
            {
                "ordinal": int(row["ordinal"]),
                "packet_index": packet_index,
                "record_id": row["record_id"],
                "prompt_tokens": len(prompt_ids),
                "prompt_sha256": prompt_token_digest(prompt_ids),
                "output_token_ids": output_ids,
            }
        )
    if not rows:
        raise ValueError("benchmark contains no rows")
    return {
        "format_version": 1,
        "contract": (
            "same-stack ordinary Target trajectories keyed by exact prompt; "
            "optimistic verifier/control ceiling only"
        ),
        "source_benchmark": benchmark.get("contract"),
        "max_input_tokens": int(benchmark["max_input_tokens"]),
        "max_new_tokens": int(benchmark["max_new_tokens"]),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--packets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark_path = args.benchmark.resolve()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    packet_root = (
        args.packets.resolve() if args.packets else Path(benchmark["packets"]).resolve()
    )
    result = build(benchmark, packet_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(result["rows"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
