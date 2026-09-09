"""Compare progressive output against the authoritative full-KV P/D path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(progressive: dict, full_pd: dict) -> dict:
    progressive_rows = progressive["rows"]
    full_rows = full_pd["rows"]
    progressive_ids = [row["record_id"] for row in progressive_rows]
    full_ids = [row["record_id"] for row in full_rows]
    if progressive_ids != full_ids:
        raise ValueError("benchmark record order differs")
    exact = [
        left["pd"]["text"] == right["pd"]["text"]
        for left, right in zip(progressive_rows, full_rows, strict=True)
    ]
    progressive_mono = [
        index for index, row in enumerate(progressive_rows) if not row["outputs_equal"]
    ]
    full_mono = [
        index for index, row in enumerate(full_rows) if not row["outputs_equal"]
    ]
    return {
        "requests": len(exact),
        "progressive_equals_full_pd": sum(exact),
        "progressive_full_pd_mismatch_ordinals": [
            index for index, matches in enumerate(exact) if not matches
        ],
        "progressive_monolithic_mismatch_ordinals": progressive_mono,
        "full_pd_monolithic_mismatch_ordinals": full_mono,
        "same_monolithic_mismatch_set": progressive_mono == full_mono,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progressive", required=True)
    parser.add_argument("--full-pd", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    progressive = json.loads(Path(args.progressive).read_text())
    full_pd = json.loads(Path(args.full_pd).read_text())
    result = {
        "contract": (
            "progressive Anchor/Residual output equality against correct-hash "
            "authoritative full-KV P/D; monolithic equality is diagnostic"
        ),
        "progressive": str(Path(args.progressive).resolve()),
        "full_pd": str(Path(args.full_pd).resolve()),
        "summary": compare(progressive, full_pd),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
