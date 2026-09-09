"""Read bounded pilot artifacts and compare matched, completed model arms."""

import argparse
import json
from pathlib import Path
import random
import statistics


def read(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None  # A live writer has not finished publishing this artifact.


def ci(values):
    rng = random.Random(20260909)
    means = sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(5000))
    return [means[125],means[4875]]


def row_key(row):
    return row["record_id"],row["order"],row["fraction"],row["memory_mode"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    args=p.parse_args()
    root=Path(args.root)
    report={"lanes":{str(i):read(root/f"lane{i}_status.json") for i in range(3)}}
    for name in ("kvshot_gpu0","full_block_gpu0","nested_block_gpu1"):
        completed=read(root/name/"summary.json")
        if completed:
            report[name]=completed
        else:
            report[name]={"status":"pending","before":read(root/name/"before.json")}
            if report[name]["before"]:
                report[name]["before"].pop("rows",None)
    attention=read(root/"attention_gpu2"/"attention.json")
    if attention:
        bound_cells={}
        for row in attention["bounds"]:
            key=f"layer{row['layer']}/f{row['fraction']:g}"
            bound_cells.setdefault(key,[]).append(row)
        report["attention"]={"summary":attention["summary"],"cells":{
            key:{"requests":len(rows),
                 "missing_mass":statistics.mean(r["mean_missing_mass"] for r in rows),
                 "missing_mass_upper":statistics.mean(r["mean_missing_mass_upper"] for r in rows),
                 "bound_over_output_norm":statistics.mean(r["mean_bound_over_output_norm"] for r in rows)}
            for key,rows in bound_cells.items()}}
    full=read(root/"full_block_gpu0"/"after.json")
    nested=read(root/"nested_block_gpu1"/"after.json")
    if full and nested:
        fm=read(root/"full_block_gpu0"/"manifest.json")
        nm=read(root/"nested_block_gpu1"/"manifest.json")
        if fm["initial_checkpoint_sha256"] != nm["initial_checkpoint_sha256"]:
            raise RuntimeError("training arms do not have the same initialization")
        full_trace=(root/"full_block_gpu0"/"train.jsonl").read_text().splitlines()
        nested_trace=(root/"nested_block_gpu1"/"train.jsonl").read_text().splitlines()
        full_cuts=[(r["record_id"],r["cut"]) for r in map(json.loads,full_trace)]
        nested_cuts=[(r["record_id"],r["cut"]) for r in map(json.loads,nested_trace)]
        if full_cuts != nested_cuts:
            raise RuntimeError("training arms differ in records/cuts/update count")
        other={row_key(r):r for r in nested["rows"]}
        differences={}
        reference_mismatches={}
        for row in full["rows"]:
            partner=other[row_key(row)]
            if any(row[k] != partner[k] for k in ("input_sha256","seed")):
                raise RuntimeError("paired input or P seed differs")
            if row["reference"] != partner["reference"]:
                reference_mismatches[row["record_id"]]={
                    "full_arm":row["reference"],"nested_arm":partner["reference"]}
            cell=f"{row['order']}/f{row['fraction']:g}/{row['memory_mode']}"
            differences.setdefault(cell,[]).append(partner["accepted"]-row["accepted"])
        report["matched_training_audit"]={"same_checkpoint":True,"same_cuts":True,
            "same_reference":not reference_mismatches,"updates":len(full_cuts),
            "reference_mismatches":reference_mismatches,
            "paired_claim_valid":not reference_mismatches}
        if reference_mismatches:
            report["paired_comparison_status"]=(
                "BLOCKED: independently replayed full-target references drift; "
                "retain all raw rows and do not publish a paired training CI. "
                "Next run must reuse one immutable P-state/reference packet.")
        else:
            report["nested_minus_full_after"]={key:{"pairs":len(values),
                "mean_accepted_delta":statistics.mean(values),"ci95":ci(values)}
                for key,values in differences.items()}
    path=root/"comparison.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(path),"lane_status":{
        k:(v or {}).get("status","pending") for k,v in report["lanes"].items()},
        "matched_training_audit":report.get("matched_training_audit"),
        "primary":report.get("nested_minus_full_after",{}).get("priority/f0.1/exact"),
        "attention":report.get("attention",{}).get("summary")},indent=2))


if __name__ == "__main__":
    main()
