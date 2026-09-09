"""Launch the bounded, three-GPU pilot after checking exclusive availability.

Each GPU has one sequential queue, a persistent log, PID, and exit status.
No training is restarted automatically after a failure.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def commands(root, lane, steps, requests):
    base = [sys.executable,"-m","experiments.lossless_pd.pilot",
            "--num-requests",str(requests),"--max-context","8192"]
    if lane == 0:
        return [base+["--mode","kvshot","--output",str(root/"kvshot_gpu0")],
                base+["--mode","block","--steps",str(steps),"--train-views","full",
                      "--output",str(root/"full_block_gpu0")]]
    if lane == 1:
        return [base+["--mode","block","--steps",str(steps),"--train-views","nested",
                      "--output",str(root/"nested_block_gpu1")]]
    return [base+["--mode","attention","--repeats","5",
                  "--output",str(root/"attention_gpu2")]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--steps",type=int,default=1000)
    parser.add_argument("--requests",type=int,default=64)
    parser.add_argument("--worker",type=int,choices=[0,1,2])
    args = parser.parse_args()
    if args.steps < 0 or args.requests < 2:
        parser.error("nonnegative steps and at least two requests required")
    root = Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if args.worker is not None:
        status_path = root/f"lane{args.worker}_status.json"
        status = {"pid":os.getpid(),"lane":args.worker,"started":time.time(),"jobs":[]}
        for command in commands(root,args.worker,args.steps,args.requests):
            status.update(status="running",current_command=command)
            status_path.write_text(json.dumps(status,indent=2)+"\n")
            result = subprocess.run(command,check=False)
            status["jobs"].append({"command":command,"exit_code":result.returncode})
            if result.returncode:
                status.update(status="failed",finished=time.time())
                status_path.write_text(json.dumps(status,indent=2)+"\n")
                raise SystemExit(result.returncode)
        status.update(status="completed",finished=time.time())
        status.pop("current_command",None)
        status_path.write_text(json.dumps(status,indent=2)+"\n")
        return
    launch_path = root/"launch.json"
    if launch_path.exists():
        raise ValueError("launch manifest exists; use a fresh output directory")
    processes = subprocess.check_output([
        "nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"
    ],text=True).strip()
    if processes:
        raise RuntimeError(f"GPU processes exist; no jobs started:\n{processes}")
    inventory = subprocess.check_output([
        "nvidia-smi","--query-gpu=index,uuid,name,memory.used","--format=csv,noheader"
    ],text=True)
    devices = {}
    for row in inventory.strip().splitlines():
        fields = [x.strip() for x in row.split(",")]
        devices[int(fields[0])] = fields[1]
    if not all(index in devices for index in range(3)):
        raise RuntimeError("three GPUs are required")
    launched = []
    for lane in range(3):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=devices[lane],OMP_NUM_THREADS="8",
                   TOKENIZERS_PARALLELISM="false")
        command = ["numactl",f"--cpunodebind={1 if lane==2 else 0}",
                   f"--membind={1 if lane==2 else 0}",sys.executable,
                   "-m","experiments.lossless_pd.launch","--output",str(root),
                   "--steps",str(args.steps),"--requests",str(args.requests),
                   "--worker",str(lane)]
        with (root/f"lane{lane}.log").open("x") as log:
            worker = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                      start_new_session=True)
        launched.append({"gpu":lane,"uuid":devices[lane],"pid":worker.pid,
                         "command":command,"log":str(root/f"lane{lane}.log")})
    launch_path.write_text(json.dumps({"started":time.time(),"inventory":inventory,
                                      "workers":launched},indent=2)+"\n")
    print(json.dumps({"output":str(root),"workers":launched},indent=2))


if __name__ == "__main__":
    main()
