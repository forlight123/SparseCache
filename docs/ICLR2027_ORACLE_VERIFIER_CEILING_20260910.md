# Oracle ceiling exposes the next SparseCache bottleneck

Date: 2026-09-10

## Outcome

The current vLLM block verifier has no measured configuration that is both
bitwise lossless and at least 1.10x faster, even when drafting is made
unrealistically perfect and free. This stops the current verifier path before
any more drafter training.

This is a structural result, not an ICLR-ready method result. It says that the
next revision must change verification numerics or its rollback cost. Improving
the 4B drafter cannot cross a ceiling that already fails with a Target oracle.

## Experimental contract

The experiment uses the real LMCache layerwise P/D deployment with NIXL
CUDA-IPC. Qwen3-8B P runs on H200 GPU 0 and the identical full-KV D Target runs
on H200 GPU 1; GPU 2 is idle. The workload is 64 length-stratified QMSum packet
prompts spanning 2,560--8,192 tokens, one request at a time, greedy decoding,
and prefix caching disabled.

An ordinary live D Target run first records a trajectory keyed by the exact
integer prompt. The oracle supplies that same-stack trajectory at zero draft
cost while P and KV movement proceed. D still invokes its normal immutable
full-KV verifier, and no oracle token is returned without verification. Every
latency cell restarts P, D, and the proxy and uses an
observe--inject--observe sandwich.

The oracle is not a deployable algorithm. It is a strict optimistic upper bound
on the current verifier/control structure.

## N=64 Pareto frontier

Here `g` is the maximum injected suffix after the D Target's first alignment
token. Positive confidence intervals report milliseconds saved by injection.

| cell | same-stack IDs | mean accepted | saved ms (95% CI) | speedup | gate |
|---|---:|---:|---:|---:|---|
| oracle `g=7` | 64/64 | 7.000 | 62.215 [57.089, 67.543] | 1.0809x | speed fail |
| oracle `g=8` | 61/64 | 7.766 | 124.924 [113.429, 137.997] | 1.1672x | exactness fail |
| oracle `g=9` | 61/64 | 8.719 | 95.398 [88.235, 102.438] | 1.1219x | exactness fail |
| oracle `g=10` | 61/64 | 9.672 | 113.076 [104.985, 120.859] | 1.1456x | exactness fail |
| oracle `g=8` + canonical replay | 64/64 | 7.766 | 59.705 [36.664, 78.749] | 1.0743x | speed fail |

The three mismatches at `g=8,9,10` are the same ordinals 30, 58, and 61. Their
oracle suffixes are rejected after 2, 6, and 1 accepted tokens respectively.
Because the proposals came from the ordinary Target itself, these are not
drafter mistakes. They are greedy argmax changes caused by the block execution
shape and floating-point reduction path.

Longer diagnostic blocks make the speed/exactness separation larger. At N=16,
`g=15` reaches 1.3004x but only 15/16 exact; one of two inject attempts also
stalls after verify feedback and times out. `g=31` reaches 1.4213x but only
13/16 exact.

## Canonical replay result

The prototype adds a fail-closed control message from D to the proxy. D reports
the accepted suffix after block verification. The proxy buffers uncommitted D
chunks and, on any rejection, discards them and reruns the ordinary canonical
Target path after FullReady. Exactly the three unstable requests replayed, at
362.292 ms mean cost, restoring 64/64 output identity. That rare slow path was
still enough to reduce the speedup to 1.0743x.

This confirms both sides of the stop-loss decision:

1. A longer useful block contains enough latency headroom to exceed 1.10x.
2. The current way of making that block bitwise canonical consumes the
   headroom.

## Next structural revision

The next candidate is a shape-invariant or numerically certified verifier, not
another draft model.

The shape-invariant design treats draft position as an outer parallel grid but
preserves the same per-token reduction order as the ordinary `q=1` Target path
for attention, projections, and logits. Its algorithmic target is:

```text
arriving exact KV + sparse-KV draft block
                 |
                 v
canonical-shape parallel verification
                 |
       +---------+---------+
       | all positions safe| ambiguous/rejected
       v                    v
commit verified IDs    bounded q=1 repair
```

A certification alternative computes a conservative logit-error envelope. A
token can be committed only when its top-1 margin exceeds twice that envelope;
otherwise it takes the canonical repair path. Unlike ordinary speculative
acceptance, the certificate is about equivalence to the declared canonical
execution, not agreement with a numerically different block kernel.

Before implementing a full kernel, the next experiment must localize the first
layer and operator where the three failing requests diverge, measure logit
margins versus numerical error, and estimate the fraction of requests requiring
repair. The new candidate advances only if an N=64 oracle `g=8` run achieves
64/64 same-stack IDs and at least 1.10x total speedup, followed by a 32-token
continuation check to detect representation drift.

## Reproduction surface

- `build_oracle_drafts.py` creates exact-prompt same-stack oracle artifacts.
- `run_oracle_ceiling_arm.py` launches one fresh two-H200 LMCache arm and cleans
  up only its own process groups.
- `analyze_oracle_ceiling_gate.py` enforces completed inputs, output equality,
  bootstrap saving, and the 1.10x gate.
- `benchmark_pd_packets.py` now writes an atomic per-request checkpoint so a
  later stream timeout does not erase completed evidence.

Machine-readable frozen evidence is in
`configs/iclr2027_oracle_verifier_ceiling_v1.json`. Raw multi-gigabyte runtime
traces remain under the ignored `outputs/` tree.
