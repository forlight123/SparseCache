# Online lossless progressive-KV result

Date: 2026-09-09

## Result

SparseCache now executes the complete online chain on a real LMCache/vLLM P/D
deployment:

```text
P Target seed -> Anchor transfer -> direct sparse-KV draft
              || Residual KV transfer
full-KV Target t1 -> target-conditioned suffix repair -> full-KV verification
```

On 64 frozen QMSum requests, exact token-ID replay, greedy decoding, and eight
generated tokens, proposal injection preserves 64/64 outputs relative to both
an observe-only P/D control and paired monolithic vLLM. In a fresh-server
observe--inject--observe sandwich, injection reduces total request latency by
24.407 ms, request-bootstrap 95% CI [19.513, 29.671] ms saved, and wins 63/64
paired requests. TTFT changes by -0.635 ms, CI [-1.579, +0.353], as expected:
the pipeline accelerates verified tokens after the first Target token rather
than exposing an unverified early token.

This is the first stage result that simultaneously passes the real-transfer,
online-draft, full-verification, exact-output, and positive-latency gates. It is
not yet the final paper result: the current screen is one host, one request at a
time, one task family, greedy only, and uses a mechanism-training checkpoint.

## Algorithmic change: target-conditioned suffix repair

Let `A` be the arrived sparse Anchor view, `R` the Residual KV, `s` the exact P
seed, and `g` the proposal horizon. While `R` is moving, compute once:

```text
Z(A,s) = sparse hidden block
       + per-position Target-head top-K candidates and scores
       + causal reranker state after s.
```

The initial proposal `q1..qg` remains useful for diagnostics, but no longer acts
as an admission gate. Once the full Target produces authoritative `t1`, advance
the cached recurrent state with `t1` and cheaply produce a repaired suffix
`q2..qg = Repair(Z,t1)`. The unchanged full-KV Target then verifies this suffix.

This factorization moves all long-context attention and the single parallel
vocabulary projection before FullReady. Only small GRU/top-K reranking remains
after `t1`. In the measured run, the hot precomputation is 24.860 ms and repair
is 3.035 ms on GPU.

### Greedy losslessness

For deterministic greedy Target transition `F`, let the verifier start at the
authoritative state after `t1`. For an arbitrary repaired proposal sequence
`q2..qg`, accept the longest prefix satisfying

```text
q_j = argmax F(x_<j, full KV),  j = 2..a,
```

and let the Target supply the first mismatch or bonus token. By induction, each
accepted token equals the sequential full-KV Target token under the already
verified prefix. Therefore the committed sequence equals ordinary full-KV
greedy decoding regardless of Anchor quality or proposal quality. Conditioning
the proposal on `t1` changes only acceptance and cost, not correctness.

No draft token is committed at AnchorReady. The current implementation does not
claim sampling losslessness; sampling requires proposal probabilities and the
corresponding rejection/resampling rule.

## Experimental contract

Hardware and software:

- GPU 0: Qwen3-8B P node and LMCache NIXL sender;
- GPU 1: Qwen3-8B D node, LMCache receiver, and direct sparse-KV drafter;
- GPU 2: paired monolithic Qwen3-8B control;
- one host with three H200 NVL GPUs;
- Python 3.12, vLLM 0.23.0, PyTorch 2.11 + CUDA 13.0, NIXL 1.4.1;
- BF16 eager execution, tensor parallel size one, prefix caching disabled;
- LMCache 256-token chunks and a 24-GiB D-side registered CUDA arena;
- fixed `PYTHONHASHSEED=0`, P=`kv_producer`, D=`kv_consumer`.

Workload and decoding:

- 64 immutable packet prompts selected before online testing;
- prompt lengths: 56 x 8,192; 2 x 5,376; 2 x 6,656; 2 x 5,120;
  1 x 6,400; 1 x 2,560;
- exact packet token IDs submitted directly, with no input text round trip;
- one request at a time, greedy, eight output tokens;
- the same hidden-256, top-64, five-summary-task checkpoint in every mode;
- Target-KV layers `[1,9,17,25,33]`, proposal horizon `g=7`;
- nominal 10% `protected_uniform` whole-chunk Anchor; actual mean visible-token
  fraction 12.557%, minimum 10%, maximum 20%.

The earlier 7,800-token text replay changed both the packet boundary and some
token sequences. Its 59/64 monolithic equality and roughly 40% first-token
match are superseded. With strict tokens and native packet lengths, all outputs
match and the raw first-token match is 54/64 (84.375%).

The deployed API audit compares raw streamed output text byte for byte; the
online trace separately confirms that vLLM's full-KV speculative verifier
processed all injected blocks. A final paper run should additionally log emitted
token IDs at the engine boundary so text equality is not used as a proxy for
token equality.

## Online proposal and verification measurements

All 64 requests produce a live draft, a proposer handoff, an injected block,
and verifier feedback; the trace contains no live-draft error and no feedback
left pending.

| quantity | result |
|---|---:|
| raw first sparse proposal equals Target `t1` | 54/64 (84.375%) |
| target-conditioned accepted prefix | 2.438 tokens/request |
| useful suffix beyond authoritative `t1` | 1.438 tokens/request |
| accepted-prefix distribution | 1:24, 2:12, 3:10, 4:13, 5:4, 6:1 |
| hot draft GPU time | 24.860 ms |
| repair GPU time | 3.035 ms |
| repair wall time | 3.118 ms |
| draft-ready lead at D handoff | 236.245 ms |

The offline exact-runtime-layout replay predicts 2.484 mean
target-conditioned accepted tokens and 1.484 useful suffix tokens. The live
values, 2.438 and 1.438, are close, supporting the metadata/packing equivalence
between frozen packets and registered LMCache buffers.

## Latency comparison

Observe mode performs the same Anchor resolution, packing, sparse-KV forward,
and target-conditioned repair as inject mode but returns an empty proposal
block. This isolates the benefit of Target verification/acceptance from the
cost of the online mechanism. Fresh servers are launched in this order:

```text
observe before -> inject -> observe after
```

Each inject request is paired with the mean of the two matching observe
requests. Request-level bootstrap resamples the 64 prompt identities.

| metric | inject - observe sandwich | 95% CI | inject faster |
|---|---:|---:|---:|
| TTFT | -0.635 ms | [-1.579, +0.353] ms | 37/64 |
| total latency | **-24.407 ms** | **[-29.671, -19.513] ms** | **63/64** |
| monolithic-adjusted TTFT | -0.680 ms | [-1.788, +0.490] ms | -- |
| monolithic-adjusted total | **-24.187 ms** | **[-29.399, -19.336] ms** | -- |

The second observe run is 4.993 ms faster than the first in total latency,
CI [-6.227, -3.708]. The sandwich and monolithic difference-in-differences
therefore matter: a single before/after subtraction would confound injection
with this machine drift.

Absolute means are 593.613 ms for inject P/D, 620.517 ms for observe-before,
and 615.524 ms for observe-after. Their paired monolithic means are 318.476,
318.478, and 318.915 ms. SparseCache reduces the incremental post-first-token
work inside this P/D stack; it does not yet make the one-host P/D service faster
than monolithic vLLM.

## What is now established

- Real LMCache allocation, paged-KV gather, NIXL transfer, and D-side registered
  CUDA buffers are used; no bandwidth sleep substitutes for transport.
- Exact chunk ranges and prompt lengths cross the control plane, and five
  Target-layer Anchor views alias their registered storage.
- The direct Target-KV model actually runs from those live views concurrently
  with Residual movement; no EAGLE state or Target hidden state is an input.
- Target-conditioned suffix repair removes first-token admission failure.
- The full-KV Target verifies every injected proposal before commitment.
- Strict-input greedy raw output equals monolithic on 64/64 requests.
- Proposal injection yields a positive, statistically separated total-latency
  effect against an equal-cost observe-only control.

## Open gates before an ICLR-level claim

1. **External validity:** train on document-disjoint, task-diverse 8K+ prefixes
   and test on task-unseen LongBench/LongBench-v2-style workloads.
2. **Serving validity:** evaluate multi-request concurrency, P/D queueing,
   throughput, and actual two-node 25/50/100-Gbps links.
3. **Algorithmic breadth:** add probability-carrying proposals and prove/sample
   with exact speculative rejection; until then the online claim is greedy.
4. **Output audit:** log committed token IDs at the vLLM engine boundary in
   addition to the present byte-for-byte streamed-text comparison.
5. **Kernel cost:** fuse packing and top-K repair, and train for the exact
   target-conditioned runtime objective. The current 24.86 + 3.04 ms leaves
   significant headroom.
6. **Ablations:** Anchor fraction/layout, horizon, reranker K, unconditioned
   admission gate, random/contiguous/protected schedules, and no-draft controls.
7. **Baselines:** full LMCache P/D, ordinary decode, a direct-KV baseline such as
   KVShot, and a correctly scoped Lynx-style progressive precision baseline.

## Reproduction artifacts

The implementation is under `experiments/lossless_pd/lmcache_pd/`. The main
ignored local artifacts are:

```text
outputs/progressive_kv/iclr2027_20260909/
  lmcache_online_repair_tokens_observe_vs_monolithic_qmsum_n64_v17.json
  lmcache_online_repair_tokens_inject_vs_monolithic_qmsum_n64_v18.json
  lmcache_online_repair_tokens_observe2_vs_monolithic_qmsum_n64_v19.json
  lmcache_online_repair_tokens_sandwich_qmsum_n64_v19.json
  lmcache_online_repair_tokens_inject_n64_v18.jsonl
  lmcache_online_repair_tokens_inject_n64_v18_summary.json
  runtime_layout_summary5_repair_qmsum_n64.json
```

Generated packets, traces, and model checkpoints are intentionally excluded
from Git. The numerical protocol and results are fixed in this document.
