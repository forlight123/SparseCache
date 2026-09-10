# SparseCache draft-horizon gate

Date: 2026-09-09

> **Protocol correction (2026-09-10).** The historical horizon table below is
> not a fully request-paired sweep: `g=4/8/12` used `sample_count=150`, whereas
> the reused `g=6` artifact used `sample_count=200`, changing 5 of the 30
> sampled requests. It must not be used to select `g=8`. A clean, identical-ID
> held-out rerun gives mean saved latency of 57.58, 83.34, 81.27, and 34.54 ms
> for `g=4/6/8/12`, respectively. The paired `g=6 - g=8` difference is only
> +2.07 ms with bootstrap 95% CI [-7.52, 11.64], so the two are statistically
> tied and `g=6` is the leaner fixed operating point. See
> `ICLR2027_JOINT_POLICY_GATE_20260910.md` for the corrected evidence.

## Decision

The learned five-layer KVShot-style drafter is rejected as the primary method.
Its task-unseen online accepted speculative suffix is 1.438 tokens/request after
removing the already-authoritative first Target token.  That result is below a
credible speculative-serving gate and must not be compared with papers that
report either accepted draft tokens or total Target-step advancement.

The direct Target self-draft from an exact 10% sparse KV view passes the initial
acceptance gate.  On the fixed 16K RULER QA2 screen, a proposal horizon of eight
accepts 5.833 sparse proposals per request and yields the best measured latency.
Because the current online vLLM bridge uses the first proposal to align with an
already-sampled D Target token, its expected injectable suffix is 4.933 tokens.
Longer proposals continue to increase absolute acceptance, but their serial
draft cost eventually dominates.

## Metric contract

Let `g` be the number of tokens proposed from sparse KV and `A` the length of
the prefix accepted by the immutable full-KV Target verifier.  This report uses:

- **accepted sparse-proposal prefix:** `E[A]`; the P seed is not counted.  The
  first sparse proposal is nevertheless an online alignment token in the
  current vLLM lifecycle because D samples that position before the custom
  proposer callback;
- **runtime-injectable suffix:** `E[max(A-1,0)]` for that one-token alignment;
- **acceptance fraction:** `E[A/g]`;
- **Target-step advancement:** approximately `E[A] + 1`, when the verifier's
  correction/bonus token is included;
- **zero/full acceptance:** `P(A=0)` and `P(A=g)`;
- **net latency:** full-transfer Target response time minus pipeline response
  time under paired order alternation.

The primary metric is `E[A]`, not the legacy online `2.438` value that included
one authoritative Target token.  That legacy learned-drafter result corresponds
to only `1.438` useful speculative suffix tokens.

## Experimental contract

- Target/drafter: Qwen3-8B in BF16; the same Target weights perform the sparse
  draft and full-KV verification.
- Data: 30 fixed RULER QA2 requests from the 16K normalized set, selected with
  seed 20260941.
- Prompt length: approximately 16K tokens; model-native Qwen chat template with
  thinking disabled.
- Sparse view: query-ranked 10% of document pages; static original RoPE
  positions.
- Transport: real pinned-CPU-to-HBM asynchronous copies, paced as one serial
  100-Gbps stream.
- Decode: greedy, fixed 32-token timing horizon; one immutable full-KV verifier;
  no token is exposed before verification.
- Measurement: paired baseline/pipeline order alternates by request; both paths
  are warmed; request bootstrap uses 10,000 resamples.
- Compute: three otherwise-idle H200 NVL GPUs ran `g=4`, `g=8`, and `g=12` on
  identical sample order. The historical `g=6` row used the same runtime
  protocol but a different sampling-pool size and therefore is not fully
  request-paired with those rows.

## Historical screening results (superseded for horizon selection)

| `g` | accepted `E[A]` (95% CI) | online suffix | acceptance | `P(A=0)` | first draft block | latency saved (95% CI) | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3.400 [2.867, 3.867] | 2.533 | 85.0% | 13.3% | 125.68 ms | 47.87 [31.85, 63.21] ms | 1.054x |
| 6 | 4.800 [4.200, 5.333] | 3.833 | 80.0% | 3.3% | 181.81 ms | 81.98 [59.38, 102.06] ms | 1.092x |
| **8** | **5.833 [4.833, 6.767]** | **4.933** | **72.9%** | **10.0%** | **218.42 ms** | **93.31 [62.45, 122.64] ms** | **1.112x** |
| 12 | 8.267 [6.633, 9.833] | 7.367 | 68.9% | 10.0% | 317.75 ms | 47.57 [-0.47, 93.58] ms | 1.068x |

The `g=8` result wins 25/30 paired requests.  The full-KV baseline and pipeline
have identical exposed task outputs and scores on all 30 requests for `g=6`,
`g=8`, and `g=12`.  The `g=4` run has one pre-EOS greedy divergence at a zero
Target logit margin; it is a finite-precision batch-shape tie, not an accepted
unverified token.  A strict bitwise claim therefore additionally needs a
canonical single-token replay on low-margin verifier events.

## Interpretation

1. The proposal horizon was not the cause of the old 1.x result.  The learned
   drafter itself failed task-unseen generalization.  Direct sparse-KV
   self-drafting supports a useful 5--8 accepted-token regime.
2. Maximizing accepted length alone is wrong.  `g=12` accepts 2.43 more tokens
   than `g=8`, but adds about 99 ms to first-block drafting and loses statistical
   separation in end-to-end latency.
3. The current optimum is workload-dependent.  A publishable scheduler should
   select `g` from the remaining transfer window, measured sparse-draft cost,
   verifier cost, and the learned survival curve `P(A >= j | state)`.
4. The 16K result is a stage gate, not an ICLR-ready paper result.  The direct
   Target path still needs real LMCache/vLLM P/D integration, multi-request and
   two-node experiments, additional tasks/models/context lengths, and a cheap
   sparse-attention/block-draft implementation that does not duplicate Target
   weights.

## Next gate

The live `g=8` integration smoke below is complete.  The next gate is to verify
engine-boundary committed token IDs and compare an observe--inject--observe
sandwich on at least 64 clean requests.  In parallel, replace the prototype HF
Target duplicate with shared vLLM weights and a paged sparse-attention kernel.
The algorithmic scheduler objective is:

```text
g* = argmax_g  E[A_g] * t_target_decode
                 - t_sparse_draft(g)
                 - delta_t_verify(g)

subject to t_sparse_draft(g) <= residual_transfer_window.
```

This objective should be estimated online and extended with a low-margin
canonical-replay cost for strict finite-precision output equivalence.

## Live LMCache/vLLM integration smoke

The direct Target mode was subsequently deployed on the real one-host LMCache
P/D stack:

- GPU 0: Qwen3-8B P plus LMCache NIXL sender;
- GPU 1: Qwen3-8B D, 24-GiB LMCache CUDA receiver arena, and a second HF
  Qwen3-8B sparse self-drafter;
- P transfers four `protected_uniform` Anchor chunks first for a 7,800-token
  request, then the remaining 27 chunks asynchronously;
- the online drafter generates nine sparse proposals; the first aligns with D's
  authoritative token and up to eight suffix tokens are injected into vLLM;
- vLLM verifies every injected suffix against complete transferred KV.

The service starts without OOM at vLLM GPU utilization 0.40 and retains capacity
for about 150K D-side KV tokens.  On the first request, the Anchor contains 888
tokens (11.385% actual), the real NIXL Anchor write is 2.12 ms, sparse drafting
is 370.08 GPU ms, the first proposal matches, eight suffix tokens are submitted,
and one is accepted.  The draft finishes 73.77 ms before its D-side handoff.

Two eight-request integration screens then give:

| packet family | requests | usable first alignment | accepted injected suffix/request | sparse draft GPU time | draft-ready lead |
|---|---:|---:|---:|---:|---:|
| QMSum | 8 | 6/8 | 2.750 | 308.87 ms | 47.14 ms |
| MultiFieldQA | 8 | 7/8 | 3.625 | 328.63 ms | 28.92 ms for the 7 handoffs |

For MultiFieldQA, one 1,536-token request completes the normal D path before the
sparse draft can be handed off; it is counted as zero, not silently omitted.
For the seven timely handoffs, the first sparse proposal matches in every case.
The engine's own speculative metrics report a mean acceptance length around
5.0 for injected batches, but that denominator excludes gate misses.  The
paper-facing metric must remain the lower all-request values above.

This smoke proves that the direct sparse Target consumes live LMCache Anchor
objects and reaches vLLM's real full-KV verifier.  It does **not** pass the paper
acceptance gate across workloads yet: QMSum is below three useful suffix tokens,
and MultiFieldQA is below four.  The next statistically valid live run must use
at least 64 clean requests and compare `observe -> inject -> observe`; before
that expense, the bridge should eliminate the one-token alignment loss and the
HF drafter should share Target weights or use a genuinely cheap sparse kernel.
