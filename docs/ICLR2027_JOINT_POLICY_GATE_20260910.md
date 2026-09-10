# SparseCache joint P/D policy gate

Date: 2026-09-10

## Outcome

There is a real lossless-overlap mechanism signal, but there is not yet an
ICLR-ready main result. The corrected identical-request screen initially
selected `g=6` as the leanest fixed sparse-D operating point and saved 83.34
ms/request at a paced 100-Gbps handoff. The larger 90-request replication now
selects `g=8`: after replacing two independently confirmed infrastructure
stalls with explicit single-request remeasurements, it saves 82.52 ms/request,
95% CI [69.39, 95.17], versus 76.41 ms for `g=6`. The paired `g=8-g=6`
advantage is only 6.11 ms, CI [0.74, 11.57].

That last result is a **measurement-repeatability gate**, not a deployable
controller result: it reuses the same request identities. A causal
minimum-margin controller selected on the old pilot did not generalize to the
held-out requests. The new 90-request sweep therefore trains a tiny stopping
policy on 45 requests and evaluates it unchanged on 45 disjoint requests.

The larger strategic finding is that the earlier exact P-runahead model was
wrong. In a live LMCache 1P1D service, four P tokens are 7.05 ms slower end to
end than the standard one-token handoff and delay TTFT by 52.53 ms on 32 paired
chunk-aligned requests. Sparse-D therefore remains the promising treatment;
P-runahead remains a mandatory control, not a co-equal main algorithm.

## Corrected 30-request protocol

- Qwen3-8B BF16 Target, greedy decoding, fixed 32-token timing horizon.
- RULER QA2 16K, sampled pool 150, seed 20260941, offsets 30--59.
- Identical request IDs and order for all four horizons.
- Query-ranked 10% document-page Anchor with original global RoPE positions.
- One serial paced 100-Gbps wire, followed by real pinned-CPU-to-HBM copies.
- One immutable full-KV verifier; no tentative token is externally exposed.
- Full-transfer and pipeline order alternates by request.
- 10,000-request bootstrap; no latency observation is deleted.

The September 9 horizon table accidentally reused a `g=6` artifact sampled
from a 200-row pool while the other horizons used a 150-row pool. Five of 30
IDs differed. That table is now marked historical and is superseded here.

## Fixed-horizon result

| horizon | accepted sparse prefix | mean saved latency | bootstrap 95% CI | faster requests |
|---:|---:|---:|---:|---:|
| 4 | 3.567 | 57.58 ms | [51.52, 63.14] | 30/30 |
| **6** | **4.767** | **83.34 ms** | **[68.97, 96.25]** | **29/30** |
| 8 | 5.767 | 81.27 ms | [59.04, 102.43] | 27/30 |
| 12 | 7.933 | 34.54 ms | [-4.52, 70.88] | 17/30 |

`g=6 - g=8` is only +2.07 ms, 95% CI [-7.52, 11.64], with 15/30
request wins. They are statistically tied; `g=6` is preferred because it
performs less speculative work. Longer accepted blocks alone do not predict
latency: `g=12` accepts the most tokens and is the worst fixed choice.

All four arms match the baseline's exposed EOS-truncated output on 30/30
requests. Post-EOS fixed-horizon timing trajectories match on 27, 29, 30, and
28 requests for `g=4/6/8/12`, respectively. These synthetic post-EOS tokens
are charged for latency but are not user-visible quality differences.

## 90-request scale replication

The scale run uses the same sampled pool and seed, with offsets 60--149. Two
paired measurements on request 64 are clear bidirectional infrastructure
stalls: `g=6` records -1692.69 ms because its pipeline pauses, while `g=8`
records +1550.06 ms because its full control pauses. Independent exact-request
remeasurement gives +44.38 and +134.70 ms. Raw observations are retained; the
following adjudicated sensitivity replaces exactly those two named cells and
records both old and new values in the machine-readable result.

| horizon | fixed-horizon accepted | accepted before draft EOS | mean saved latency | bootstrap 95% CI | exposed output equality |
|---:|---:|---:|---:|---:|---:|
| 4 | 3.422 | 2.911 | 48.86 ms | [43.11, 54.01] | 88/90 |
| 6 | 4.611 | 3.378 | 76.41 ms | [66.81, 85.16] | 89/90 |
| **8** | **5.633** | **3.567** | **82.52 ms** | **[69.39, 95.17]** | **90/90** |
| 12 | 7.578 | 3.811 | 37.27 ms | [16.65, 57.46] | 89/90 |

The first acceptance column includes artificial post-EOS tokens generated only
to keep a 32-token timing horizon. It must not be presented as ordinary
speculative acceptance. The pre-EOS column is the user-meaningful metric; on
this short-answer task `g=8` is 3.567. Draft EOS is already reached in 81/90
`g=8` requests, so RULER QA2 cannot support a serious long-horizon controller
claim. A long-form task with at least 32 natural output tokens is mandatory.

The offline HF verifier is also not the final exactness implementation. Its
`g=4/6/12` arms have 2/1/1 exposed greedy divergences caused by BF16 execution
shape, whereas `g=8` happens to have 90/90 equality. This entire scale table is
marked diagnostic for bitwise exactness. The live vLLM verifier and the audited
shape-invariant kernel are the valid routes to the lossless claim.

## Finite-candidate headroom and independent retiming

Choosing the best of `g={4,6,8,12}` separately for every request on the first
measurement gives 102.39 ms mean saving, 95% CI [86.89, 116.89]. Its advantage
over the best fixed `g=6` is 19.05 ms, CI [11.72, 28.07]. The oracle selects
`g=4/6/8/12` for 5/9/11/5 requests.

This maximum is biased upward by construction. To measure how much of it is
stable rather than timing noise, the selected action for each request was
frozen and all four actions were independently retimed on rotated GPUs. The
frozen selection saves 90.12 ms and remains +12.09 ms over the
source-selected fixed `g=6`, CI [1.25, 20.59]. The independently fastest
action repeats on 22/30 requests.

The second `g=8` run contains one 3.04-second positive-savings point caused by
a 3.97-second full-path stall; its transfer deadlines are normal. A third
measurement of that exact request gives 4.83 ms, confirming infrastructure
jitter. The raw point remains in the formal no-deletion result. It does not
affect the frozen-versus-`g=6` comparison because the source selector chose
`g=4` for that request. As a transparent sensitivity only, removing that one
point changes second-run fixed `g=8` from 184.79 to 86.27 ms.

## Why the first confidence controller is rejected

An exploratory one-threshold rule chose `g=8` when the early minimum sparse
logit margin was at most 1.9375 and `g=12` otherwise. On the pilot it appeared
to beat `g=6` by 24.93 ms, CI [4.35, 45.68]. Frozen on the held-out split, it
instead loses 10.69 ms, CI [-25.37, 2.52]. First-block minimum margin alone is
not a sufficient horizon predictor.

The replacement controller is deliberately constrained and causal. It starts
at the shortest admissible horizon, observes only margins already computed,
and applies backward-trained thresholds to decide `stop` or `continue` at
each gate. Threshold candidates are fixed in advance, every nonconstant leaf
needs at least ten training requests, and the final 45-request split is never
used for selection. Failure means adaptive horizon is removed, rather than
expanded into a higher-capacity model after seeing validation.

That failure occurred. The learned rule continues past `g=4` only when the
minimum pre-EOS sparse margin is at least 2.0, then always stops at `g=6`. It
selects `g=4/g=6` on 32/13 of the 45 evaluation requests and loses 16.31 ms to
the training-selected fixed `g=6`, CI [-23.02, -9.80]. It also selects one arm
with a numerical exposed-output mismatch. Minimum margin is therefore removed
from the proposed contribution. EOS-aware prefix validation remains in the
analysis so fixed-horizon post-EOS fillers cannot leak into future controllers.

## Exact P-runahead control: model rejected, live control completed

The following was the optimistic model that motivated the control. It uses
the full-path per-token decode cost, overlaps exact P decoding with the
residual transfer, and omits generated-KV handoff and P/NIC contention.

| extra exact P tokens | saving vs full transfer | P GPU occupancy | completion minus sparse `g=6` |
|---:|---:|---:|---:|
| 1 | 45.43 ms | 21.90 ms | +37.91 ms |
| 4 | 111.12 ms | 87.59 ms | **-27.78 ms** |
| 8 | 198.71 ms | 175.17 ms | **-115.37 ms** |
| 16 | 267.22 ms | 350.35 ms | **-183.87 ms** |

A negative final column means P runahead completes sooner. At low load, four
exact P tokens appeared to dominate the current sparse-D implementation. The
live result rejects that conclusion.

The measured control runs Qwen3-8B BF16 on one H200 P and one H200 D, LMCache
NIXL/UCX, a 24-GiB receiver arena, greedy eight-token output, disabled vLLM
prefix caching, two warm-up requests, alternating paired order, and integer-ID
comparison. For 32 requests whose prompt boundary is an LMCache 256-token
boundary (5,888--7,680 tokens), `k=4-k=1` is:

| metric | paired mean | bootstrap 95% CI | `k=4` faster |
|---|---:|---:|---:|
| TTFT | +52.53 ms | [51.25, 53.74] | 0/32 |
| total latency | +7.05 ms | [5.33, 8.65] | 3/32 |
| exact output IDs | 32/32 equal | -- | -- |

At a deliberately unaligned 7,800-token boundary, output IDs match only 7/8:
D retrieves 7,680 transferred tokens and recomputes the final 120-token prompt
suffix, which can change later BF16 greedy decisions. This is a useful
implementation theorem boundary: Target identity alone is insufficient for a
bitwise claim; the state boundary and numerical kernel must also be canonical.

On aligned prompts D still retrieves only the complete prompt chunks and
recomputes the small P-generated suffix. That cost is fully charged. The three
extra P tokens replace roughly three D decode steps but do not hide enough
additional transfer to pay for the handoff, invalidating the analytical model's
assumption that suffix production and transfer are free to overlap.

## Proposed algorithm: lossless bounded handoff

The wait/P/sparse-D equations below are retained as a control framework, not as
evidence that P runahead helps. Measure time from Anchor arrival at D. Let `R`
be remaining full-KV transfer
time, `n` the remaining output budget, `tau_D` Target decode time/token,
`C_P(k)` exact P cost for `k` extra tokens, `H(k)` suffix handoff cost,
`C_D(g)` sparse-D proposal cost, `V(g)` final verification cost, and `A_g` the
full-Target-accepted sparse prefix. Three candidate completion predictors are:

```text
T_wait       = R + n * tau_D
T_P(k)       = max(R, C_P(k) + H(k)) + (n - k) * tau_D
T_D(g)       = max(R, C_D(g)) + V(g) + (n - E[A_g | x]) * tau_D
```

The dispatcher does not minimize isolated latency alone. For action `a`, it
minimizes a queue-priced objective:

```text
J(a | x, q) = T_hat(a | x)
            + lambda_P(q) * GPU_P_work(a)
            + lambda_D(q) * GPU_D_work(a)
            + lambda_N(q) * network_bytes(a)
```

- low P pressure: select exact P runahead only if a measured, fully charged
  profile predicts a win; the current profile does not;
- high P pressure with residual wire slack: select sparse-D draft;
- no profitable safe candidate: wait for full KV;
- choose `k` or `g` only from profiled candidates whose work fits the overlap
  window and relevant TBT SLO.

The theoretical layer should be built around two statements:

1. **Endpoint invariance.** P-runahead tokens are sampled by the immutable
   Target. Sparse-D tokens are tentative and pass the ordinary full-KV Target
   acceptance rule before exposure. Therefore the output distribution equals
   Target-only decoding, independent of scheduler actions, assuming identical
   numerical Target kernels.
2. **Queue-aware optimality within the candidate set.** A drift-plus-penalty
   selector using P/D/network queue lengths as shadow prices can be analyzed
   for the usual `O(1/V)` utility gap and `O(V)` queue trade-off under bounded
   service estimates. This theorem is only a plan until the real multi-request
state machine and estimator-error bounds exist. Because measured P runahead is
dominated, this theorem is deferred until sparse-D wins under multi-request
load; it is not currently the shortest path to a paper result.

The greedy BF16 prototype still has an engineering exactness boundary. A new
scale-run `g=4` request diverges after the verified block because batched
verification and sequential decoding produce slightly different generated KV
representations. No unverified token was exposed, but bitwise equality is not
guaranteed across CUDA reduction shapes. The paper must use the real vLLM
Target verifier, a shape-invariant kernel, or a fully charged canonical replay;
it must not turn a logit-margin heuristic into a proof. The current real vLLM
observe--inject--observe path has 64/64 greedy output equality. An isolated
`shape_invariant_full` audit also obtains zero logit/KV error and 64/64 output
equality for a 15-token block, at 219.14 ms versus 475.16 ms for sequential
verification; integrating and charging that kernel in the P/D path remains
open.

## Novelty boundary after September 2026 search

- Lynx already performs progressive partial-KV speculation and exact final
  verification, splitting by quantization bits rather than token pages.
- SparseSpec-L already provides training-free sparse same-model drafting and
  adaptive speculation length outside the P/D handoff problem.
- SmartGen already selects and fetches important KV during disaggregated
  decoding, although its endpoint is an accuracy/latency trade-off.
- Kairos already makes load-aware P/D placement decisions by deflecting
  chunked prefill to D; FlowKV also has flexible role/load scheduling.
- Pallas already overlaps source-side decoding with proactive KV migration in
  a mobile-handover setting.

Consequently, neither “partial KV can draft,” “adaptive gamma,” nor generic
load-aware P/D scheduling is a defensible contribution alone. The remaining
paper hypothesis is narrower: **a lossless, bounded P-to-D state handoff can
choose between exact source runahead and partial-state destination speculation
using one resource-priced control law, and improve service-level latency
without changing Target outputs or permanently colocating phases.** This must
beat wait-only, Lynx-style progressive verification, sparse self-speculation,
prefill deflection, and exact P-runahead controls under matched resources.

## Immediate gates

1. Replace the second HF Target copy with a shared-weight paged sparse-attention
   proposer. It must bring the live direct-Target draft below the residual
   transfer slack while preserving at least three pre-EOS accepted suffix
   tokens on task-unseen data.
2. Run a natural long-form horizon gate; RULER QA2 terminates too early to
   identify `g>6` or validate an adaptive controller.
3. Integrate the standard vLLM verifier or the charged shape-invariant kernel,
   then repeat `observe -> inject -> observe` for at least 100 requests.
4. Replay a multi-request arrival trace and sweep P:D pressure only after the
   optimized sparse-D path beats wait-only in the isolated fully charged cell.
5. Repeat on at least a second task, context length, and model family, then run
   two physical nodes. Without these, the work remains a mechanism study.

## Artifacts

- Analysis: `experiments/analyze_joint_pd_policy.py`
- Causal controller: `experiments/analyze_adaptive_horizon_policy.py`
- Same-measurement result:
  `outputs/progressive_kv/iclr2027_20260910/horizon_policy_headroom_ruler16k_val_n30.json`
- Independent retiming:
  `outputs/progressive_kv/iclr2027_20260910/horizon_policy_replication_ruler16k_val_n30.json`
- Scale diagnostic with explicit stall adjudication:
  `outputs/progressive_kv/iclr2027_20260910/horizon_policy_headroom_ruler16k_scale_n90_adjudicated_diagnostic.json`
- Rejected causal controller:
  `outputs/progressive_kv/iclr2027_20260910/adaptive_horizon_margin_ruler16k_scale_n90_adjudicated_diagnostic.json`
- Live exact P-runahead control:
  `outputs/progressive_kv/iclr2027_20260910/lmcache_p_runahead_k4_aligned7680_n32.json`
- Raw results remain under the ignored
  `outputs/progressive_kv/iclr2027_20260910/` directory.
