# SparseCache-PD structural revision and stop-loss contract

Date: 2026-09-10

## Decision

Continue conditionally.  The new main path is:

```text
P exact prefill
  -> token-sparse Anchor arrives at D
  -> hidden direct-KV block draft is sealed
  -> exact full-precision KV continues in target-layer order
  -> Target layer l runs as soon as its complete exact KV is ready
  -> one immutable full-Target acceptance step
  -> only verified tokens become visible
```

This is not finite-W progressive commitment.  Sparse KV is never authoritative,
all original Target KV bytes eventually arrive, and no candidate token is
visible before the unique full-Target verifier.  The primary claim is the usual
speculative-decoding distributional losslessness.  Greedy experiments also
require 100% output-token equality.  Bitwise equality across different CUDA
shapes is a stronger implementation property and is not inferred from the
real-arithmetic proof.

The method removes three rejected branches from the main contribution:

1. exact P-side runahead, which is slower in the live LMCache control;
2. minimum-logit-margin horizon selection, which fails held-out evaluation;
3. multi-stage partial-KV commitment, which is lossy unless all decisions are
   revisited by the final Target.

## Structural difference from Lynx

Lynx transfers quantization Anchor bits, drafts while Residual bits move, and
starts its single parallel verification **after** the Residual is fully received
and dequantized.  SparseCache-PD instead uses token/layer-sparse exact Target KV
for the hidden proposal and removes the second barrier: exact Target layer `l`
may execute after the complete original KV for that layer arrives, while later
layers are still moving.  This is a narrower contribution than generic
"partial KV can draft" and must be evaluated against a wait-full verifier arm.

The Lynx timing boundary is stated in Sections 4.2--4.3 of
<https://arxiv.org/html/2607.01831v1>.  This audit does not establish that no
other concurrent system has layer-ready speculative verification; the broader
related-work search remains a paper gate.

## Exact dependency model

Let `D_g` be the time at which a `g`-token proposal is sealed, `R_l` the time at
which the full original KV for Target layer `l` is ready at D, and `c_l(g)` the
exact Target cost of verifying the block at layer `l`.  The layer-ready finish
time is

```text
F_-1(g) = D_g
F_l(g)  = max(F_{l-1}(g), R_l) + c_l(g).
```

The wait-full control is

```text
F_wait(g) = max(D_g, R_{L-1}) + sum_l c_l(g).
```

Therefore the verifier-over-transfer saving is exactly

```text
H(g) = F_wait(g) - F_{L-1}(g) >= 0.
```

When `D_g <= R_0`, drafting is entirely inside pre-verifier transfer slack and
cannot delay the first exact Target layer.  A candidate must be rejected before
launch when its conservative draft time exceeds that slack.  The full serving
objective remains

```text
Gain(g) = U_g * tau_D - V_g - [D_g - R_0]_+,
```

where `U_g` is exact Target progress delivered by the verified block and `V_g`
is verifier work not hidden by the layer-ready schedule.  This separates the
proposal value from the transfer/verify overlap and prevents double counting.

## New order-balanced 2x2 result

The new runner executes all four arms in one process on one H200 and cyclically
rotates their order inside each request/repetition:

```text
                       no draft             sparse draft
wait for all KV        control A            Lynx-style timing control B
layer-ready Target     control C            SparseCache-PD D
```

Protocol: Qwen3-8B BF16, QMSum development packets, nominal 10% Anchor, `g=7`,
100-Gbps paced single link, real pinned H2D, 64 requests x 3 paired repetitions.
The same requests, proposals, exact progress and 1.1566-GB mean payload are used
in all arms.  P prefill, NIC/RDMA and allocation are outside this mechanism
probe.

| Quantity | Result |
|---|---:|
| Layer-ready sparse path | 94.309 ms |
| Wait-full sparse path | 143.138 ms |
| Layer-ready speedup on sparse path | 1.519x |
| Hidden time, no-draft | 48.730 ms [47.472, 49.719] |
| Hidden time, sparse-draft | 48.830 ms [47.260, 49.910] |
| Difference in differences | +0.099 ms [-0.545, 0.681] |
| Mean exact progress per block | 3.3125 tokens |
| Mean accepted proposals | 2.3125 tokens |
| Mean draft time / residual slack | 14.525 / 91.129 ms |
| Draft overrun | 0 ms |
| Equal-progress saving of full method | 67.935 ms [55.927, 80.569] |
| Equal-progress speedup of full method | 1.733x |
| First-commit delta | -0.302 ms [-0.568, 0.179] |

The near-zero interaction is important: approximately 48.8 ms comes from
starting exact verification before FullReady in either draft arm, while sparse
drafting independently converts hidden time into 3.31 exact output tokens.  The
two effects are not the same latency counted twice.  Exact progress includes the
Target correction/bonus token; the actual accepted proposal prefix is only
2.3125 and remains below the frozen 3.5-token proposer gate.

The bandwidth table uses layer-ready arms with the same byte/request protocol.
The 25/50-Gbps target-start controls were run in separate processes; the
100-Gbps row comes from the within-process order-balanced experiment.  All are
preliminary because they remain below the final sample and exactness gates:

| Link | Ratio of mean latencies | Saving CI | Draft/slack |
|---:|---:|---:|---:|
| 25 Gbps | 1.226x | [71.01, 97.30] ms | 22.06/364.83 ms |
| 50 Gbps | 1.419x | [65.99, 91.54] ms | 22.81/182.37 ms |
| 100 Gbps | 1.720x | [55.93, 80.57] ms | 14.52/91.13 ms |

All three cells have one eager batched-versus-sequential mismatch request,
`qmsum:44`.  They therefore demonstrate systems headroom but fail the strict
greedy-output gate.  The order-balanced 100-Gbps run reproduces the same one
request, so it is not dismissed as timing noise.

## Frozen stop-loss gates

The machine-readable policy is
`configs/iclr2027_stoploss_v1.json`; the current evidence manifest is
`configs/iclr2027_round4_evidence.json`; and the evaluator is
`experiments/evaluate_iclr2027_stoploss.py`.

| Gate | Required evidence | Hard threshold | Stop action |
|---|---|---|---|
| Lossless structure | runtime contract | sparse state only drafts; full original KV; one immutable Target; no early exposure | stop immediately if violated |
| Layer-ready mechanism | at least 64 requests | copy+verify >=1.25x, positive saving CI, same-work bitwise equality | abandon layer-ready contribution |
| Useful proposer | at least 2 task-unseen long-form tasks and 100 requests; natural output >=32; `g>=8` | mean accepted proposal prefix >=3.5 | after 2 proposer revisions or 200 H200-hours, stop learned direct-KV proposer |
| Integrated proxy | 25 and 50 Gbps, at least 100 requests per cell | >=1.10x equal-progress, positive CI, first-commit upper CI <=5 ms, zero output mismatch, byte error <=0.1%, zero mean draft overrun | abandon ICLR systems claim or redesign verifier |
| Real deployment | physical two-node LMCache, at least 100 requests | >=1.10x end-to-end, positive CI, zero output mismatch | after 2 measured optimization rounds, stop the ICLR main line |
| Breadth | untouched evaluation | at least 2 Target models and 2 task-unseen tasks | no paper-scale claim until satisfied |

The evaluator has three semantics:

- `PASS`: the gate has enough eligible evidence and clears its threshold;
- `MISSING`: evidence is absent/too small/contaminated, or an allowed budget
  remains; this yields `ITERATE` rather than a false negative;
- `FAIL`: a mature gate misses its threshold, or a below-target proposer/live
  path exhausts its frozen budget; this yields `STOP_OR_REDESIGN`.

The current automatic verdict is `ITERATE`.  The lossless method contract and
the 1.572x all-layer copy+verify mechanism gate pass.  The proposer evidence is
excluded because its tasks informed training, its natural reference horizon is
only seven tokens, and `g=7` is below the frozen paper horizon.  The 25/50-Gbps
cells are below 100 requests and have one strict output mismatch.  Physical
two-node and second-model evidence do not yet exist.

## Next implementation, in dependency order

1. Change the live LMCache wire object contract from token-chunk-major/all-layer
   payloads to layer-addressable payloads and publish a completion event per
   Target layer.  The current object shape cannot unlock a complete layer early.
2. Dispatch the D request after the proposal is sealed rather than after the
   proxy's global `wait_decode_kv_ready` barrier.  vLLM's
   `wait_for_layer_load(layer_name)` is the intended synchronization point.
3. Keep the final attention/acceptance computation on the ordinary full Target
   path.  No HTTP/SSE token may be released until that pass completes.
4. Resolve the greedy numerical gate in the real vLLM verifier.  The unfused
   shape-invariant Python oracle is a correctness reference, not a viable path.
5. Only after the live runtime is correct, build task-unseen `g=8/16` natural
   trajectories and spend the bounded proposer-training budget.

This ordering avoids spending hundreds of GPU-hours on proposal quality before
the only paper-distinguishing data-plane schedule exists in the deployed
system.
