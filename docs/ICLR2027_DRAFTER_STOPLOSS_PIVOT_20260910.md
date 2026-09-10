# Drafter stop-loss result and lossless portfolio pivot

Date: 2026-09-10

## Decision

The learned sparse-Target-KV adapter line is stopped.  Two structurally
different revisions failed the same frozen self-autoregressive selection gate;
more optimizer steps are not authorized as a response.  The next integrated
candidate is a budget-aware portfolio of pretrained 0.6B/4B drafters plus the
existing layer-ready exact Target verifier.

This is not yet an ICLR-ready result.  It is a mechanism-selection milestone.
The positive 4B result is offline, uses one Target model and three task
families, and has not yet passed paired live P/D latency, two-node, sampling,
multi-request, or resource-normalized gates.

## Frozen protocol

- Target: Qwen3-8B BF16 immutable packet trajectories.
- Draft horizon: `g=8`; greedy exact-token accepted prefix.
- Adapter training: six LongBench task families, 64 packets each.
- Selection: QMSum records 64--127 only.
- Untouched confirmation packets were not opened after the adapter gate failed.
- The pretrained-model screen uses QMSum 0--63, MultiFieldQA 0--63, and
  Qasper 0--63, 192 requests total.

## Sparse Target-KV adapter negative result

The first revision writes a 330,240-parameter head-wise residual into five
selected Qwen3-0.6B cache layers.  Three survival-discount cells were trained
for 300 steps.  QMSum base acceptance was 2.8906; the cells reached 2.9063,
2.6719, and 2.7500.  The best paired gain was only +0.0156 with 95% CI
[-0.3438, +0.4219].

The one allowed structural revision adds an explicit memory read instead of
mutating old cache entries.  Five 0.6B decoder blocks cross-attend directly to
five sparse Target-KV layers through 10.66M trainable parameters; both language
models remain frozen.  Three seeds were trained for 600 steps.  Their last-50
teacher-forced prefixes were 5.44, 5.38, and 5.88, but their real QMSum greedy
accepted prefixes fell to 2.5625, 2.3281, and 2.5625.  This teacher-forcing to
free-running gap is precisely why the frozen gate exists.

The failure is structural evidence, not proof that Target KV contains no
useful information.  It shows that these small, LongBench-trained feature
alignment modules do not generalize enough to justify further main-line
budget.  The adapter implementations and zero-KV controls remain as negative
baselines.

## Pretrained drafter result

Qwen3-4B is run by vLLM on the exact prompt plus the authoritative P seed.

| Task | Requests | Mean accepted / 8 | 95% CI | p95 total draft |
|---|---:|---:|---:|---:|
| QMSum | 64 | 4.703 | [3.969, 5.453] | 231.10 ms |
| MultiFieldQA | 64 | 5.016 | [4.313, 5.719] | 231.37 ms |
| Qasper | 64 | 3.828 | [3.109, 4.563] | 230.32 ms |

The macro mean is 4.516 accepted tokens.  The corresponding 0.6B macro mean
is 3.464 with a maximum task p95 of 52.68 ms.  These two models form useful
slack tiers rather than a single universally optimal drafter.

The live protocol also starts a `seed + g` 4B branch concurrently with P.
Without fallback, the exact P seed matches that branch on 98.4% of QMSum,
70.3% of MultiFieldQA, and 51.6% of Qasper requests.  A miss never submits the
wrong branch: it triggers a prefix-cached `prompt + exact P seed` request and
then uses the ordinary full-Target verifier.  The branch-only acceptance
numbers are therefore not reported as final portfolio quality.

## Losslessness invariant

Let `q` be any proposal policy, including a wrong or task-adaptive small model,
and let `T` be the single immutable complete-KV Target transition.  Greedy
verification accepts the longest proposal prefix equal to repeated `T`
transitions and obtains the first mismatch/bonus token from `T`.  Induction on
the accepted prefix gives the same committed sequence as ordinary greedy
Target decoding.  Drafter choice, KV arrival time, and fallback affect cost and
acceptance only; no draft token is externally committed.

Sampling remains outside the current claim until the runtime implements the
standard probability-ratio acceptance rule.  "Lossless" below therefore means
the validated greedy endpoint plus the standard speculative distributional
construction, not an unimplemented sampling claim.

## Slack-priced portfolio algorithm

For candidate model `m`, horizon `g`, remaining layer-ready schedule `R_l`,
draft finish `D_(m,g)`, per-layer verifier costs `c_l(g)`, ordinary Target
token interval `tau`, and expected verified progress `U_(m,g)`, define

```text
F_-1(m,g) = D_(m,g)
F_l(m,g)  = max(F_(l-1)(m,g), R_l) + c_l(g)

Value(m,g) = E[U_(m,g)] * tau
             - (F_last(m,g) - R_last)
             - lambda * ResourceCost(m,g).
```

The controller chooses `(off, 0)`, `(0.6B, g)`, or `(4B, g)` with maximum
lower-confidence-bound value subject to a conservative no-overrun constraint.
The current offline tiers suggest 0.6B for short slack and 4B only once the
available request-to-verifier window exceeds roughly 231 ms.  These are
measured operating points, not hard-coded universal thresholds.

The seed branch introduces a second stopping decision.  Once both the 4B root
and exact P seed are known, retain the in-flight branch on equality; otherwise
cancel it and launch the prefix-cached conditioned branch.  This is a
one-step optimal stopping problem under measured cancellation and restart
costs.  The current non-streaming live bridge implements the same semantics but
waits for the whole first branch; streaming root synchronization is the next
latency optimization.

## Live gate result

The implemented bridge performs:

```text
client prompt -> P Target prefill/seed ---------> exact KV layer stream -> D
              -> 4B sidecar seed branch --match?-- proposal block -------> D
                                           \--miss: cached seed repair ---/
D: proposal + arriving exact layers -> one immutable full Target verifier
```

The pre-registered observe--inject--observe gate was run on 64 requests with
fresh P, D, proxy, and 4B sidecar processes for every arm.  Packet token IDs
were submitted directly.  The third H200 was used only by the 4B sidecar, so
its resource cost is explicit rather than hidden by a monolithic control.

| metric | result | gate |
|---|---:|---:|
| Three-arm live token-ID equality | 64/64 | 64/64 |
| Accepted injected suffix | 3.953/7 | at least 3.5 |
| Positive-acceptance requests | 49/64 | diagnostic |
| Concurrent seed-branch hits | 63/64 | diagnostic |
| Draft available before FullReady | 64/64 | 64/64 |
| Sandwich observe total | 839.684 ms | -- |
| Inject total | 808.994 ms | -- |
| Total saving | 30.690 ms, CI [21.814, 39.479] | CI above zero |
| Total speedup | **1.0379x** | **at least 1.10x** |
| TTFT change | +3.192 ms, CI [+1.552, +5.289] | diagnostic |

The live vLLM outputs differ from the Transformers-built immutable packet on
the same nine request ordinals in all three arms.  This is a backend-numerics
audit result, not a speculative mismatch: all three live token sequences are
identical on 64/64.  A losslessness claim therefore uses same-stack ordinary
Target output as its executable reference and retains the HF packets only as
immutable input and cross-backend diagnostics.

The portfolio fails exactly one pre-registered criterion: 1.10x speedup.  It
is stopped as the ICLR main line before co-location or broader task scaling.
The positive 30.7-ms saving remains a mechanism result, but a dedicated third
GPU for 3.8% latency improvement is not a publishable system tradeoff.

This is also evidence that another round of drafter training is the wrong next
move.  Requests with all seven injected suffix tokens accepted save only 52.3
ms on average in this run; requests with zero accepted tokens add 11.7 ms.  At
the current eight-output endpoint, even the empirical perfect-acceptance cell
is only about 1.066x, below the fixed gate.  The next structural candidate must
remove verifier/control cost or amortize one verification over a materially
larger useful block.  It must first demonstrate a measured upper bound above
1.10x; otherwise it is rejected without training.
