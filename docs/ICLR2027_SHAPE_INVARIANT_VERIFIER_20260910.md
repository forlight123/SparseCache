# SparseCache: Shape-Invariant Verifier Pivot

Date: 2026-09-10

## Decision

The real LMCache pipeline now passes the original short-horizon oracle gate,
but it fails the stronger continuation gate. This is a useful structural
result, not an ICLR-ready result.

We stop searching attention backends and speculative horizons. The next
mainline is a **shape-invariant block verifier**: preserve the canonical `q=1`
floating-point reduction order inside every token row while parallelizing
across draft positions.

## Experimental contract

- Real layerwise LMCache P/D deployment using NIXL CUDA-IPC.
- Qwen3-8B BF16 on two H200 NVL GPUs: GPU 0 is P and GPU 1 is D.
- 64 immutable, length-stratified QMSum packets; 2,560--8,192 prompt tokens.
- Greedy decoding, one request at a time, fresh servers per arm.
- The proposal is the same stack's prerecorded ordinary Target trajectory at
  zero draft cost. It is an optimistic systems/verifier ceiling, not a
  deployable drafter.
- Losslessness is tested more strictly than the usual exact-arithmetic claim:
  token IDs must equal a fresh ordinary `q=1` trajectory from the same backend.

## Stage result 1: the short gate now passes

For an eight-token verifier horizon and nine output tokens, forcing
`FLASHINFER`, enabling request-scoped prefix caching, and replaying only a
rejected block gives:

| Metric | Result |
|---|---:|
| Same-stack exact trajectories | 64 / 64 |
| Mean accepted suffix | 6.656 tokens |
| Acceptance distribution | 60x7, 2x2, 2x1 |
| Canonical replays | 4 / 64 |
| Mean prefix-cached replay | 134.62 ms |
| Observe sandwich | 869.55 ms |
| Inject | 785.87 ms |
| Speedup | **1.1065x** |
| Mean saving | 83.68 ms |
| Saving bootstrap 95% CI | [69.12, 96.60] ms |

This is the first cell satisfying all three predeclared gates: 64/64 exact,
at least 1.10x total speedup, and a strictly positive saving CI.

Two implementation details were necessary:

1. A unique vLLM `cache_salt` isolates each logical request, while P, D, and
   its optional replay share the same salt. This permits cheap same-request
   replay without accidental cross-request prefix hits.
2. LMCache 0.5.4rc5's empty layerwise retrieval protocol was completed locally
   so an already-vLLM-cached prefix cannot dereference an undefined consumer.

## Stage result 2: accepted representations are not canonical

We then generated 33 total tokens: one P token, one initial D token, one
verified block, and ordinary continuation. The result failed:

| Metric | Result |
|---|---:|
| Same-stack exact trajectories | 58 / 64 |
| Representation-drift requests | 6 / 64 |
| First divergent output positions | 13, 13, 16, 19, 29, 31 |
| Drift blocks reported fully accepted | 6 / 6 |
| Observe total | 1235.20 ms |
| Inject total | 1154.57 ms |
| Speedup | 1.0698x |

The initial block token IDs were correct. Its BF16 KV representations were
nevertheless produced by a different execution shape and later changed a
near-tied argmax. Rejection feedback cannot detect this class of failure.

This distinguishes two obligations:

- **token correctness:** the verified block contains canonical token IDs;
- **state correctness:** the accepted block leaves the exact canonical Target
  state for all subsequent decoding.

An actually lossless system needs both.

## Stop-loss results

Changing only execution parameters did not establish a safe cell:

- `FLASH_ATTN` repaired four of the six `FLASHINFER` drift cases, but two
  remained.
- `TRITON_ATTN` plus rejection replay was exact on eight of a nine-case hard
  union, but a different fully accepted request drifted.
- Shortening the `FLASHINFER` horizon was non-monotone: `g=6` was exact on 4/6
  hard cases and `g=4` on only 2/6.

Therefore no backend or horizon is a proof of state correctness. We do not run
more cells of this form.

## Why a margin-only fallback also stops

The diagnostic run recorded top-2 logprob margins for all 64 continuations.
Every first divergent token had a margin from 0 to 0.25. However:

| Threshold | Drift recall | Requests sent to repair |
|---|---:|---:|
| 0.00 | 3 / 6 | 19 / 64 |
| 0.25 | 6 / 6 | 46 / 64 |
| 0.50 | 6 / 6 | 52 / 64 |

Threshold 0.25 is empirically complete on this sample but has 40 false
positives. More importantly, a measured margin is not a certified upper bound
on floating-point state error. Margin-only replay is neither efficient nor a
proof, so it remains a diagnostic.

## New algorithm: shape-invariant block verification

Let `F_1` denote the serving stack's canonical one-token transition and
`F_g` a normal `g`-token block forward. Exact arithmetic makes the two
equivalent on a correct draft, but their BF16 kernels use different tilings and
reduction trees. The new verifier `F_g^SI` must satisfy:

`State(F_g^SI(x, d_1...d_g)) = State(F_1^g(x, d_1...d_g))`.

The proposed implementation is layer-major:

1. At each Target layer, launch a group of `g` independent `M=1` projection
   operations. Each group member uses the same primitive and reduction order
   as canonical decode.
2. Evaluate causal attention for query `j` against prefix length `n+j` using
   the canonical segmented reduction. Queries are independent once that
   layer's draft K/V rows exist, so they can run concurrently.
3. Apply normalization and MLP as grouped `M=1` primitives, again preserving
   the canonical per-row order.
4. Write these canonical rows directly into the authoritative paged KV cache.
   Accepted tokens therefore need no later representation refresh.
5. Full-KV verification remains the only commit point; residual KV transfer
   continues on its copy stream while the sparse drafter runs.

### Inductive correctness argument

Assume the prompt state and all earlier accepted token rows equal canonical
`q=1`. At layer `l`, every grouped primitive for token `j` has the same inputs,
operation order, and output rounding as canonical decode. Its Q/K/V,
attention, residual, normalization, and MLP output are therefore identical.
Induction over token position and layer makes all final logits and stored KV
rows identical. The ordinary speculative acceptance rule then preserves the
canonical greedy trajectory, including future continuation.

The research contribution is not merely deterministic execution. It is a
parallel schedule that keeps the per-token numerical semantics fixed while
exposing concurrency across the token dimension, and co-schedules that work
with progressive KV arrival.

## Next gates

We implement the smallest kernel prototype before touching the drafter again.

1. **Primitive gate:** 1,000 random and real Qwen states; bitwise-equal K/V and
   logits to `q=1`; at most 1.5x standard-block latency and at most 0.45x eight
   serial `q=1` steps.
2. **Hard-set gate:** 9/9 exact 32-token trajectories.
3. **N=64 gate:** 64/64 exact 32-token trajectories and at least 1.08x total
   speedup under the same real LMCache P/D contract.
4. Only after these pass do we resume progressive sparse-KV drafter training.

Raw run artifacts are under `outputs/progressive_kv/iclr2027_20260910/` and
remain intentionally gitignored. The machine-readable stage summary is
`configs/iclr2027_shape_invariant_verifier_v1.json`.
