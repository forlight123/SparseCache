# SparseCache: Shape-Invariant Verifier Pivot

Date: 2026-09-10

## Decision

The real LMCache pipeline now passes both the original short-horizon oracle
gate and the stronger 32-token continuation gate. This validates the verifier
mechanism, but it is still an oracle ceiling rather than an ICLR-ready system:
the deployable sparse-KV drafter remains unsolved.

The successful pivot is a **shape-invariant block verifier**: preserve the
canonical `q=1` floating-point reduction partition inside every token row while
parallelizing across draft positions.

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

## Implemented FlashInfer schedule

The H200 implementation does not launch eight streams or run eight complete
model steps. At every Target layer it writes all draft K/V rows normally, then
represents the `g` attention queries as `g` logical decode sequences:

- every logical sequence references the same physical vLLM KV pages;
- sequence `j` has visible length `n+j`, enforcing the causal prefix;
- `fixed_split_size=64` pages gives every row the same 1,024-token reduction
  partition, independent of verifier batch shape;
- QKV/MLP work stays block-parallel; only attention metadata is expanded;
- page tables are expanded on GPU through vLLM's existing Triton copy kernel,
  avoiding a critical-path D2H synchronization.

The opt-in integration is in
`experiments/lossless_pd/lmcache_pd/shape_invariant_vllm.py`. It deliberately
fails closed outside a pure, uniform speculative-decode batch. Ordinary
prefill remains unchanged.

### Attention primitive sweep

On GPU2, FlashInfer 0.6.12, BF16 Qwen3 geometry (`Hq=32`, `Hkv=8`, `d=128`,
page size 16), we swept five prefix lengths (509--65,529), draft lengths
4/8/16, four fixed split sizes, and five random query seeds per cell:

| Fixed split | Exact cells | Geomean speedup vs serial `q=1` | Minimum speedup |
|---:|---:|---:|---:|
| 32 pages | 15/15 | 3.36x | 1.53x |
| 64 pages | 15/15 | 3.69x | 1.70x |
| 128 pages | 15/15 | 4.16x | 1.79x |
| 256 pages | 15/15 | 5.31x | 2.34x |

All 300 batched-versus-serial comparisons were bitwise equal. We selected 64
pages rather than the largest split because larger splits slow ordinary `q=1`
decode; at the deployed 2.5K--8K range, 64 pages preserves baseline latency
while giving `g=8` attention speedups from 6.81x to 3.31x.

The reproducible sweep is
`benchmark_shape_invariant_flashinfer.py`; raw results are in
`outputs/progressive_kv/iclr2027_20260910/sibv_flashinfer_microbench.json`.

## Continuation gate now passes

The same 64 immutable real-LMCache P/D requests were rerun with 33 output
tokens. The zero-cost same-stack Target oracle supplies one eight-token block;
the last 24 tokens are ordinary decode and therefore test accepted state, not
just immediate token acceptance.

| Metric | Optimized run 1 | Independent inject run 2 |
|---|---:|---:|
| Exact 33-token trajectories | 64/64 | 64/64 |
| Accepted injected suffix | 7.0 | 7.0 |
| Canonical replays | 0 | 0 |
| Observe sandwich | 1212.99 ms | 1212.99 ms |
| Inject | 1119.78 ms | 1121.06 ms |
| Speedup | **1.0832x** | **1.0820x** |
| Mean saving | 93.21 ms | 91.93 ms |
| Saving bootstrap 95% CI | [84.31, 101.82] ms | [85.92, 98.34] ms |

Both repetitions pass the pre-registered 64/64, 1.08x, and positive-CI gates.
The unoptimized Python D2H page-table path took 1129.19 ms (1.0742x); GPU
metadata expansion recovered 9.41 ms without changing any output token.

## Gate decision and next work

The attention primitive and N=64 continuation gates pass. The earlier
Transformers reference verifier also established bitwise-equal generated K/V
and logits on 64 real states when every row-sensitive primitive is held to its
`q=1` shape. The live vLLM test establishes the serving-stack token-trajectory
contract; direct extraction of vLLM's accepted KV rows remains a useful audit,
not a reason to weaken the external losslessness gate.

The next stage resumes drafter work with fixed stop-loss criteria:

1. replace the zero-cost oracle with a deployable direct sparse-KV proposer;
2. require mean accepted injected suffix at least 3.0 for `g=7` on task-unseen
   requests, with p10 accepted length at least 1;
3. retain 64/64 same-stack continuation equality through this verifier;
4. require at least 1.05x end-to-end speedup after charging drafter compute and
   memory, before scaling training or claiming an ICLR system result.

Raw run artifacts are under `outputs/progressive_kv/iclr2027_20260910/` and
remain intentionally gitignored. The machine-readable stage summary is
`configs/iclr2027_shape_invariant_verifier_v1.json`.
