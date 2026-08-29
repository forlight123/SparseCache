# SparseCache-PD stage report — 2026-08-27

## What is actually deployed

The current live test bed is real vLLM plus `LMCacheMPConnector` in a 1P1D
layout. The P worker runs on one H200 and performs the complete prompt prefill;
the D worker runs on a second H200 and consumes the transferred prompt KV.
LMCache CPU L1 is the handoff store. The two workers are on one host, while a
serialized controlled-link scheduler charges every exact KV byte at 25 Gbps.
This is a real LMCache/vLLM execution path with real CPU-to-GPU installation,
but it is not yet a physical two-node network result.

The primary controlled input is RULER QA-2 at exactly 65,536 prompt tokens,
greedy decoding, and one request at a time. P emits one hidden exact target
seed after prefill. D receives an atomic protected anchor, then uses the same
Llama-3.1-8B target weights to propose 8 or 16 internal tokens while exact
residual KV continues to arrive. No draft token is returned to the user. Once
all prompt KV is installed, D rewinds the tentative tail and performs exactly
one full-KV verification before producing the fixed 32-token response. Thus
`P length` is the full 65,536-token prompt; `D length` is not another prompt,
but a short tentative proposal horizon followed by verified target decode.

“Progressive” means exact BF16 prompt pages become visible to draft attention
only after their connector completion event. It does not mean mixed precision,
repeated intermediate verification, or early commitment. The exact policy is
`W=infinity`: complete KV is the one immutable verifier.

## Valid stage results

1. The state machine is real. Shared-prefill mechanism audits passed exact
   producer-seed pairing, 160/160 sparse forwards, 20/20 verifier batches, and
   full output equality across five schedules by two visibility arms.
2. Coarse progressive arrival is ineffective. On 30 paired 64K/25-Gbps
   requests, continuous masks changed useful accepted tokens in none of 150
   schedule-request pairs; every latency CI crossed zero.
3. Fine arrival does not rescue it. With 1% tranches and `gamma=16`, all 50
   continuous requests observed 6--8 visibility levels and 8 draft sequences
   changed, but useful acceptance changed in 0/50 fixed/continuous pairs. The
   100-bundle transfer added substantial fragmentation/control overhead.
4. Current page-only sparse draft is not economically cheap. In the valid
   trace-isolated 128K `n=30` crossover, 5% visibility addressed 6.44 GiB per
   eight-token draft instead of 127.75 GiB, yet took 168.92 ms versus 161.05 ms
   at 100% (`0.953x`; paired saved-time CI `[-16.25, 0.43]` ms). It fails the
   preregistered `>=1.10x` crossover gate.
5. The implementation bottleneck is understood. Reusing the compact block
   table removes repeated 2K-page construction and improves the earlier 5%
   implementation from 181.96 to 168.92 ms across separate balanced runs.
   Nsight then shows about 32 ms less attention-kernel work but 256 additional
   per-layer FA3 scheduling kernels plus CPU launch gaps. A compact AOT schedule
   reduced the extra GPU operations from 269 to 21, but its synchronous build
   cost about 40 ms and was rejected rather than hidden in the result.

## Decision relative to Lynx

There is no evidence that the current method is better than Lynx. Lynx and this
prototype share the broad exact `draft while residual arrives -> one final
verify` skeleton. Lynx makes its early representation cheaper through KV bit
precision; this prototype changes exact token/page visibility. The latter is
orthogonal and potentially composable, but the present page-only proposer is
slower and its multi-level visibility does not improve useful acceptance.
Consequently the defensible current contribution is a validated mechanism and
a falsified design point, not a superior end-to-end system.

## Research pivot

The next candidate is a two-level, two-dimensional exact anchor:

- prioritize protected tokens plus selected prompt pages for only an early
  subset of target layers;
- draft with both page-sparse attention and an early-exit/layer-sparse target
  path, so projections and MLPs become cheaper as well as KV reads;
- transfer every remaining layer/page in the background;
- retain the same one-shot full-layer, full-page verifier and exact endpoint.

The immediate selection experiment crosses draft depth `8/16/24/32` with page
visibility `2/5/10%`, context `64K/128K`, and `gamma=4/8/16`. It must report
full-model draft cost, accepted prefix, final exact equality, and the same
transfer/verification break-even equation. Only a point with at least `1.10x`
draft speedup and positive predicted end-to-end gain advances to full RULER,
LongBench, Qwen, concurrency, and real two-node testing.

## Authoritative artifacts

- Live shared-prefill screen:
  `outputs/progressive_kv/live_scheduling_screen_s5_n30_shared_prefill/llama31_8b/scheduling_selection_ruler_65536_original_bw25_n30_offset0/aggregate_eos_aware/summary.json`
- Fine-tranche screen:
  `outputs/progressive_kv/live_scheduling_screen_s5_t1_g16_n10/llama31_8b/scheduling_selection_ruler_65536_original_bw25_n10_offset0_tranche100bp/aggregate/summary.json`
- Valid trace-isolated 128K crossover:
  `outputs/progressive_kv/sparse_draft_crossover/crossover_llama31_8b_c131008_g8_r30_compactreuse_traceisolated/summary.json`
- Matching profiler trace:
  `outputs/progressive_kv/nsys_crossover_llama31_8b_c131008_g8_compactreuse_traceisolated.nsys-rep`
