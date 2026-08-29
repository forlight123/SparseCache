# Frozen non-contiguous layer-subset draft screen

Status: pre-registered selection experiment, not an end-to-end P/D claim.

The preceding 30-request experiment showed that taking only the first N frozen
Llama layers is fast but has near-zero speculative acceptance. This screen asks
whether retaining the target's final layers restores alignment without training.

## Frozen design

- Model/data: the same Llama-3.1-8B-Instruct and 30 RULER QA2 16K requests as
  the valid contiguous-depth screen.
- Verifier and committed seed: reused byte-for-byte from that valid target run.
- Candidates: uniform and first/last (sandwich) subsets at 8, 16, and 24 active
  layers, frozen in `configs/layer_subset_patterns_llama32.json`.
- Page visibility: 5% and 100%, exact physical masking through PROGRESSIVE_KV.
- Horizon: 8 greedy proposal tokens.
- Timing: one exclusive H200; every timed request must reuse the entire prompt
  prefix and compute only the committed seed plus proposal horizon.
- The full model's dense latency is frozen from the target run. This makes the
  screen suitable for branch selection; a winning pattern must later be rerun
  with interleaved same-process controls before a paper speed claim.

The path passes only if at least one cell reaches both 50% accepted draft tokens
and 1.5x speedup over the frozen dense control. Failure rejects untrained layer
subsets, after which a trained feature/exit draft is required.
