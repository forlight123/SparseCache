# SparseCache-PD related-work and baseline map

Status: working evidence map, updated 2026-09-10. “Code” means a public repository was
found, not that its results have been reproduced locally.

## 1. Positioning by bottleneck and correctness endpoint

| method | bottleneck attacked | incomplete remote KV used for generation? | cheap path | final target | endpoint class | public code | experiment role |
|---|---|---:|---|---|---|---:|---|
| Lynx | P/D transition transfer | yes | dense top-4-bit KV draft | reconstructed hierarchical INT8 `Q_full` | exact to compressed target; BF16 task-score equivalent | not found | nearest communication/speculation baseline; author-paper only |
| SmartGen | P/D transition plus later on-demand fetch | yes | sparse InfiniGen/HATA target attention | sparse selected KV; full cache arrives gradually | approximate task-quality endpoint | not found; author page link is empty | selective-transfer Pareto baseline |
| OasisKV | decode-node HBM capacity and tier traffic | yes in its P/D mode | EAGLE lookahead plus sparse prefetched attention | sparse working set over remote full cache | approximate, reported small quality loss | not found | memory-tier/throughput Pareto baseline |
| SparseSpec-L | local long-context draft cost | no | training-free same-model sparse attention | full target verifier | speculative exactness candidate | not found | nearest sparse self-draft baseline |
| Kairos | P/D queueing and transfer | no; it avoids transfer | chunked prefill deflected to D | ordinary decode on D | exact target | linked repository returned 404 | load-aware placement baseline |
| Pallas | active-request state migration | target reconstructs while source decodes | source exact decode plus target prefix reconstruction | migrated exact state | exact target | not found | source-runahead/migration concept baseline |
| SparseCache-PD | P/D transition transfer | yes, internally only | exact sparse-page same-model draft | original full BF16/FP16 KV | exact target after one verifier | local implementation | proposed method |
| MagicDec | local long-context draft cost | no | sparse-KV draft model | full target verifier | speculative exactness candidate | yes | local sparse-draft compute baseline |
| TriForce | local/offloaded long-context draft cost | no | retrieved sparse self-draft plus small model | full target verifier | speculative exactness candidate | yes | hierarchical sparse-draft baseline |
| QuantSpec | local long-context draft cost | no | hierarchical 4-bit KV/weight self-draft | high-precision target verifier | speculative exactness candidate | yes | closest quantized cheap-draft baseline |
| Dustin | long-context target verification cost | no | sparse target verification selected by draft/history | sparse verifier | approximate, negligible reported loss | not found | explicitly lossy verification control |
| SpecPV | long-context target verification cost | no | partial-KV verification with periodic full refresh | time-varying partial/full verifier | explicitly minor-loss endpoint | yes | explicitly lossy verification control |

The table separates “where the cache comes from” from “which cache the target
uses.” MagicDec, TriForce, QuantSpec, Dustin, and SpecPV do not remove the P/D
stage-transition wait. SmartGen and OasisKV do remove full-transfer blocking,
but directly execute sparse target attention rather than hiding tentative
tokens behind one original-full-KV verifier.

## 2. Primary sources and code status

- **Lynx**: [paper](https://arxiv.org/abs/2607.01831). The detailed evidence
  boundary is frozen in `docs/LYNX_BASELINE_PROTOCOL.md`.
- **SmartGen**: [paper](https://arxiv.org/abs/2607.28150). The authors' page
  displays a Code label with an empty URL as of the status date. The paper uses
  InfiniGen/HATA sparse attention, proactive/on-demand/background transfer,
  LongBench quality, and TTST; background “speculative transfer” is not
  speculative token verification.
- **OasisKV**: [paper](https://arxiv.org/abs/2608.08097). It is a vLLM-based
  system, but no author repository was found. It retains full KV in a lower
  tier and predicts/prefetches a sparse HBM working set with draft lookahead.
- **SparseSpec-L**: [paper](https://arxiv.org/abs/2607.27735). It already
  combines training-free same-model sparse attention with an adaptive
  speculation horizon, but does not address a P/D state handoff. No public
  author repository was found in the September 10 search.
- **Kairos**: [paper](https://arxiv.org/abs/2607.02043). It performs load-aware
  prefill deflection onto D and removes KV transfer for those requests; generic
  load-aware P/D placement is therefore not new. The paper points to a GitHub
  repository, but that URL returned 404 during the September 10 audit.
- **Pallas**: [paper](https://arxiv.org/abs/2608.16477). It overlaps ongoing
  source decoding and suffix-KV streaming with target-side prefix
  reconstruction for mobile handover. Exact source runahead during state
  movement is therefore prior art outside the P/D-serving setting.
- **MagicDec**: [paper](https://arxiv.org/abs/2408.11049),
  [code](https://github.com/Infini-AI-Lab/MagicDec).
- **TriForce**: [paper](https://arxiv.org/abs/2404.11912),
  [code](https://github.com/Infini-AI-Lab/TriForce).
- **QuantSpec**: [paper](https://arxiv.org/abs/2502.10424),
  [code](https://github.com/SqueezeAILab/QuantSpec). Its pinned source,
  environment mismatch, benchmark limitations, and missing repository license
  are frozen in `docs/QUANTSPEC_BASELINE_AUDIT.md`; it remains a compute-only
  reproduction pending the common BF16 equality gate.
- **Dustin**: [paper](https://arxiv.org/abs/2606.24957). It reports sparse
  verification gains at large batch/long context; no author repository was
  found.
- **SpecPV**: [paper](https://arxiv.org/abs/2512.02337),
  [code](https://github.com/TanZhendong/SpecPV). Its released checkpoints use
  EAGLE3 modules extended to 64K and therefore introduce model/checkpoint
  dependencies absent from SparseCache-PD.

Code availability must be rechecked at artifact-freeze time because several
papers are only weeks old.

## 3. Mandatory comparison groups

### Group A: same P/D blocking problem

1. monolithic original-BF16 transfer;
2. layer-wise original-BF16 transfer;
3. SparseCache-PD fixed S1;
4. SparseCache-PD continuous exact-page visibility;
5. Lynx under the evidence policy;
6. SmartGen and OasisKV if source becomes available, otherwise author-paper
   numbers in separate hardware blocks.

This group answers whether waiting for the P/D cache can be eliminated. Total
physical bytes, bytes before first compute, target precision, and whether a
generated token was verified must be explicit.

### Group B: cheap draft path

1. same-model dense draft;
2. same-model exact sparse-page draft;
3. MagicDec/TriForce-style sparse draft after full transfer;
4. QuantSpec hierarchical quantized draft after full transfer;
5. SparseCache-PD with the same proposal cap and verifier.

As of the 2026-09-10 audit, Lynx starts its parallel verifier only after the
Residual stream is fully received and dequantized (Sections 4.2--4.3).  The
current SparseCache-PD structural hypothesis instead starts exact Target layer
`l` when that layer's complete original KV is ready, while later-layer KV is
still in flight.  This precise timing distinction must be tested with a
wait-full/layer-ready 2x2 ablation and must not be broadened into a claim that
SparseCache invented partial-KV drafting.

This group answers whether the proposed sparse draft is actually cheap. It
must use matched model/checkpoint, context, batch, proposal length, and target
verifier. Network waiting is excluded or identically charged.

### Group C: relaxed verification Pareto controls

1. Dustin sparse verifier;
2. SpecPV partial/full verifier;
3. SparseCache-PD finite-W commitment;
4. SmartGen/OasisKV sparse target paths.

These rows report official task quality and distribution/token drift, never as
exact speedups. They test how much additional speed becomes available only by
weakening the endpoint.

## 4. Frozen claims that the comparison must not make

- Do not call task-score parity “lossless” or “BF16 exact.”
- Do not compare author-paper latency across Ascend, L20, A100, and H200 as a
  same-hardware speedup.
- Do not call SmartGen's background cache delivery speculative decoding.
- Do not claim OasisKV/Dustin/SpecPV solve stage-transition exactness.
- Do not claim MagicDec/TriForce/QuantSpec overlap remote P/D transfer unless a
  new integration explicitly implements and charges that overlap.
- Do not label a reimplementation official unless the author artifact and its
  reported quality/acceptance have been reproduced first.

## 5. Execution priority

1. Finish the clean three-arm SparseCache-PD core cells.
2. Reproduce QuantSpec and MagicDec/TriForce compute-only draft curves on the
   common Llama-3.1-8B 32K/64K cells.
3. Attempt SpecPV only as a lossy control after acquiring its exact checkpoint
   hashes and charging checkpoint memory/training provenance.
4. Recheck Lynx, SmartGen, OasisKV, and Dustin repositories immediately before
   paper artifact freeze; integrate official source if released.
5. Keep author-paper numbers visually and statistically separate whenever no
   same-testbed reproduction exists.
