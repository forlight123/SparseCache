# QuantSpec baseline audit

Audit date: 2026-08-27 UTC
Repository: <https://github.com/SqueezeAILab/QuantSpec>
Audited commit: `661366739d89c14269c6204dddcb164727e8be7b`
Evidence label: **official source audited; local reproduction pending**

This document freezes what may be claimed before QuantSpec is used as the
cheap-draft baseline for SparseCache-PD. Finding the authors' repository is not
the same as reproducing its results.

## 1. Why this baseline matters

QuantSpec isolates the second half of the SparseCache-PD hypothesis: can a
same-model draft path be made cheap enough at long context? It uses a
hierarchical quantized KV cache (and optionally quantized MLP weights) for the
draft, followed by a higher-precision target verification pass. It is therefore
a compute-path comparator, not a P/D transition comparator: the public code
does not progressively expose exact remote KV pages while they are in flight.

The primary comparison must consequently have two panels:

1. **Compute-only cheap-draft panel**: full cache is already local; match model,
   prompt, proposal cap, target precision, output horizon, and batch size.
2. **End-to-end P/D panel**: charge the same remote BF16/FP16 bytes and paced
   link to every arm. Any new QuantSpec-plus-transfer integration must be named
   as our integration, not the authors' reported system.

QuantSpec must not appear in the exact-BF16 endpoint table until greedy output
equality to the common full-BF16 target has passed locally. Its paper-reported
acceptance or task quality is not evidence of that endpoint in this test bed.

## 2. Audited implementation facts

- The repository is a standalone gpt-fast-style engine, not a vLLM plugin.
- The documented environment is Python 3.11, CUDA 12.1, GCC 13.2, PyTorch
  2.4.0, Transformers 4.36.2, and NumPy 1.26.3.
- The current SparseCache vLLM environment uses a materially newer PyTorch and
  Transformers stack. A separate environment is mandatory.
- The model must be converted to the repository's single `model.pth` format.
- The benchmark and model import `marlin` unconditionally, including when
  optional weight quantization is disabled. The Marlin submodule/build must be
  resolved or the import path must be minimally patched and disclosed.
- The public benchmark evaluates at most ten dataloader batches, resets the
  accumulated measurements after each of the first three requests, uses custom
  timing, and does not emit paired request-level records or a full-BF16 token
  equality gate.
- Its built-in datasets are PG-19, MultiLexSum, and InfiniteBench. They do not
  directly match the frozen LongBench/RULER confirmation set.

No repository-level `LICENSE`, `COPYING`, or `NOTICE` file was found at the
audited commit. Some inherited source files refer to a separate license file
which is also absent. Therefore the source is **not vendored into this
workspace**. We may execute a temporary checkout for research reproduction,
but copying or modifying the implementation in-tree requires a clarified
license or explicit permission.

## 3. Required local reproduction gates

An official-source reproduction label is allowed only after all of these pass:

1. record repository and Marlin submodule commit hashes, environment lock,
   compiler, CUDA driver/runtime, GPU, and converted model checksum;
2. reproduce one authors' acceptance/throughput cell with their script and
   report the raw value, not only the claimed paper number;
3. add a non-invasive wrapper which accepts the frozen request IDs and emits
   one JSONL record per request;
4. use at least 100 paired latency requests per confirmation cell, warm both
   paths, alternate execution order, and retain every request;
5. emit draft time, verification time, accepted/proposed tokens, target steps,
   peak memory, total latency, and generated token IDs;
6. compare those token IDs against the common full-BF16 greedy target under the
   same tokenizer, prompt template, and fixed output horizon;
7. run the common compute-only contexts (16K, 32K, 64K where supported), then
   separately charge identical P/D link bytes if an integration is evaluated.

Until then the allowed table label is:

> QuantSpec (authors' source, reproduction pending)

It must not be labeled `our reproduction`, `exact BF16`, or `P/D progressive
transfer`.

## 4. Integration decision

Do not install PyTorch 2.4 into the existing vLLM environment and do not vendor
the unlicensed source. The next executable step, once a GPU is exclusively
available, is a separate `uv` Python 3.11 environment plus an external pinned
checkout. First run the authors' Llama-3.1-8B path unchanged; only after that
passes should a request-level wrapper be added.

This baseline does not replace the main SparseCache-PD mechanism. It supplies
the most important ablation for the statement that a genuinely cheap draft
path—not merely a smaller visible KV set—is necessary for transfer overlap to
produce a net latency gain.
