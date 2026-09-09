# Sparse-KV draft model training runbook

> 2026-09-09 decision: stop scaling the EAGLE adapter. The active, lossless
> direct-KV/block training protocol is
> [ICLR2027_LOSSLESS_EXPLORATION.md](docs/ICLR2027_LOSSLESS_EXPLORATION.md).
> The recommendations below are historical experiment records; Phase B/C
> EAGLE expansion is no longer the next action.

## Active direct-KV plan (supersedes the historical runbook below)

Do not launch any EAGLE job in this file. The executable method and evidence are
documented in [ICLR2027_ALGORITHM.md](docs/ICLR2027_ALGORITHM.md),
[ICLR2027_LOSSLESS_EXPLORATION.md](docs/ICLR2027_LOSSLESS_EXPLORATION.md), and
[ICLR2027_ONLINE_LMCACHE_RESULT_20260909.md](docs/ICLR2027_ONLINE_LMCACHE_RESULT_20260909.md).

The inherited five-layer direct-KV block accepts 1.531 target tokens at the 10%
priority view on 64 frozen long-prompt requests. A top-64 hidden-space causal
reranker trained on 128 disjoint QMSum mechanism-development requests raises this
to 2.188 at horizon 15 (paired delta +0.656, 95% CI [0.359,0.953]). At the
latency-oriented horizon 7 it advances 3.141 target-verified output tokens per
block including the correction/bonus. Zero KV accepts 0 and other-request KV
accepts 0, so it still uses the correct KV content. This is not EAGLE: no Target
hidden state enters the drafter.

Do not scale the first full-vocabulary residual head unchanged. Although it
improves acceptance, repeated vocabulary projections contend with transfer and
make the 100-Gbps pipeline slower. The active implementation screen is the
top-K hidden-space causal reranker in
`experiments/lossless_pd/causal_correction_train.py`. It reuses one parallel base
vocabulary projection and scores only gathered candidate LM-head rows. Promotion
requires both held-out acceptance and positive integrated accepted-tokens/ms.
That latency gate now passes in a clean, serial 100-Gbps one-host proxy: top-64,
10% priority KV and `g=7` save 62.348 ms to equal output progress (95% CI
[50.209,74.642]) with +0.032 ms mean first-commit delta. Do not run timing cells
on multiple GPUs simultaneously; shared host-memory/PCIe contention invalidated
the control as well as the treatment in the first attempt.

The QMSum-only reranker does not generalize to MultiNews/GovReport, so it has
been superseded by a 3000-step five-summary-set hidden-256 checkpoint. At `g=7`
that checkpoint improves held-out-document acceptance from 1.531 to 2.313 on
QMSum, 1.016 to 2.891 on MultiNews, and 1.000 to 4.547 on GovReport; paired deltas
have 95% CIs [0.453,1.125], [1.672,2.078], and [3.219,3.875]. Its clean
100-Gbps integrated QMSum run advances 3.312 verified output tokens and saves
68.787 ms at equal progress (95% CI [56.626,81.581]) with -0.028 ms mean
first-commit delta. These tasks occur in mechanism-training data, so the result
is held-out-document evidence, not a final paper generalization claim.

The real LMCache 1P1D integration now consumes the arrived Anchor online. The D
node resolves zero-copy Target-KV views, runs the direct block model concurrently
with Residual movement, repairs the suffix from the first authoritative Target
token, and submits it to vLLM's full-KV verifier. In a strict 64-request
token-ID observe--inject--observe sandwich, greedy output equals monolithic on
64/64 requests and injection saves 24.407 ms total latency, 95% CI
[19.513,29.671] ms, with no statistically separated TTFT change. Mean verified
progress is 2.438 tokens, including 1.438 injected suffix tokens; hot draft and
repair cost 24.860 and 3.035 ms. See
[ICLR2027_ONLINE_LMCACHE_RESULT_20260909.md](docs/ICLR2027_ONLINE_LMCACHE_RESULT_20260909.md).

The next GPU spend is model/data improvement, not another systems wiring pass.
Train on exact 256-token `protected_uniform` runtime views and optimize suffix
acceptance conditioned on authoritative `t1`. Use document-disjoint native 8K+
prefixes, exclude every pilot/final evaluation document, and screen at least
three seeds only after a cheap architecture run passes both held-out acceptance
and live accepted-tokens/ms. In parallel, profile fusion of Anchor packing and
the 3-ms repair; do not scale a checkpoint whose online latency gate is
negative.

Continue reporting `target_in_base_topk_rate` and
`first_base_error_target_in_topk_rate` from `packet_eval.py`: low first-error
coverage calls for changing the candidate generator/top-K, while high coverage
with low realized repair calls for more diverse block-wise training. The winning
checkpoint is
`outputs/progressive_kv/iclr2027_20260909/causal_rerank_summary5_s3000_h256_k64/checkpoint`
on this development machine; generated checkpoints remain outside git.

The next training corpus must have at least 8K native prefixes and must exclude
QMSum pilot IDs 0..63 and all final LongBench/LongBench-v2 test documents. Split
by source conversation/document before extracting windows. For each example,
one immutable teacher pass produces:

```text
input:  known P seed token
        exact Target K/V at layers 1,9,17,25,33 for nested 5/10/20% views
        original absolute positions, prompt length, stage ID
teacher: full-target distributions for 15 future positions
audit:  prompt/reference/checkpoint hashes, page order, zero/shuffled controls
output: full-vocabulary proposal logits; no target hidden state and no EAGLE state
```

Use the same frozen teacher packet and update sequence for every arm. Do not
independently replay BF16 greedy targets: a fresh deterministic replay changed
5/64 suffixes in the current audit. Teacher packets belong on a sharded/lazy
reader; never materialize every full 36-layer cache in RAM or on disk.

The first paper-relevant architecture screen should match supervised positions,
data and optimizer across three arms:

1. the present one-pass parallel block model with full-view training;
2. the same model with correctly rounded nested long-prefix views;
3. a direct-KV causal correction path (small GRU/2-layer AR head) conditioned on
   earlier proposed tokens, still using only sparse Target KV and the known seed.

The third arm addresses the present block model's exposure problem without
returning to EAGLE. Evaluate proposal lengths 3/7/15 and visibility 5/10/20%,
report accepted-prefix survival, zero-acceptance probability, accepted tokens per
draft millisecond, and the integrated first-commit/equal-progress endpoints.
Scale beyond the first 10K unique long prompts only if held-out 10% acceptance
improves with a positive request-bootstrap CI and the 100-Gbps latency gate stays
positive. Three training seeds are required for finalists, not for the initial
architecture rejection screen.

The current batched verifier is lossless in the standard speculative-decoding
real-arithmetic sense but not bitwise identical to sequential BF16 execution on
all requests. A minimal row-invariant oracle localized the required endpoint-shape
operators to attention reduction, MLP down projection and RMSNorm for this Qwen3
path. It is bitwise exact for logits and generated KV on 64/64 requests and takes
219.14 ms versus 475.16 ms for whole-layer rowwise execution, but remains too
slow. Training quality claims may use frozen references; a deployment-level
bitwise claim requires these row programs to be fused as described in the
algorithm document.

Mechanism-only reranker screen (LongBench-derived QMSum rows 64..191 train and
0..63 evaluation; do not use these rows for final paper quality claims):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
  experiments.lossless_pd.causal_correction_train \
  --train-packets \
outputs/progressive_kv/iclr2027_20260909/train_packets_qmsum_offset64_n64,outputs/progressive_kv/iclr2027_20260909/train_packets_qmsum_offset128_n64 \
  --eval-packets \
outputs/progressive_kv/iclr2027_20260909/reference_packets_n64 \
  --base-checkpoint \
outputs/progressive_kv/iclr2027_20260909/pilot_n64_s1000/nested_block_gpu1/checkpoint \
  --output outputs/progressive_kv/causal_rerank_k64 \
  --steps 1000 --eval-requests 64 --fractions .05,.1,.2 \
  --correction-mode rerank --correction-topk 64 \
  --correction-hidden 128 --correction-bottleneck 128 \
  --objective error_correct --preserve-weight 1 --lr 2e-4
```

---

## Historical EAGLE runbook (do not execute)

Status: cloud-training handoff, 2026-08-29.

This document answers three practical questions:

1. Why is the current trained drafter weak?
2. Should we only add data, or change the draft model?
3. What can be run now after cloning this repository on a GPU machine?

The short answer is: **do not spend a large cloud budget scaling the current
frozen-adapter configuration unchanged**. The result is limited by both data
scale and model/training structure. First run a controlled ceiling experiment
that trains the EAGLE input projection and draft layer together with the
sparse-KV path. Only scale beyond that after replacing the full-KV teacher
files with a compact or online data path.

## 1. Current result and diagnosis

The best local checkpoint used Llama-3.1-8B-Instruct as target, an EAGLE-3
checkpoint as initialization, 600 multi-task requests, 16,378 continuation
tokens, 9,600 single-request optimizer steps, and 5/10/20% priority-ordered KV
views. Of 449.9M total drafter parameters, only 75.5M parameters in the sparse
adapter and EAGLE input projection were trainable.

Held-out post-verification boundaries contain 707 windows and 4,525 tokens:

| memory supplied to drafter | top-1 | mean accepted prefix, block <= 8 |
|---|---:|---:|
| exact sparse target KV, 5% | 24.42% | 0.423 |
| zero KV | 20.33% | 0.380 |
| same-size KV from another request | 19.43% | 0.334 |

This establishes that the model reads real KV content: exact KV beats both
zero and request-shuffled KV. It is nevertheless unusable as a serving draft
model because the accepted prefix is below one token and cannot repay draft
plus verification cost.

There are four distinct causes.

### 1.1 The data/model ratio is far too small

Sixteen thousand supervised continuation positions are insufficient for
75.5M trainable parameters. The samples are revisited many times, so adding
optimizer steps mostly increases memorization. As a scale reference rather
than a directly comparable result, KVShot (arXiv:2604.26412) uses about 70k
requests for three-epoch architecture ablations and 280k requests with
target-regenerated responses for its end-to-end study.

### 1.2 The pretrained EAGLE computation is not adapted deeply enough

The current path freezes EAGLE's draft transformer and output head. Training
only the sparse cross-attention and input projection asks a small residual
branch to repair a representation whose self-attention was trained for a
different, full-context hidden-state regime. More data can improve this, but
scaling only the same adapter is unlikely to remove the architectural ceiling.

### 1.3 The sparse-KV branch starts almost closed

The original prototype initialized the scalar memory injection to
`sigmoid(-4)`, approximately 1.8%. This protects the pretrained EAGLE path but
also starves the new KV branch of gradient. The training CLI now exposes
`--adapter-scale-init` and `--adapter-scale-warmup-steps`. New ceiling runs
should start at logit 0 (50% injection) and hold that scale fixed briefly while
the KV query/output projections learn a useful direction.

### 1.4 Autoregressive training gives sparse gradients to the KV path

Each optimizer step contains only a short draft trajectory and only the
selected pages. The cross-attention projections therefore see much less
diverse supervision than the language path. Longer traces and boundary
sampling help, but the final model should use block-wise or multi-token
training so that many future queries supervise the same KV view in parallel.

## 2. Model decision

The immediate choice is not “more data or another model.” The recommended
sequence is:

| phase | model | purpose | decision gate |
|---|---|---|---|
| A | current sparse adapter + EAGLE `fc` | reproduce the existing result | exact KV must beat shuffled KV |
| B | EAGLE `fc` + full draft midlayer + sparse adapter | measure the ceiling of the hybrid design | accepted prefix >= 1.0 before large scaling |
| C | properly gated hybrid, trained on 70k+ requests | candidate deployment drafter | accepted prefix >= 2.0 and positive latency model |
| D | 2–4 layer sparse-KV-native/block-wise drafter | only if Phase C saturates | better accepted tokens/ms than Phase C |

Phase B is the next run. It keeps the strong hidden-state path for immediate
tokens while allowing sparse KV to correct longer-range predictions. This is
safer than switching immediately to a pure KV-only model: a shallow KV-only
model must infer good target queries from token embeddings and tends to have
weak first-token accuracy. Increasing it to 2–4 layers improves query quality
but also increases draft latency.

The final architecture should use a vector-valued gated delta rather than the
current scalar residual:

```text
self = EAGLE self-attention(hidden anchor, draft history)
cross = cross-attention(query, arrived sparse target KV)
delta = cross - self
output = self + sigmoid(gate(self, cross, delta, stage)) * project(delta)
```

The gate must be monitored during training. A gate collapsing to zero means
the model has reverted to hidden-only EAGLE; a permanently saturated gate
means the sparse branch is overriding the short-range language anchor.

## 3. Training contract

For the current exact `W=infinity` system, the final verifier always uses full
KV. A training item is therefore:

```text
input:
  committed target seed token
  target boundary hidden states from layers 2, 16, and 29
  exact target K/V from layers 8, 20, and 31 for arrived pages only
  original RoPE positions, prompt length, and visible fraction
  previous draft tokens within the current training window

teacher:
  full-KV target top-64 distribution and greedy token

negative controls:
  zero KV
  same-size KV from a different request/task

output:
  next-token logits in the EAGLE compressed vocabulary
```

The full-KV teacher is correct for the current one-shot final verifier. If the
method later changes to finite `W`, each sample must instead pair stage `S_i`
with the actual `S_(i+W)` verifier distribution and boundary hidden state.
Those stage-specific traces are not produced by the current extractor.

The current objective combines:

- prefix-weighted hard-label cross entropy;
- top-64 distribution distillation;
- a margin between exact and request-shuffled KV;
- a smaller margin between exact and memory-disabled predictions;
- sampling of initial and post-verification boundaries;
- nested 5/10/20% priority views.

All reported quality numbers must come from held-out requests. The default
split reserves indices satisfying `index % 5 == 0` and never trains on them.

## 4. Repository and external assets

The Git repository intentionally contains code, tests, configs, and
documentation only. It does not contain model weights, EAGLE checkpoints,
teacher KV, raw benchmark datasets, virtual environments, or generated
outputs.

Required external paths on the cloud machine:

- target model directory, currently Llama-3.1-8B-Instruct;
- matching EAGLE-3 checkpoint directory containing `config.json` and
  `pytorch_model.bin`;
- tokenized request packs or a prebuilt teacher manifest and safetensors;
- an output volume with enough space for teacher KV and checkpoints.

Teacher manifests currently record absolute model and base-manifest paths.
Regenerating teacher data on the cloud machine is the simplest option. If
teacher files are copied from another host, preserve the original mount paths
or update `target_model`, `requests_jsonl`, and `base_manifest` in the JSON
manifests before training. The model fingerprint must remain unchanged.

## 5. Environment setup

```bash
git clone git@github.com:forlight123/SparseCache.git
cd SparseCache

uv venv --python 3.12 .venv
uv sync --python .venv/bin/python

PYTHONPATH=src:. .venv/bin/pytest -q \
  tests/test_sparse_kv_eagle.py \
  tests/test_train_sparse_kv_drafter.py \
  tests/test_extract_sparse_kv_teacher.py \
  tests/test_eval_sparse_kv_boundaries.py
```

The reference code uses one CUDA device per process and BF16. Set explicit
paths rather than relying on machine-specific defaults:

```bash
export SC_TARGET_MODEL=/models/Llama-3.1-8B-Instruct
export SC_EAGLE_CHECKPOINT=/models/EAGLE3-LLaMA3.1-Instruct-8B/pytorch_model.bin
export SC_PACK_ROOT=/data/sparsecache/pd_packs/llama31_8b/quality
export SC_WORK_ROOT=/data/sparsecache/training
```

## 6. Build a deterministic multi-task corpus

The pack root is expected to contain `requests.jsonl`, `metadata.jsonl`, and
`manifest.json` under each selected dataset directory.

```bash
PYTHONPATH=src:. .venv/bin/python -m experiments.build_sparse_kv_draft_corpus \
  --pack-root "$SC_PACK_ROOT" \
  --datasets qmsum,musique,qasper,multi_news,hotpotqa,narrativeqa \
  --samples-per-dataset 100 \
  --output-dir "$SC_WORK_ROOT/corpus_n600"
```

The 600-request corpus is a code-path gate, not a sufficient training corpus.

## 7. Generate teacher data

The v3 extractor stores exact prompt KV, an eight-token initial trace, page
priorities, and target hidden states. The v5 extension regenerates a longer
continuation and adds post-verification boundary hidden states while reusing
the v3 static KV.

```bash
export SC_REQUESTS="$SC_WORK_ROOT/corpus_n600/requests.jsonl"
export SC_STATIC_TEACHER="$SC_WORK_ROOT/teacher_static_n600"
export SC_BOUNDARY_TEACHER="$SC_WORK_ROOT/teacher_boundary64_n600"

SC_REQUEST_INDICES=$(
  .venv/bin/python -c 'print(",".join(map(str, range(600))))'
)

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. \
.venv/bin/python -m experiments.extract_sparse_kv_teacher \
  --model "$SC_TARGET_MODEL" \
  --requests-jsonl "$SC_REQUESTS" \
  --request-indices "$SC_REQUEST_INDICES" \
  --output-dir "$SC_STATIC_TEACHER" \
  --max-context-tokens 16384 \
  --continuation-tokens 8 \
  --teacher-topk 64 \
  --kv-layers 8,20,31 \
  --seed-layers 2,16,29 \
  --priority-page-size 64 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. \
.venv/bin/python -m experiments.extend_sparse_kv_teacher \
  --base-manifest "$SC_STATIC_TEACHER/manifest.json" \
  --output-dir "$SC_BOUNDARY_TEACHER" \
  --continuation-tokens 64 \
  --teacher-topk 64 \
  --device cuda:0
```

At the observed average prompt length, the 600-request full static teacher is
about 86 GiB. The current loader materializes all selected teacher samples in
CPU RAM. Do not increase this format to tens of thousands of requests.

Before large-scale training, implement one of these data paths:

1. store only the maximum required priority prefix (for example, top 20% KV)
   plus original token indices, then derive 5/10/20% nested views from it;
2. memory-map and lazily load one request/shard at a time;
3. generate target responses and sparse KV online on separate teacher GPUs.

Option 1 is the recommended first implementation. It reduces disk and CPU
memory by roughly the retained fraction and preserves exact target KV for the
views actually consumed by the drafter.

## 8. Reproduce the old adapter result

This run exists only as a regression control. Passing `-4` preserves the old
1.8% initial KV injection.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
.venv/bin/python -m experiments.train_sparse_kv_eagle \
  --teacher-manifest "$SC_BOUNDARY_TEACHER/manifest.json" \
  --eagle-checkpoint "$SC_EAGLE_CHECKPOINT" \
  --output-dir "$SC_WORK_ROOT/eagle_adapter_regression" \
  --eval-modulus 5 --eval-remainder 0 \
  --visibility-fractions 0.05,0.10,0.20 \
  --selection-mode priority --page-size 64 \
  --train-fc \
  --adapter-scale-init -4 \
  --training-window-tokens 8 --initial-window-prob 0.25 \
  --steps 9600 --learning-rate 1e-4 \
  --max-eval-tokens 8 --log-every 100 \
  --device cuda:0
```

## 9. Recommended next ceiling run

This opens the EAGLE input projection and complete draft midlayer, initializes
the sparse injection at 50%, and freezes the scalar for the first 1,000 steps
so the cross-attention projections cannot be silenced immediately.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
.venv/bin/python -m experiments.train_sparse_kv_eagle \
  --teacher-manifest "$SC_BOUNDARY_TEACHER/manifest.json" \
  --eagle-checkpoint "$SC_EAGLE_CHECKPOINT" \
  --output-dir "$SC_WORK_ROOT/eagle_fullhybrid_ceiling" \
  --eval-modulus 5 --eval-remainder 0 \
  --visibility-fractions 0.05,0.10,0.20 \
  --selection-mode priority --page-size 64 \
  --train-fc --train-midlayer \
  --adapter-scale-init 0 \
  --adapter-scale-warmup-steps 1000 \
  --training-window-tokens 8 --initial-window-prob 0.25 \
  --steps 30000 --learning-rate 2e-5 --weight-decay 0.01 \
  --prefix-decay 0.9 \
  --hard-weight 1.0 --soft-weight 0.5 \
  --contrast-weight 0.5 --base-contrast-weight 0.25 \
  --max-eval-tokens 8 --log-every 100 --save-every 5000 \
  --device cuda:0
```

`--save-every` writes model-only snapshots. Optimizer state is deliberately
not embedded because a fully opened draft layer creates very large optimizer
checkpoints. `--resume-adapter` restores model weights and starts a fresh
AdamW schedule for the requested additional number of steps:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
.venv/bin/python -m experiments.train_sparse_kv_eagle \
  --teacher-manifest "$SC_BOUNDARY_TEACHER/manifest.json" \
  --eagle-checkpoint "$SC_EAGLE_CHECKPOINT" \
  --resume-adapter \
    "$SC_WORK_ROOT/eagle_fullhybrid_ceiling/adapter_step_00030000.pt" \
  --output-dir "$SC_WORK_ROOT/eagle_fullhybrid_resume_30k" \
  --eval-modulus 5 --eval-remainder 0 \
  --visibility-fractions 0.05,0.10,0.20 \
  --selection-mode priority --page-size 64 \
  --train-fc --train-midlayer \
  --adapter-scale-init 0 \
  --training-window-tokens 8 --initial-window-prob 0.25 \
  --steps 30000 --learning-rate 1e-5 \
  --max-eval-tokens 8 --log-every 100 --save-every 5000 \
  --device cuda:0
```

Use the same trainability flags when resuming. `--steps` means additional
steps in the new output directory, not the global accumulated step number.

## 10. Boundary evaluation and mandatory ablations

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
.venv/bin/python -m experiments.eval_sparse_kv_boundaries \
  --checkpoint "$SC_WORK_ROOT/eagle_fullhybrid_ceiling/adapter.pt" \
  --teacher-manifest "$SC_BOUNDARY_TEACHER/manifest.json" \
  --output "$SC_WORK_ROOT/eagle_fullhybrid_ceiling/boundary_ablation.json" \
  --eval-modulus 5 --eval-remainder 0 \
  --visibility-fractions 0.05,0.10,0.20 \
  --selection-mode priority --window-tokens 8 \
  --max-windows-per-sample 8 --device cuda:0
```

Always report:

- per-step acceptance `alpha_0 ... alpha_7` when available;
- mean accepted prefix and full-block acceptance;
- exact, zero, and request-shuffled KV under identical budgets;
- 5/10/20% KV views and each realized token fraction;
- initial boundary separately from post-verification boundaries;
- metrics by task, especially short-answer versus summarization tasks;
- draft time per token, verification time, and accepted tokens/ms;
- final answer EM/F1 and exact output equality for the immutable verifier.

## 11. Go/no-go rules

Do not advance a model merely because training loss falls.

### Advance from the 600-request ceiling run when all are true

- exact KV mean accepted prefix is at least 1.0 for an eight-token proposal;
- exact KV exceeds shuffled KV by at least 0.25 accepted token;
- improvement appears on held-out requests and more than one task;
- the KV injection/gate remains active rather than collapsing to zero;
- 10% or 20% KV improves over 5%, or the saturation is explained by measured
  page-level attention coverage.

### Advance to a full systems run when all are true

- mean accepted prefix is at least 2.0 at the chosen sparse budget;
- exact versus shuffled gain is at least 0.5 accepted token;
- draft cost is below 20–25% of one target decode step per proposed token, or
  the transfer-overlap latency equation remains positive with measured costs;
- the final full-KV verifier preserves the target output/quality contract;
- gain survives multi-request load and is not only a single-request TTFT win.

If Phase B remains below one accepted token, do not scale it to 70k unchanged.
Implement the gated hybrid and block-wise training first. If that model reaches
good acceptance but is too slow, reduce cross-attended target layers and draft
width before considering a pure KV-only drafter.

## 12. GPU2 controlled runs (2026-08-30)

The first 1,200-request expansion was built from six task packs (200 examples
per task) and split by `request_index % 5`: 960 train / 240 held out.  The
static v3 teacher files are about 170 GiB; the v5 eight-token trace extension
adds about 230 MiB.  The long 64-token trace is intentionally a separate
follow-up artifact because it is much more expensive to extract.

The main run used the EAGLE-3 Llama-3.1-8B checkpoint, priority page selection,
5/10/20% visible KV, an eight-token random boundary window, gated-delta fusion,
trainable `fc`, and full-target hidden feature distillation (`--feature-weight
0.25`).  It ran 9,600 steps on GPU2 (142.6M trainable parameters).  The
independent boundary evaluator measured:

| KV ablation | teacher-forced top-1 | mean accepted prefix | full-block accept |
|---|---:|---:|---:|
| exact, 5% | 26.77% | 0.424 | 14.74% |
| exact, 10% | 26.71% | 0.425 | 15.05% |
| exact, 20% | 26.69% | 0.426 | 14.82% |
| zero KV | 23.32% | 0.387 | 12.87% |
| shuffled KV | 17.92% | 0.320 | 8.19% |

For comparison, the same checkpoint evaluated on the earlier n600 corpus with
64-token traces reaches 0.777 exact versus 0.736 zero and 0.588 shuffled
accepted prefix.  This distribution gap is why all future claims must include
both the expanded held-out split and the original 64-token boundary split.

The full-hybrid control (also 9,600 steps, 385.9M trainable parameters) reached
0.466 exact versus 0.463 zero on the expanded boundary split.  It improves
absolute prefix over gated-only, but its small exact-zero gap shows that simply
unfreezing the EAGLE recurrent layer causes the drafter to rely more on its
language prior than on sparse KV.  The gated-only run is therefore the current
mechanistic baseline.

The training loss now tolerates a window whose teacher labels are outside the
compressed EAGLE vocabulary: hard-label and contrast terms become zero while
valid top-k soft distillation remains active.  Such windows are reported by
`label_coverage` rather than aborting a long run.

Reproduce the expanded run after materializing the private teacher manifest:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:. \
.venv/bin/python -m experiments.train_sparse_kv_eagle \
  --teacher-manifest outputs/progressive_kv/sparse_kv_draft_teacher/boundary64_multitask_n1200_trace8_20260830/manifest.json \
  --eagle-checkpoint /data/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B/pytorch_model.bin \
  --output-dir outputs/progressive_kv/sparse_kv_eagle/gateddelta_feature_fcadapt_boundary8_n1200_s9600_trace8_20260830_run2 \
  --eval-modulus 5 --eval-remainder 0 \
  --visibility-fractions 0.05,0.10,0.20 --selection-mode priority --page-size 64 \
  --fusion-mode gated_delta --train-fc \
  --adapter-scale-init 0 --adapter-scale-warmup-steps 1000 \
  --training-window-tokens 8 --initial-window-prob 0.25 \
  --steps 9600 --learning-rate 1e-4 --weight-decay 0.01 \
  --prefix-decay 0.9 --hard-weight 1.0 --soft-weight 0.5 \
  --feature-weight 0.25 --contrast-weight 0.75 --base-contrast-weight 0.5 \
  --max-eval-tokens 8 --log-every 400 --save-every 2400 --device cuda:0
```

## 13. What is not in Git

Never commit these artifacts to this repository:

- `*.safetensors`, `*.pt`, `*.bin`, model directories, or optimizer states;
- teacher KV and request-specific caches;
- raw datasets whose license or size prevents redistribution;
- `outputs/`, profiler traces, virtual environments, or local third-party
  source copies.

Move checkpoints through an object store, private model registry, or an
explicit GitHub Release/LFS workflow. Normal GitHub blobs are not suitable for
the current 300MB adapter checkpoint, and a full-midlayer checkpoint will be
larger.
