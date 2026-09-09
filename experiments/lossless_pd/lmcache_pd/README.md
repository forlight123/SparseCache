# LMCache 1P1D validation stage

This directory pins the first real LMCache transport gate for SparseCache. It
uses the unmodified LMCache/vLLM sources in the adjacent workspace and keeps
all experiment-specific state here.

The initial topology is one host with Qwen3-8B:

```text
GPU 0: vLLM prefiller + LMCache NIXL sender
GPU 1: vLLM decoder   + LMCache NIXL receiver
CPU:   request proxy on 19100
```

This gate establishes functional P/D separation and measures the full-KV
baseline. The current hook also delivers a real AnchorReady notification to a
decoder mailbox, resolves the advertised keys to the receiver's CUDA-backed
``MemoryObj`` instances, and exposes ordered zero-copy views for the five
drafter layers. It does not yet invoke the trained drafter or verifier online.

The opt-in runtime hook provides two progressively stronger measurement modes:

* ``SPARSECACHE_ANCHOR_PATCH=1`` splits the NIXL write after upstream LMCache
  has gathered every chunk.  This isolates wire cost but cannot improve
  AnchorReady.
* additionally setting ``SPARSECACHE_GATHER_FIRST=1`` partitions
  ``LMCacheEngine.store`` itself.  It gathers and submits whole-token-chunk
  anchors first, then gathers and submits every residual chunk.  The final
  decoder notification is still withheld until both phases finish, so the
  target-visible behavior and exact output remain unchanged.  JSONL traces
  distinguish synchronous ``gather_submit`` events from asynchronous
  ``nixl_write`` completions using a common request id and monotonic clock.

This whole-chunk implementation is a systems timing gate, not the final sparse
draft layout.  It sends all model layers for about 10% of token chunks.  The
paper design subsequently packs only the drafter's required layer/token pages
and adds an AnchorReady signal for speculative execution.

The P-side exact seed is sampled after the model forward but upstream vLLM
normally calls LMCache ``wait_for_save`` before sampling.  Launching the
prefiller with vLLM's custom-class speculative lifecycle and
``seed_signal_proposer.SeedSignalProposer`` reverses those two control events:
the hook queues the exact target seed before store and returns zero draft
tokens.  It is not EAGLE and it does not predict anything.  The runtime binds
that seed to the Anchor NIXL completion; when
``SPARSECACHE_ANCHOR_NOTIFY=udp://HOST:PORT`` is set, the same trace row is also
emitted as a best-effort UDP AnchorReady control message.  Final outputs still
come only from the original target path.

On the decoder, set ``SPARSECACHE_RECEIVER_PATCH=1`` and
``SPARSECACHE_ANCHOR_LISTEN=udp://HOST:PORT``. Optional
``SPARSECACHE_DRAFT_LAYERS=1,9,17,25,33`` makes the mailbox pin the live Anchor
objects, construct basic-slice tensor views, validate that every view aliases
the registered CUDA storage, and release the owners. The public mailbox claim
API retains the pins for a future asynchronous draft task.

The launch contract is strict:

* set ``PYTHONHASHSEED=0`` in both P and D before process creation;
* launch P with vLLM ``kv_role=kv_producer``;
* launch D with vLLM ``kv_role=kv_consumer``;
* use the same model, LMCache chunk size, dtype, and hash policy on both nodes.

Using ``kv_both`` on D is invalid for this unidirectional backend: a decoder
boundary block may be stored with no receiver transfer specification. Omitting
the fixed hash can make D miss transferred cache keys and silently recompute,
which produces deceptively exact monolithic outputs but does not test KV reuse.

The configured 24 GiB CUDA transfer buffer covers a length-stratified batch of
distinct prompts without exhausting the receiver's registered PD arena. A 2
GiB arena covers one 8K Qwen3-8B BF16 KV cache, but is not a valid multi-request
benchmark configuration because transferred chunks remain resident long enough
to make later allocations time out. NIXL must be installed in the adjacent
LMCache Python 3.12 environment. Logs and benchmark outputs belong under
`outputs/` and are excluded from git.
