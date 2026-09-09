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
baseline. It does not by itself measure SparseCache, because upstream LMCache
does not expose a progressively consumable subset to the direct-KV drafter.
Promotion requires a connector extension that sends the five drafter-layer
anchor pages first, preserves their original token positions, sends every
remaining target byte exactly once, and exposes per-layer readiness to the
verifier.

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

The configured 24 GiB CUDA transfer buffer covers a length-stratified batch of
distinct prompts without exhausting the receiver's registered PD arena. A 2
GiB arena covers one 8K Qwen3-8B BF16 KV cache, but is not a valid multi-request
benchmark configuration because transferred chunks remain resident long enough
to make later allocations time out. NIXL must be installed in the adjacent
LMCache Python 3.12 environment. Logs and benchmark outputs belong under
`outputs/` and are excluded from git.
