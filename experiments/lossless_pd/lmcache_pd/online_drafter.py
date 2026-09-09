"""Online direct-KV drafting from decoder-resident LMCache Anchor objects.

The receiver publishes an Anchor only after NIXL remote completion.  This
module packs the selected Target-KV layer views on a private CUDA stream, runs
the trained direct-KV block drafter while the residual is still moving, and
hands the completed block to vLLM's custom proposer lifecycle.

The current injection path is intentionally greedy-only.  It first reconciles
the drafter's first token with the authoritative Target token already sampled
by vLLM, then returns only the remaining suffix.  vLLM's normal full-KV target
pass verifies that suffix.  Sampling requires proposal probabilities and is
therefore outside this prototype's losslessness contract.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from experiments.lossless_pd.lmcache_pd.receiver_runtime import (
        AnchorMailbox,
        ClaimedLayerViews,
    )


_TRACE_LOCK = threading.Lock()


def _append_trace(path: str, row: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK, destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


@dataclass
class PackedTargetKV:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    prompt_tokens: int


def dynamic_cache_from_packed(config: Any, packed: PackedTargetKV):
    """Install compact, already-RoPE'd Target KV without copying it again."""

    from transformers import DynamicCache

    expected_layers = int(config.num_hidden_layers)
    if packed.keys.shape != packed.values.shape or packed.keys.ndim != 5:
        raise ValueError("packed Target KV must have matching [B,L,H,T,D] shapes")
    if packed.keys.shape[1] != expected_layers:
        raise ValueError("target self-draft requires every Target KV layer")
    cache = DynamicCache(config=config)
    if len(cache.layers) != expected_layers:
        raise ValueError("DynamicCache layer count does not match the Target")
    for index, layer in enumerate(cache.layers):
        key = packed.keys[:, index]
        value = packed.values[:, index]
        layer.keys = key
        layer.values = value
        layer.dtype = key.dtype
        layer.device = key.device
        layer.is_initialized = True
    return cache


def pack_claimed_target_kv(
    claim: ClaimedLayerViews,
    *,
    num_key_value_heads: int,
    head_dim: int,
) -> PackedTargetKV:
    """Pack KV_2LTD basic views into the drafter's compact BLHTD layout."""

    import torch

    chunks = len(claim.objects)
    if chunks == 0 or not claim.layers:
        raise ValueError("an online draft requires nonempty chunks and layers")
    if len(claim.token_ranges) != chunks:
        raise ValueError("one exact token range is required per LMCache chunk")
    if claim.prompt_tokens <= 0:
        raise ValueError("the original prompt length is required")
    if min(num_key_value_heads, head_dim) <= 0:
        raise ValueError("KV-head geometry must be positive")

    ordering = sorted(range(chunks), key=lambda index: claim.token_ranges[index])
    ordered_ranges = [claim.token_ranges[index] for index in ordering]
    previous_end = -1
    for start, end in ordered_ranges:
        if start < 0 or end <= start or end > claim.prompt_tokens:
            raise ValueError("LMCache token range lies outside the prompt")
        if start < previous_end:
            raise ValueError("LMCache Anchor token ranges overlap")
        previous_end = end

    keys_by_layer = []
    values_by_layer = []
    for layer in claim.layers:
        views = claim.views.get(layer, ())
        if len(views) != chunks:
            raise ValueError("every drafter layer must expose every Anchor chunk")
        key_chunks = []
        value_chunks = []
        for index in ordering:
            view = views[index]
            start, end = claim.token_ranges[index]
            logical_tokens = end - start
            if view.ndim != 4 or view.shape[:2] != (2, 1):
                raise ValueError("LMCache KV_2LTD view must have shape [2,1,T,D]")
            if view.shape[-2] < logical_tokens:
                raise ValueError(
                    "LMCache object is shorter than its logical token range"
                )
            if view.shape[-1] != num_key_value_heads * head_dim:
                raise ValueError("flattened Target-KV width does not match the drafter")
            logical = view[:, 0, :logical_tokens]
            key_chunks.append(
                logical[0]
                .reshape(logical_tokens, num_key_value_heads, head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
            )
            value_chunks.append(
                logical[1]
                .reshape(logical_tokens, num_key_value_heads, head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
            )
        keys_by_layer.append(torch.cat(key_chunks, dim=-2))
        values_by_layer.append(torch.cat(value_chunks, dim=-2))

    positions = torch.cat(
        [
            torch.arange(start, end, device=keys_by_layer[0].device)
            for start, end in ordered_ranges
        ]
    )
    return PackedTargetKV(
        keys=torch.stack(keys_by_layer, dim=1),
        values=torch.stack(values_by_layer, dim=1),
        positions=positions,
        prompt_tokens=claim.prompt_tokens,
    )


@dataclass(frozen=True)
class ReadyDraft:
    request_id: str
    pd_request_id: str
    prompt_tokens: int
    seed_token_id: int
    proposals: tuple[int, ...]
    anchor_received_ns: int
    draft_started_ns: int
    draft_finished_ns: int
    pack_gpu_ms: float
    model_gpu_ms: float
    total_gpu_ms: float
    wall_ms: float
    visible_tokens: int
    actual_fraction: float
    continuation: Any | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PendingVerify:
    draft: ReadyDraft
    prompt_signature: tuple[int, ...]


def _prompt_signature(token_ids_cpu, row: int, prompt_tokens: int) -> tuple[int, ...]:
    if prompt_tokens <= 0:
        return ()
    indexes = tuple(dict.fromkeys((0, prompt_tokens // 2, prompt_tokens - 1)))
    return tuple(int(token_ids_cpu[row, index]) for index in indexes)


class DraftRegistry:
    """Small FIFO registry joining early receiver work to a later D request."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._ready: deque[ReadyDraft] = deque()

    def publish(self, draft: ReadyDraft) -> None:
        with self._lock:
            self._ready.append(draft)
            while len(self._ready) > self.max_entries:
                self._ready.popleft()

    def claim(self, prompt_tokens: int, seed_token_id: int) -> ReadyDraft | None:
        with self._lock:
            for index, draft in enumerate(self._ready):
                if (
                    draft.prompt_tokens == prompt_tokens
                    and draft.seed_token_id == seed_token_id
                ):
                    del self._ready[index]
                    return draft
        return None

    def clear(self) -> None:
        with self._lock:
            self._ready.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._ready)


_REGISTRY = DraftRegistry()


class _VllmLMHeadAdapter:
    """Expose ParallelLMHead weights through the HF-style linear contract."""

    def __init__(self, target_head: Any, vocab_size: int) -> None:
        self.target_head = target_head
        self.vocab_size = vocab_size

    @property
    def weight(self):
        return self.target_head.weight[: self.vocab_size]

    def __call__(self, hidden):
        import torch.nn.functional as F

        return F.linear(hidden, self.weight)


class _VllmEmbeddingAdapter:
    """Bypass vLLM's compiled serving wrapper for a tiny private-stream batch."""

    def __init__(self, target_embedding: Any, vocab_size: int) -> None:
        self.target_embedding = target_embedding
        self.vocab_size = vocab_size

    @property
    def weight(self):
        return self.target_embedding.weight[: self.vocab_size]

    def __call__(self, token_ids):
        import torch.nn.functional as F

        return F.embedding(token_ids, self.weight)


class LiveSparseDrafter:
    """One private-stream worker that consumes pinned LMCache Anchor views."""

    def __init__(
        self,
        checkpoint: str,
        *,
        layers: tuple[int, ...],
        draft_tokens: int,
        trace_path: str,
    ) -> None:
        import torch

        from experiments.blockdraft.model import BlockKVDraft
        from experiments.lossless_pd.core import ProgressiveBlock

        checkpoint_path = Path(checkpoint).resolve()
        base = BlockKVDraft.load_checkpoint(checkpoint_path)
        model = ProgressiveBlock(base)
        stage = checkpoint_path / "stage.pt"
        if stage.exists():
            model.stage.load_state_dict(
                torch.load(stage, map_location="cpu", weights_only=True)
            )
        if len(layers) != base.config.num_target_kv_layers:
            raise ValueError("runtime layer list does not match the drafter checkpoint")
        if not 0 < draft_tokens <= base.config.block_size - 1:
            raise ValueError("draft horizon exceeds the checkpoint block size")
        self.checkpoint = checkpoint_path
        self.layers = layers
        self.draft_tokens = draft_tokens
        self.trace_path = trace_path
        self.model = model.eval()
        self.embedding = None
        self.lm_head = None
        self.device = None
        self.stream = None
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._closed = False

    def load_model(self, target_model: Any) -> None:
        """Attach vLLM's already loaded embedding/head and materialize the draft."""

        import torch

        if self._thread is not None:
            return
        embedding = target_model.model.embed_tokens
        target_head = target_model.lm_head
        if (
            getattr(embedding, "tp_size", 1) != 1
            or getattr(target_head, "tp_size", 1) != 1
        ):
            raise ValueError(
                "the first online prototype supports tensor parallel size 1"
            )
        device = target_head.weight.device
        vocab_size = self.model.base.config.vocab_size
        if target_head.weight.shape[0] < vocab_size:
            raise ValueError("vLLM LM head is smaller than the drafter vocabulary")
        self.model = self.model.to(device=device, dtype=torch.bfloat16).eval()
        self.embedding = _VllmEmbeddingAdapter(embedding, vocab_size)
        self.lm_head = _VllmLMHeadAdapter(target_head, vocab_size)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._warmup()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="sparsecache-live-drafter",
        )
        self._thread.start()

    def _warmup(self) -> None:
        """Pay lazy CUDA/kernel initialization before the first Anchor arrives."""

        import torch

        tokens = int(os.environ.get("SPARSECACHE_DRAFTER_WARMUP_TOKENS", "1024"))
        if tokens <= 0:
            return
        assert self.device is not None and self.stream is not None
        assert self.embedding is not None and self.lm_head is not None
        config = self.model.base.config
        prompt_tokens = max(tokens + 1, 8192)
        started_ns = time.perf_counter_ns()
        with torch.cuda.stream(self.stream), torch.inference_mode():
            begin = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            begin.record(self.stream)
            keys = torch.zeros(
                1,
                config.num_target_kv_layers,
                config.num_key_value_heads,
                tokens,
                config.head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
            values = torch.zeros_like(keys)
            positions = torch.arange(tokens, device=self.device)
            seed = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.model.propose(
                    seed,
                    self.embedding,
                    self.lm_head,
                    keys,
                    values,
                    positions,
                    prompt_tokens,
                    length=self.draft_tokens,
                )
            finished.record(self.stream)
        finished.synchronize()
        finished_ns = time.perf_counter_ns()
        if self.trace_path:
            _append_trace(
                self.trace_path,
                {
                    "event": "live_sparse_draft_warmup",
                    "visible_tokens": tokens,
                    "gpu_ms": begin.elapsed_time(finished),
                    "wall_ms": (finished_ns - started_ns) / 1e6,
                },
            )

    def submit(self, mailbox: AnchorMailbox, row: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._thread is None:
            if self.trace_path:
                _append_trace(
                    self.trace_path,
                    {
                        "event": "live_sparse_draft_error",
                        "request_id": row.get("request_id", ""),
                        "error": "Target model was not attached before AnchorReady",
                    },
                )
            return
        self._jobs.put((mailbox, row))

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            mailbox, row = job
            self._run_one(mailbox, row)

    def _run_one(self, mailbox: AnchorMailbox, row: dict[str, Any]) -> None:
        import torch

        claim = None
        started_ns = time.perf_counter_ns()
        try:
            claim = mailbox.claim_layer_views(str(row["request_id"]), self.layers)
            if claim is None:
                raise RuntimeError("Anchor objects disappeared before the draft claim")
            seed_record = row.get("seed_record") or {}
            seed_token_id = int(seed_record["seed_token_id"])
            assert self.device is not None and self.stream is not None
            assert self.embedding is not None and self.lm_head is not None
            with torch.cuda.stream(self.stream), torch.inference_mode():
                begin = torch.cuda.Event(enable_timing=True)
                packed_event = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                begin.record(self.stream)
                packed = pack_claimed_target_kv(
                    claim,
                    num_key_value_heads=self.model.base.config.num_key_value_heads,
                    head_dim=self.model.base.config.head_dim,
                )
                packed_event.record(self.stream)
                seed = torch.tensor(
                    [[seed_token_id]], device=self.device, dtype=torch.long
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    continuation = self.model.prepare_rerank_block(
                        seed,
                        self.embedding,
                        self.lm_head,
                        packed.keys,
                        packed.values,
                        packed.positions,
                        packed.prompt_tokens,
                        length=self.draft_tokens,
                    )
                    proposals = self.model.propose_prepared_rerank(
                        continuation, self.embedding, self.lm_head
                    )
                finished.record(self.stream)
            finished.synchronize()
            finished_ns = time.perf_counter_ns()
            proposal_ids = tuple(int(token) for token in proposals[0].tolist())
            draft = ReadyDraft(
                request_id=str(row.get("request_id", "")),
                pd_request_id=str(row.get("pd_request_id", "")),
                prompt_tokens=packed.prompt_tokens,
                seed_token_id=seed_token_id,
                proposals=proposal_ids,
                anchor_received_ns=int(row["receiver_received_ns"]),
                draft_started_ns=started_ns,
                draft_finished_ns=finished_ns,
                pack_gpu_ms=begin.elapsed_time(packed_event),
                model_gpu_ms=packed_event.elapsed_time(finished),
                total_gpu_ms=begin.elapsed_time(finished),
                wall_ms=(finished_ns - started_ns) / 1e6,
                visible_tokens=int(packed.positions.numel()),
                actual_fraction=packed.positions.numel() / packed.prompt_tokens,
                continuation=continuation,
            )
            _REGISTRY.publish(draft)
            if self.trace_path:
                _append_trace(
                    self.trace_path,
                    {
                        "event": "live_sparse_draft",
                        **{
                            item.name: getattr(draft, item.name)
                            for item in fields(ReadyDraft)
                            if item.name != "continuation"
                        },
                        "proposals": list(draft.proposals),
                        "checkpoint": str(self.checkpoint),
                        "layers": list(self.layers),
                    },
                )
        # A failed request must release pinned LMCache objects and leave the
        # long-lived worker available for later requests; the trace is the
        # fail-closed signal consumed by the benchmark gate.
        except Exception as error:  # noqa: BLE001
            if self.trace_path:
                _append_trace(
                    self.trace_path,
                    {
                        "event": "live_sparse_draft_error",
                        "request_id": str(row.get("request_id", "")),
                        "pd_request_id": str(row.get("pd_request_id", "")),
                        "draft_started_ns": started_ns,
                        "draft_finished_ns": time.perf_counter_ns(),
                        "error": repr(error),
                    },
                )
        finally:
            if claim is not None:
                claim.release()

    def repair_suffix(
        self, draft: ReadyDraft, first_target_token: int
    ) -> tuple[list[int], float, float]:
        """Cheaply condition cached sparse-KV state on authoritative Target t1."""

        import torch

        if draft.continuation is None:
            raise ValueError("draft has no prepared rerank continuation")
        assert self.device is not None and self.embedding is not None
        stream = torch.cuda.current_stream(self.device)
        started_ns = time.perf_counter_ns()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            begin = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            begin.record(stream)
            token = torch.tensor(
                [[first_target_token]], device=self.device, dtype=torch.long
            )
            suffix = self.model.propose_prepared_rerank(
                draft.continuation,
                self.embedding,
                self.lm_head,
                first_target_token=token,
            )
            finished.record(stream)
        finished.synchronize()
        finished_ns = time.perf_counter_ns()
        return (
            [int(token) for token in suffix[0].tolist()],
            begin.elapsed_time(finished),
            (finished_ns - started_ns) / 1e6,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self._jobs.put(None)
            self._thread.join(timeout=10)


class LiveSparseTargetDrafter:
    """Run exact-layer sparse self-drafting with a colocated HF Target copy."""

    def __init__(
        self,
        model_path: str,
        *,
        layers: tuple[int, ...],
        draft_tokens: int,
        trace_path: str,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.layers = layers
        self.draft_tokens = draft_tokens
        self.trace_path = trace_path
        self.model = None
        self.device = None
        self.stream = None
        self.eos_ids: set[int] = set()
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._closed = False

    def load_model(self, target_model: Any) -> None:
        """Load one HF Target copy after vLLM has established the D device."""

        import torch
        from transformers import AutoModelForCausalLM

        if self._thread is not None:
            return
        device = target_model.lm_head.weight.device
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        expected_layers = int(self.model.config.num_hidden_layers)
        if self.layers != tuple(range(expected_layers)):
            raise ValueError("target self-draft needs SPARSECACHE_DRAFT_LAYERS=0..L-1")
        generation = self.model.generation_config
        eos = generation.eos_token_id
        if eos is None:
            eos_values = ()
        elif isinstance(eos, (list, tuple, set)):
            eos_values = eos
        else:
            eos_values = (eos,)
        self.eos_ids = {int(value) for value in eos_values if value is not None}
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._warmup()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="sparsecache-live-target-drafter",
        )
        self._thread.start()

    def _generate(self, packed: PackedTargetKV, seed_token_id: int):
        import torch

        assert self.model is not None and self.device is not None
        visible = int(packed.keys.shape[-2])
        cache = dynamic_cache_from_packed(self.model.config, packed)
        current = torch.tensor([[seed_token_id]], device=self.device, dtype=torch.long)
        attention_mask = torch.ones(
            (1, visible + 1), device=self.device, dtype=torch.long
        )
        proposals = []
        for step in range(self.draft_tokens):
            output = self.model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=torch.tensor(
                    [[packed.prompt_tokens + step]],
                    device=self.device,
                    dtype=torch.long,
                ),
                cache_position=torch.tensor(
                    [visible + step], device=self.device, dtype=torch.long
                ),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            token = int(output.logits[:, -1].argmax(dim=-1).item())
            proposals.append(token)
            cache = output.past_key_values
            if token in self.eos_ids:
                break
            current = torch.tensor([[token]], device=self.device, dtype=torch.long)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=torch.long),
                ),
                dim=1,
            )
        return tuple(proposals)

    def _warmup(self) -> None:
        import torch

        tokens = int(os.environ.get("SPARSECACHE_TARGET_DRAFTER_WARMUP_TOKENS", "1024"))
        if tokens <= 0:
            return
        assert self.model is not None and self.device is not None
        assert self.stream is not None
        config = self.model.config
        head_dim = int(
            getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            )
        )
        started_ns = time.perf_counter_ns()
        with torch.cuda.stream(self.stream), torch.inference_mode():
            begin = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            begin.record(self.stream)
            keys = torch.zeros(
                1,
                config.num_hidden_layers,
                config.num_key_value_heads,
                tokens,
                head_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            packed = PackedTargetKV(
                keys=keys,
                values=torch.zeros_like(keys),
                positions=torch.arange(tokens, device=self.device),
                prompt_tokens=max(tokens + 1, 8192),
            )
            self._generate(packed, 0)
            finished.record(self.stream)
        finished.synchronize()
        if self.trace_path:
            _append_trace(
                self.trace_path,
                {
                    "event": "live_sparse_target_draft_warmup",
                    "visible_tokens": tokens,
                    "gpu_ms": begin.elapsed_time(finished),
                    "wall_ms": (time.perf_counter_ns() - started_ns) / 1e6,
                },
            )

    def submit(self, mailbox: AnchorMailbox, row: dict[str, Any]) -> None:
        if not self._closed:
            self._jobs.put((mailbox, row))

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._run_one(*job)

    def _run_one(self, mailbox: AnchorMailbox, row: dict[str, Any]) -> None:
        import torch

        claim = None
        started_ns = time.perf_counter_ns()
        try:
            claim = mailbox.claim_layer_views(str(row["request_id"]), self.layers)
            if claim is None:
                raise RuntimeError("Anchor objects disappeared before target draft")
            seed_token_id = int((row.get("seed_record") or {})["seed_token_id"])
            assert self.model is not None and self.stream is not None
            config = self.model.config
            head_dim = int(
                getattr(
                    config,
                    "head_dim",
                    config.hidden_size // config.num_attention_heads,
                )
            )
            with torch.cuda.stream(self.stream), torch.inference_mode():
                begin = torch.cuda.Event(enable_timing=True)
                packed_event = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                begin.record(self.stream)
                packed = pack_claimed_target_kv(
                    claim,
                    num_key_value_heads=int(config.num_key_value_heads),
                    head_dim=head_dim,
                )
                packed_event.record(self.stream)
                proposals = self._generate(packed, seed_token_id)
                finished.record(self.stream)
            finished.synchronize()
            finished_ns = time.perf_counter_ns()
            draft = ReadyDraft(
                request_id=str(row.get("request_id", "")),
                pd_request_id=str(row.get("pd_request_id", "")),
                prompt_tokens=packed.prompt_tokens,
                seed_token_id=seed_token_id,
                proposals=proposals,
                anchor_received_ns=int(row["receiver_received_ns"]),
                draft_started_ns=started_ns,
                draft_finished_ns=finished_ns,
                pack_gpu_ms=begin.elapsed_time(packed_event),
                model_gpu_ms=packed_event.elapsed_time(finished),
                total_gpu_ms=begin.elapsed_time(finished),
                wall_ms=(finished_ns - started_ns) / 1e6,
                visible_tokens=int(packed.positions.numel()),
                actual_fraction=packed.positions.numel() / packed.prompt_tokens,
            )
            _REGISTRY.publish(draft)
            if self.trace_path:
                _append_trace(
                    self.trace_path,
                    {
                        "event": "live_sparse_draft",
                        **{
                            item.name: getattr(draft, item.name)
                            for item in fields(ReadyDraft)
                            if item.name != "continuation"
                        },
                        "proposals": list(draft.proposals),
                        "drafter_kind": "target",
                        "model": str(self.model_path),
                        "layers": list(self.layers),
                    },
                )
        except Exception as error:  # noqa: BLE001
            if self.trace_path:
                _append_trace(
                    self.trace_path,
                    {
                        "event": "live_sparse_draft_error",
                        "request_id": str(row.get("request_id", "")),
                        "pd_request_id": str(row.get("pd_request_id", "")),
                        "draft_started_ns": started_ns,
                        "draft_finished_ns": time.perf_counter_ns(),
                        "drafter_kind": "target",
                        "error": repr(error),
                    },
                )
        finally:
            if claim is not None:
                claim.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self._jobs.put(None)
            self._thread.join(timeout=10)


_SERVICE_LOCK = threading.Lock()
_SERVICE: LiveSparseDrafter | LiveSparseTargetDrafter | None = None


def get_online_drafter() -> LiveSparseDrafter | LiveSparseTargetDrafter | None:
    """Return the process-local service, creating it from explicit env only."""

    global _SERVICE
    kind = os.environ.get("SPARSECACHE_DRAFTER_KIND", "block")
    if kind not in {"block", "target"}:
        raise ValueError("SPARSECACHE_DRAFTER_KIND must be block or target")
    checkpoint = os.environ.get("SPARSECACHE_DRAFTER_CHECKPOINT", "")
    target_path = os.environ.get("SPARSECACHE_TARGET_DRAFTER_MODEL", "")
    if kind == "block" and not checkpoint:
        return None
    if kind == "target" and not target_path:
        return None
    with _SERVICE_LOCK:
        if _SERVICE is None:
            layers = tuple(
                int(value)
                for value in os.environ.get(
                    "SPARSECACHE_DRAFT_LAYERS", "1,9,17,25,33"
                ).split(",")
                if value.strip()
            )
            common = {
                "layers": layers,
                "draft_tokens": int(os.environ.get("SPARSECACHE_DRAFT_TOKENS", "7")),
                "trace_path": os.environ.get("SPARSECACHE_DRAFT_TRACE", ""),
            }
            if kind == "block":
                _SERVICE = LiveSparseDrafter(checkpoint, **common)
            else:
                _SERVICE = LiveSparseTargetDrafter(target_path, **common)
        return _SERVICE


class OnlineSparseKVProposer:
    """vLLM bridge for observing or injecting a precomputed sparse-KV block."""

    def __init__(self, vllm_config: Any) -> None:
        speculative = vllm_config.speculative_config
        self.num_speculative_tokens = speculative.num_speculative_tokens
        self.mode = os.environ.get("SPARSECACHE_ONLINE_DRAFT_MODE", "observe")
        if self.mode not in {"observe", "inject"}:
            raise ValueError("online draft mode must be observe or inject")
        self.trace_path = os.environ.get("SPARSECACHE_DRAFT_TRACE", "")
        self.service = get_online_drafter()
        if self.service is None:
            raise ValueError(
                "configure SPARSECACHE_DRAFTER_CHECKPOINT for block mode or "
                "SPARSECACHE_TARGET_DRAFTER_MODEL for target mode"
            )
        self._pending_feedback: list[PendingVerify] = []

    def load_model(self, target_model: Any) -> None:
        self.service.load_model(target_model)

    def propose(
        self,
        sampled_token_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        *,
        slot_mappings=None,
    ):
        del slot_mappings
        outputs = []
        for row, sampled in enumerate(sampled_token_ids):
            num_tokens = int(num_tokens_no_spec[row])
            if num_tokens <= 0 or not sampled:
                outputs.append([])
                continue
            for pending in tuple(self._pending_feedback):
                draft = pending.draft
                seed_position = draft.prompt_tokens
                if (
                    num_tokens > seed_position
                    and int(token_ids_cpu[row, seed_position]) == draft.seed_token_id
                    and _prompt_signature(token_ids_cpu, row, draft.prompt_tokens)
                    == pending.prompt_signature
                ):
                    accepted = min(
                        len(draft.proposals),
                        max(1, num_tokens - (draft.prompt_tokens + 2)),
                    )
                    accepted_injected_suffix = max(0, accepted - 1)
                    if self.trace_path:
                        _append_trace(
                            self.trace_path,
                            {
                                "event": "online_verify_feedback",
                                "request_id": draft.request_id,
                                "pd_request_id": draft.pd_request_id,
                                # Kept for compatibility with the first online
                                # report.  It includes the authoritative t1
                                # alignment position and is not the standard
                                # serving acceptance metric.
                                "accepted_prefix": accepted,
                                "accepted_injected_suffix": (
                                    accepted_injected_suffix
                                ),
                                "proposal_tokens": len(draft.proposals),
                                "num_tokens_no_spec": num_tokens,
                            },
                        )
                    self._pending_feedback.remove(pending)
                    break
            # vLLM 0.23 calls CPU custom proposers after bookkeeping, so the
            # newly sampled Target token occupies the final slot and the P
            # seed is one slot earlier.  Keep the pre-bookkeeping form as a
            # compatibility fallback for adjacent vLLM revisions.
            candidates = []
            if num_tokens >= 2:
                candidates.append(
                    (num_tokens - 2, int(token_ids_cpu[row, num_tokens - 2]))
                )
            candidates.append((num_tokens - 1, int(token_ids_cpu[row, num_tokens - 1])))
            draft = None
            for prompt_tokens, seed_token_id in candidates:
                draft = _REGISTRY.claim(prompt_tokens, seed_token_id)
                if draft is not None:
                    break
            if draft is None:
                outputs.append([])
                if self.trace_path:
                    _append_trace(
                        self.trace_path,
                        {
                            "event": "online_draft_miss",
                            "mode": self.mode,
                            "num_tokens_no_spec": num_tokens,
                            "candidate_prompt_seed": candidates,
                            "pending_drafts": len(_REGISTRY),
                        },
                    )
                continue
            first_target_token = int(sampled[0])
            first_matches = bool(
                draft.proposals and draft.proposals[0] == first_target_token
            )
            repair_gpu_ms = None
            repair_wall_ms = None
            if draft.continuation is not None:
                conditioned, repair_gpu_ms, repair_wall_ms = self.service.repair_suffix(
                    draft, first_target_token
                )
            else:
                conditioned = list(draft.proposals[1:]) if first_matches else []
            suffix = (
                conditioned[: self.num_speculative_tokens]
                if self.mode == "inject"
                else []
            )
            outputs.append(suffix)
            if suffix:
                self._pending_feedback.append(
                    PendingVerify(
                        draft,
                        _prompt_signature(token_ids_cpu, row, draft.prompt_tokens),
                    )
                )
            if self.trace_path:
                _append_trace(
                    self.trace_path,
                    {
                        "event": "online_draft_handoff",
                        "request_id": draft.request_id,
                        "pd_request_id": draft.pd_request_id,
                        "mode": self.mode,
                        "prompt_tokens": draft.prompt_tokens,
                        "seed_token_id": draft.seed_token_id,
                        "first_target_token": first_target_token,
                        "first_proposal_matches": first_matches,
                        "target_conditioned_repair": draft.continuation is not None,
                        "conditioned_proposals": conditioned,
                        "repair_gpu_ms": repair_gpu_ms,
                        "repair_wall_ms": repair_wall_ms,
                        "available_proposals": len(draft.proposals),
                        "returned_proposals": suffix,
                        "draft_ready_lead_ms": (
                            time.perf_counter_ns() - draft.draft_finished_ns
                        )
                        / 1e6,
                    },
                )
        return outputs
