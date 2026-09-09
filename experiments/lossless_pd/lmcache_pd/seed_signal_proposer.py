"""vLLM custom proposer that exposes the exact P token before KV transfer.

This is deliberately not a draft model.  Activating any custom proposer makes
vLLM sample the target token before ``wait_for_save``; this hook records that
token for SparseCache's AnchorReady control message and returns no proposals.
The actual sparse-KV drafter remains a D-side component.
"""

from __future__ import annotations

from experiments.lossless_pd.lmcache_pd.anchor_runtime import publish_seed_batch


class SeedSignalProposer:
    def __init__(self, vllm_config) -> None:
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
        )

    def propose(
        self,
        sampled_token_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        *,
        slot_mappings=None,
    ):
        del num_tokens_no_spec, token_ids_cpu, slot_mappings
        publish_seed_batch(sampled_token_ids)
        return [[] for _ in sampled_token_ids]
