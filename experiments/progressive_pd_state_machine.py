# SPDX-License-Identifier: Apache-2.0
"""Runtime invariants for progressive sparse drafting during P/D KV loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ProgressivePDPhase(Enum):
    """Mutually exclusive phases of one decode-side request."""

    WAIT_ANCHOR = auto()
    DRAFT_WHILE_LOADING = auto()
    READY_TO_VERIFY = auto()
    TARGET_DECODING = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class DraftProposal:
    """One tentative token and the exact prompt pages visible to it."""

    token_id: int
    visible_priority_pages: tuple[int, ...]


@dataclass
class ProgressivePDRequestState:
    """Fail-closed orchestration state for one progressive P/D request.

    Transfer completions may arrive out of order, but draft visibility advances
    only through the longest completed prefix of ``priority_pages``. No token is
    exposed until exactly one final verifier runs after every page is complete.
    """

    priority_pages: tuple[int, ...]
    anchor_pages: int
    max_draft_tokens: int
    phase: ProgressivePDPhase = ProgressivePDPhase.WAIT_ANCHOR
    completed_pages: set[int] = field(default_factory=set)
    proposals: list[DraftProposal] = field(default_factory=list)
    external_token_ids: list[int] = field(default_factory=list)
    verifier_calls: int = 0

    def __post_init__(self) -> None:
        if not self.priority_pages:
            raise ValueError("priority page order cannot be empty")
        if len(self.priority_pages) != len(set(self.priority_pages)):
            raise ValueError("priority page order contains duplicates")
        if not 0 < self.anchor_pages <= len(self.priority_pages):
            raise ValueError("anchor_pages must lie in [1, number of pages]")
        if self.max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")

    @property
    def visible_priority_pages(self) -> tuple[int, ...]:
        """Return the completed contiguous prefix in priority order."""
        visible = []
        for page in self.priority_pages:
            if page not in self.completed_pages:
                break
            visible.append(page)
        return tuple(visible)

    @property
    def visible_fraction(self) -> float:
        return len(self.visible_priority_pages) / len(self.priority_pages)

    def record_page_completion(self, pages: set[int]) -> None:
        """Apply a monotonic connector completion event."""
        if self.phase in {
            ProgressivePDPhase.READY_TO_VERIFY,
            ProgressivePDPhase.TARGET_DECODING,
            ProgressivePDPhase.FINISHED,
        }:
            raise RuntimeError("page completion arrived after full-KV readiness")
        unknown = pages - set(self.priority_pages)
        if unknown:
            raise ValueError(f"completion contains unknown pages: {sorted(unknown)}")
        if pages & self.completed_pages:
            raise ValueError("connector reported a page completion twice")
        self.completed_pages.update(pages)
        if len(self.completed_pages) == len(self.priority_pages):
            self.phase = ProgressivePDPhase.READY_TO_VERIFY
        elif len(self.visible_priority_pages) >= self.anchor_pages:
            self.phase = ProgressivePDPhase.DRAFT_WHILE_LOADING

    def record_draft_token(self, token_id: int) -> None:
        """Record an internal-only sparse-draft proposal."""
        if self.phase != ProgressivePDPhase.DRAFT_WHILE_LOADING:
            raise RuntimeError("drafting is allowed only after S1 and before full KV")
        if len(self.proposals) >= self.max_draft_tokens:
            raise RuntimeError("draft token limit reached")
        self.proposals.append(
            DraftProposal(
                token_id=token_id,
                visible_priority_pages=self.visible_priority_pages,
            )
        )

    def record_final_verification(
        self, accepted_prefix: int, correction_token: int
    ) -> None:
        """Commit only the immutable full-KV verifier's result."""
        if self.phase != ProgressivePDPhase.READY_TO_VERIFY:
            raise RuntimeError("final verification requires complete prompt KV")
        if self.verifier_calls:
            raise RuntimeError("the immutable verifier may run only once")
        if not 0 <= accepted_prefix <= len(self.proposals):
            raise ValueError("accepted prefix exceeds the proposal sequence")
        self.verifier_calls = 1
        self.external_token_ids.extend(
            proposal.token_id for proposal in self.proposals[:accepted_prefix]
        )
        self.external_token_ids.append(correction_token)
        self.phase = ProgressivePDPhase.TARGET_DECODING

    def finish(self) -> None:
        if self.phase != ProgressivePDPhase.TARGET_DECODING:
            raise RuntimeError("only target decoding may finish a request")
        self.phase = ProgressivePDPhase.FINISHED
