"""Typed turn artifacts. Model output is a proposal; host code owns authority and terminal reasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from insurance_desk.store import Store


class Lane(StrEnum):
    policy = "policy"
    claims = "claims"
    billing = "billing"
    knowledge = "knowledge"
    clarify = "clarify"
    escalate = "escalate"


TerminalReason = Literal[
    "read_only_refused",
    "specialist_answer",
    "clarify",
    "escalate",
    "no_evidence",
    "provider_failure",
    "retrieval_unavailable",
]


class RouterDecision(BaseModel):
    """Router proposal. Host still binds the customer and may skip the specialist."""

    lane: Lane
    intent_summary: str = Field(min_length=1, max_length=400)
    resolved_ids: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)


class SpecialistAnswer(BaseModel):
    """Specialist proposal. Host keeps citations only when a tool on this turn returned them."""

    message: str = Field(min_length=1, max_length=2000)
    citation_ids: list[str] = Field(default_factory=list)
    needs_human: bool = False


@dataclass(frozen=True)
class Envelope:
    """Host-authored specialist brief. Authority fields come from the session, not the model."""

    correlation_id: str
    goal: str
    intent_summary: str
    resolved_ids: dict[str, str]
    user_turn: str
    customer_id: str
    tenant: str
    constraint: str = "read-only: answer and guide; the customer decides"
    answer_schema: str = "SpecialistAnswer: message, citation_ids, needs_human"


@dataclass
class TurnDeps:
    """Per-turn specialist dependencies. Customer and tenant are session facts."""

    store: Store
    customer_id: str
    tenant: str
    evidence_ids: set[str] = field(default_factory=set)
    tools_used: list[str] = field(default_factory=list)

    def note_id(self, value: str) -> None:
        self.evidence_ids.add(value)

    def note_tool(self, name: str) -> None:
        self.tools_used.append(name)


@dataclass(frozen=True)
class TurnResult:
    correlation_id: str
    terminal_reason: TerminalReason
    message: str
    lane: Lane | None
    citation_ids: tuple[str, ...]
    tools_used: tuple[str, ...]
    ticket_id: str | None
    usage: dict[str, int]
    model_calls: int
