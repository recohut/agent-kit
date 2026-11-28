"""One service-desk turn. Host owns the customer, gate, citations, and terminal reason."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from insurance_desk.config import CUSTOMER_ID, ROUTER_CONFIDENCE_FLOOR
from insurance_desk.contracts import (
    Envelope,
    Lane,
    RouterDecision,
    SpecialistAnswer,
    TerminalReason,
    TurnDeps,
    TurnResult,
)
from insurance_desk.gate import mutation_request
from insurance_desk.router import router_agent, router_prompt
from insurance_desk.specialists import specialist_for
from insurance_desk.store import Snapshot, Store

NO_EVIDENCE_MESSAGE = "I do not have grounded evidence for that on this desk."
REFUSAL_MESSAGE = (
    "This desk is read-only. It can explain a policy, claim, or bill, "
    "but it cannot take a payment, file a claim, or change cover. A person will continue if needed."
)
GAP_MESSAGE = "I am handing this to a person on the service desk."


def _usage_counts(result: object) -> dict[str, int]:
    raw = getattr(result, "usage", None)
    if raw is None:
        return {"requests": 0, "input_tokens": 0, "output_tokens": 0}
    return {
        "requests": int(getattr(raw, "requests", 0) or 0),
        "input_tokens": int(getattr(raw, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(raw, "output_tokens", 0) or 0),
    }


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: left.get(key, 0) + right.get(key, 0) for key in ("requests", "input_tokens", "output_tokens")}


def _finish(
    store: Store,
    customer_id: str,
    message: str,
    correlation_id: str,
    terminal: TerminalReason,
    text: str,
    *,
    lane: Lane | None = None,
    citation_ids: tuple[str, ...] = (),
    tools_used: tuple[str, ...] = (),
    ticket_id: str | None = None,
    usage: dict[str, int] | None = None,
    model_calls: int = 0,
) -> TurnResult:
    store.record_audit(correlation_id, customer_id, message, None if lane is None else lane.value, terminal, model_calls)
    return TurnResult(
        correlation_id=correlation_id,
        terminal_reason=terminal,
        message=text,
        lane=lane,
        citation_ids=citation_ids,
        tools_used=tools_used,
        ticket_id=ticket_id,
        usage=usage or {"requests": 0, "input_tokens": 0, "output_tokens": 0},
        model_calls=model_calls,
    )


def _envelope(decision: RouterDecision, snapshot: Snapshot, message: str, correlation_id: str) -> Envelope:
    return Envelope(
        correlation_id=correlation_id,
        goal=f"Answer the {decision.lane.value} question with grounded records only",
        intent_summary=decision.intent_summary,
        resolved_ids=dict(decision.resolved_ids),
        user_turn=message,
        customer_id=snapshot.customer.id,
        tenant=snapshot.customer.tenant,
    )


def keep_citations(proposed: list[str], evidence_ids: set[str]) -> tuple[str, ...]:
    return tuple(item for item in proposed if item in evidence_ids)


async def run_turn(
    store: Store,
    message: str,
    *,
    router_model: Model,
    specialist_model: Model,
    customer_id: str = CUSTOMER_ID,
) -> TurnResult:
    """Run one bound-customer turn. A mutation request never reaches a model."""
    correlation_id = uuid.uuid4().hex
    if mutation_request(message):
        ticket_id = store.open_ticket(customer_id, correlation_id, "read_only_refused", message)
        return _finish(
            store,
            customer_id,
            message,
            correlation_id,
            "read_only_refused",
            REFUSAL_MESSAGE,
            ticket_id=ticket_id,
        )
    snapshot = store.snapshot(customer_id)
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
    try:
        routed = await router_agent(router_model).run(
            router_prompt(snapshot, message),
            usage_limits=UsageLimits(request_limit=1),
        )
    except Exception as exc:
        return _finish(
            store,
            customer_id,
            message,
            correlation_id,
            "provider_failure",
            f"Router failed: {type(exc).__name__}",
        )
    usage = _merge_usage(usage, _usage_counts(routed))
    decision: RouterDecision = routed.output
    model_calls = usage["requests"]
    if decision.lane is Lane.clarify:
        return _finish(
            store,
            customer_id,
            message,
            correlation_id,
            "clarify",
            decision.intent_summary,
            lane=Lane.clarify,
            usage=usage,
            model_calls=model_calls,
        )
    if decision.lane is Lane.escalate or decision.confidence < ROUTER_CONFIDENCE_FLOOR:
        ticket_id = store.open_ticket(customer_id, correlation_id, "escalate", message)
        return _finish(
            store,
            customer_id,
            message,
            correlation_id,
            "escalate",
            GAP_MESSAGE,
            lane=Lane.escalate,
            ticket_id=ticket_id,
            usage=usage,
            model_calls=model_calls,
        )
    deps = TurnDeps(store=store, customer_id=snapshot.customer.id, tenant=snapshot.customer.tenant)
    envelope = _envelope(decision, snapshot, message, correlation_id)
    try:
        specialist = await specialist_for(decision.lane, specialist_model).run(
            _specialist_prompt(envelope),
            deps=deps,
            usage_limits=UsageLimits(request_limit=3),
        )
    except Exception as exc:
        return _finish(
            store,
            customer_id,
            message,
            correlation_id,
            "provider_failure",
            f"Specialist failed: {type(exc).__name__}",
            lane=decision.lane,
            usage=usage,
            model_calls=model_calls,
        )
    usage = _merge_usage(usage, _usage_counts(specialist))
    model_calls = usage["requests"]
    answer: SpecialistAnswer = specialist.output
    citations = keep_citations(answer.citation_ids, deps.evidence_ids)
    if not citations:
        return _finish(
            store,
            customer_id,
            message,
            correlation_id,
            "no_evidence",
            NO_EVIDENCE_MESSAGE,
            lane=decision.lane,
            tools_used=tuple(deps.tools_used),
            usage=usage,
            model_calls=model_calls,
        )
    ticket_id = None
    terminal: TerminalReason = "specialist_answer"
    text = answer.message
    if answer.needs_human:
        ticket_id = store.open_ticket(customer_id, correlation_id, "needs_human", message)
        terminal = "escalate"
        text = f"{answer.message}\n{GAP_MESSAGE}"
    return _finish(
        store,
        customer_id,
        message,
        correlation_id,
        terminal,
        text,
        lane=decision.lane,
        citation_ids=citations,
        tools_used=tuple(deps.tools_used),
        ticket_id=ticket_id,
        usage=usage,
        model_calls=model_calls,
    )


def _specialist_prompt(envelope: Envelope) -> str:
    resolved = ", ".join(f"{key}={value}" for key, value in envelope.resolved_ids.items()) or "none"
    return (
        f"Correlation {envelope.correlation_id}. Goal: {envelope.goal}. "
        f"Intent: {envelope.intent_summary}. Resolved ids: {resolved}. "
        f"Authoritative customer: {envelope.customer_id}. Tenant: {envelope.tenant}. "
        f"Constraint: {envelope.constraint}. User turn: {envelope.user_turn}"
    )


def snapshot_line(snapshot: Snapshot) -> str:
    return f"{snapshot.customer.name} · {snapshot.customer.id} · tenant {snapshot.customer.tenant}"


def result_payload(result: TurnResult, *, model_name: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "correlation_id": result.correlation_id,
        "terminal_reason": result.terminal_reason,
        "lane": None if result.lane is None else result.lane.value,
        "message": result.message,
        "citation_ids": list(result.citation_ids),
        "tools_used": list(result.tools_used),
        "ticket_id": result.ticket_id,
        "usage": result.usage,
        "model_calls": result.model_calls,
    }
    if model_name is not None:
        payload["model"] = model_name
    return payload
