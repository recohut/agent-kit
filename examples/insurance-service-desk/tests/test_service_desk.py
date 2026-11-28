"""Five host-owned service-desk cases. FunctionModel only; no Grok key."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from insurance_desk.config import CUSTOMER_ID
from insurance_desk.session import run_turn
from insurance_desk.store import Store
from insurance_desk.tools import (
    BILLING_TOOLS,
    CLAIMS_TOOLS,
    KNOWLEDGE_TOOLS,
    POLICY_TOOLS,
    WRITE_TOOL_NAMES,
)


def _store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "desk.sqlite")
    store.initialize()
    return store


def _output_response(info: AgentInfo, payload: dict[str, Any], call_id: str) -> ModelResponse:
    if info.output_tools:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload, call_id)])
    return ModelResponse(parts=[TextPart(json.dumps(payload))])


def _tool_payload(part: ToolReturnPart) -> dict[str, Any]:
    raw = part.content
    if isinstance(raw, str):
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    return cast(dict[str, Any], raw)


def _router(lane: str, confidence: float = 0.95, resolved: dict[str, str] | None = None, summary: str = "route") -> FunctionModel:
    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _output_response(
            info,
            {
                "lane": lane,
                "intent_summary": summary,
                "resolved_ids": resolved or {},
                "confidence": confidence,
            },
            "router-1",
        )

    return FunctionModel(respond)


def test_payment_sentence_never_reaches_a_model(tmp_path: Path) -> None:
    store = _store(tmp_path)
    calls: list[str] = []

    def boom(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        calls.append("called")
        raise AssertionError("model must not run on a mutation request")

    model = FunctionModel(boom)
    result = asyncio.run(
        run_turn(
            store,
            "Take the premium payment.",
            router_model=model,
            specialist_model=model,
        )
    )
    offered = POLICY_TOOLS + CLAIMS_TOOLS + BILLING_TOOLS + KNOWLEDGE_TOOLS
    assert calls == []
    assert result.model_calls == 0
    assert result.terminal_reason == "read_only_refused"
    assert result.ticket_id is not None
    assert WRITE_TOOL_NAMES == ()
    assert store.open_tickets(CUSTOMER_ID) == (result.ticket_id,)
    assert "pay" not in offered and "file_claim" not in offered and "update_policy" not in offered


def test_claims_specialist_can_read_only_the_session_claim(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seen: list[set[str]] = []

    def specialist(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append({tool.name for tool in info.function_tools})
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "get_claim",
                        {"claim_id": "CLM-1001", "customer_id": "cust-other"},
                        "claim-1",
                    )
                ]
            )
        payload = _tool_payload(returns[-1])
        assert payload["id"] == "CLM-1001"
        assert payload["customer_id"] == CUSTOMER_ID
        return _output_response(
            info,
            {
                "message": f"Claim {payload['id']} is {payload['status']}.",
                "citation_ids": [payload["id"]],
                "needs_human": False,
            },
            "out-1",
        )

    result = asyncio.run(
        run_turn(
            store,
            "What is the status of claim CLM-1001?",
            router_model=_router("claims", resolved={"claim_id": "CLM-1001"}),
            specialist_model=FunctionModel(specialist),
        )
    )
    assert seen
    assert seen[0] == set(CLAIMS_TOOLS)
    assert "get_policy" not in seen[0]
    assert "get_bill" not in seen[0]
    assert result.terminal_reason == "specialist_answer"
    assert result.citation_ids == ("CLM-1001",)
    assert result.tools_used == ("get_claim",)


def test_escalate_and_clarify_skip_the_specialist(tmp_path: Path) -> None:
    store = _store(tmp_path)
    specialist_calls: list[str] = []

    def boom(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        specialist_calls.append("called")
        raise AssertionError("specialist must not run")

    specialist = FunctionModel(boom)
    escalated = asyncio.run(
        run_turn(
            store,
            "I want to speak to a person.",
            router_model=_router("escalate", confidence=0.9, summary="user asked for a person"),
            specialist_model=specialist,
        )
    )
    clarified = asyncio.run(
        run_turn(
            store,
            "Update me on my claim",
            router_model=_router("clarify", summary="Which claim, CLM-1001 or CLM-1002?"),
            specialist_model=specialist,
        )
    )
    assert specialist_calls == []
    assert escalated.terminal_reason == "escalate"
    assert escalated.ticket_id is not None
    assert store.open_tickets(CUSTOMER_ID) == (escalated.ticket_id,)
    assert clarified.terminal_reason == "clarify"
    assert clarified.ticket_id is None
    assert "CLM-1001" in clarified.message and "CLM-1002" in clarified.message


def test_knowledge_search_respects_tenant_and_typed_empty_results(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hit = store.search_passages("demo-motor", "excess")
    assert hit.status == "ok"
    ids = {passage.id for passage in hit.passages}
    assert "K-EXCESS-1" in ids
    assert "K-FLEET-EXCESS-1" not in ids
    passage = next(item for item in hit.passages if item.id == "K-EXCESS-1")
    assert passage.version and passage.source and passage.heading and passage.fingerprint
    assert passage.tenant == "demo-motor"
    empty = store.search_passages("demo-motor", "zzzznosuchterm")
    assert empty.status == "no_evidence"
    assert empty.passages == ()
    with store.connect(write=True) as connection:
        connection.execute("DROP TABLE knowledge_fts")
    broken = store.search_passages("demo-motor", "excess")
    assert broken.status == "retrieval_unavailable"
    assert broken.passages == ()


def test_ungrounded_citation_is_dropped(tmp_path: Path) -> None:
    store = _store(tmp_path)

    def specialist(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("get_claim", {"claim_id": "CLM-1001"}, "claim-1")])
        return _output_response(
            info,
            {
                "message": "The excess is waived.",
                "citation_ids": ["K-FAKE"],
                "needs_human": False,
            },
            "out-1",
        )

    result = asyncio.run(
        run_turn(
            store,
            "What is the status of claim CLM-1001?",
            router_model=_router("claims", resolved={"claim_id": "CLM-1001"}),
            specialist_model=FunctionModel(specialist),
        )
    )
    assert result.terminal_reason == "no_evidence"
    assert result.citation_ids == ()
    assert "waived" not in result.message
