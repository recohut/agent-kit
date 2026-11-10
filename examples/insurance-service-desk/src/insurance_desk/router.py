"""Router agent. Classifies a turn; it has no tools and cannot change records."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from insurance_desk.contracts import RouterDecision
from insurance_desk.store import Snapshot

ROUTER_INSTRUCTIONS = """\
You route a read-only insurance service-desk turn for one bound customer.
Pick one lane: policy, claims, billing, knowledge, clarify, or escalate.
Use claims for a named or unique claim status. Use policy for cover and excess.
Use billing for amount due and due date. Use knowledge for wording with no record id.
Use clarify when two claims, policies, or bills could match and the user did not name one.
Use escalate when the user asks for a person or the request is outside this desk.
Do not invent ids. Confidence is 0 to 1. You have no tools.
"""


def router_agent(model: Model) -> Agent[None, RouterDecision]:
    return Agent(
        model,
        output_type=RouterDecision,
        retries=0,
        instructions=ROUTER_INSTRUCTIONS,
    )


def router_prompt(snapshot: Snapshot, message: str) -> str:
    claims = ", ".join(f"{item.id} ({item.status})" for item in snapshot.claims) or "none"
    policies = ", ".join(item.id for item in snapshot.policies) or "none"
    bills = ", ".join(f"{item.id} ({item.status})" for item in snapshot.bills) or "none"
    return (
        f"Customer {snapshot.customer.name} ({snapshot.customer.id}), "
        f"tenant {snapshot.customer.tenant}.\n"
        f"Policies: {policies}. Claims: {claims}. Bills: {bills}.\n"
        f"User: {message}"
    )


async def route(model: Model, snapshot: Snapshot, message: str) -> RouterDecision:
    result = await router_agent(model).run(
        router_prompt(snapshot, message),
        usage_limits=UsageLimits(request_limit=1),
    )
    return result.output
