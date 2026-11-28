"""Desk rules for the commercial placement article.

A field is stored only when its span occurs in the named source. Appetite is a
function of those fields. The outbox stays empty until that exact action id is
approved, and the same id does not write a second row.

This file does not import the private Recohut kit. The durable run store in the
labs package is a separate owner. These functions are the desk rules the article
measured.
"""

from __future__ import annotations

from dataclasses import dataclass


class DeskRefused(Exception):
    """The host refused the write."""


EXCLUDED_OCCUPANCIES = frozenset({"tyre recycling", "tire recycling"})
ALLOWED_TERRITORIES = frozenset({"united kingdom", "uk", "england", "scotland", "wales"})
MAX_TIV_GBP = 25_000_000


@dataclass(frozen=True)
class Cited:
    name: str
    value: str
    source_id: str
    span: str


@dataclass(frozen=True)
class Appetite:
    decision: str
    reason: str


@dataclass
class OutboxRow:
    action_id: str
    market_id: str
    body: str


class Desk:
    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents
        self.fields: dict[str, Cited] = {}
        self.drafts: dict[str, str] = {}
        self.outbox: dict[str, OutboxRow] = {}
        self.pending: dict[str, str] = {}

    def apply(self, item: Cited) -> None:
        body = self.documents.get(item.source_id)
        if body is None or not item.span or item.span not in body:
            raise DeskRefused(f"span is not in source {item.source_id}")
        self.fields[item.name] = item

    def appetite(self) -> Appetite:
        values = {name: item.value.strip().lower() for name, item in self.fields.items()}
        occupancy = values.get("occupancy", "")
        if occupancy in EXCLUDED_OCCUPANCIES:
            return Appetite("decline", f"occupancy {occupancy!r} is excluded")
        territory = values.get("territory", "")
        if territory and territory not in ALLOWED_TERRITORIES:
            return Appetite("decline", "territory is outside appetite")
        raw = values.get("tiv", "").replace(",", "").replace("£", "").replace("gbp", "")
        try:
            tiv = int(float(raw)) if raw else None
        except ValueError:
            tiv = None
        if not occupancy or not territory or tiv is None:
            return Appetite("decline", "occupancy, territory and TIV are required")
        if tiv > MAX_TIV_GBP:
            return Appetite("decline", "TIV exceeds the ceiling")
        return Appetite("place", "within appetite")

    def prepare(self, market_id: str, body: str) -> None:
        if not market_id.strip() or not body.strip():
            raise DeskRefused("market and body are required")
        self.drafts[market_id] = body

    def propose_send(self, action_id: str, market_id: str) -> str:
        """Record a pending send. Does not write the outbox."""
        if self.appetite().decision != "place":
            raise DeskRefused("appetite does not permit a send")
        if market_id not in self.drafts:
            raise DeskRefused(f"no draft for {market_id}")
        self.pending[action_id] = market_id
        return "waiting"

    def approve(self, action_id: str) -> OutboxRow:
        if action_id in self.outbox:
            return self.outbox[action_id]
        market_id = self.pending.get(action_id)
        if market_id is None:
            raise DeskRefused("no pending send for that action")
        if self.appetite().decision != "place":
            raise DeskRefused("appetite does not permit a send")
        row = OutboxRow(action_id, market_id, self.drafts[market_id])
        self.outbox[action_id] = row
        return row
