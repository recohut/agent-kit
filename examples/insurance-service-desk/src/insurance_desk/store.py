"""SQLite owner for customer facts, knowledge passages, tickets, and turn audit."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from insurance_desk.config import seed_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS policies (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    product TEXT NOT NULL,
    status TEXT NOT NULL,
    excess_gbp INTEGER NOT NULL,
    cover_summary TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_on TEXT NOT NULL,
    summary TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE IF NOT EXISTS bills (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    amount_gbp TEXT NOT NULL,
    due_on TEXT NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE IF NOT EXISTS knowledge_passages (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    source TEXT NOT NULL,
    heading TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    applicability TEXT NOT NULL,
    tenant TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    body TEXT NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 1
);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    id,
    heading,
    body,
    tokenize = 'porter'
);
CREATE TABLE IF NOT EXISTS handoff_tickets (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    user_turn TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_turns (
    id TEXT PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    user_turn TEXT NOT NULL,
    lane TEXT,
    terminal_reason TEXT NOT NULL,
    model_calls INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""

SearchStatus = Literal["ok", "no_evidence", "retrieval_unavailable"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fingerprint(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def fts_query(raw: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", raw)
    return " OR ".join(tokens)


@dataclass(frozen=True)
class Customer:
    id: str
    tenant: str
    name: str


@dataclass(frozen=True)
class PolicyRecord:
    id: str
    tenant: str
    customer_id: str
    product: str
    status: str
    excess_gbp: int
    cover_summary: str


@dataclass(frozen=True)
class ClaimRecord:
    id: str
    tenant: str
    customer_id: str
    policy_id: str
    status: str
    opened_on: str
    summary: str


@dataclass(frozen=True)
class BillRecord:
    id: str
    tenant: str
    customer_id: str
    policy_id: str
    amount_gbp: str
    due_on: str
    status: str


@dataclass(frozen=True)
class Passage:
    id: str
    version: str
    source: str
    heading: str
    effective_from: str
    effective_to: str | None
    applicability: str
    tenant: str
    fingerprint: str
    body: str
    synthetic: bool = True


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    passages: tuple[Passage, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class Snapshot:
    customer: Customer
    policies: tuple[PolicyRecord, ...]
    claims: tuple[ClaimRecord, ...]
    bills: tuple[BillRecord, ...]


class Store:
    """Checkout-owned SQLite file. Seed JSON is the committed corpus; this file is local."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if write:
            connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            if write:
                connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, corpus: Path | None = None) -> None:
        source = corpus or seed_path()
        with self.connect(write=True) as connection:
            connection.executescript(SCHEMA)
            existing = connection.execute("SELECT count(*) FROM customers").fetchone()[0]
            if existing:
                return
            payload = json.loads(source.read_text(encoding="utf-8"))
            self._seed(connection, payload)

    def _seed(self, connection: sqlite3.Connection, payload: dict[str, object]) -> None:
        for row in payload["customers"]:
            connection.execute(
                "INSERT INTO customers (id, tenant, name) VALUES (:id, :tenant, :name)",
                row,
            )
        for row in payload["policies"]:
            connection.execute(
                """INSERT INTO policies
                   (id, tenant, customer_id, product, status, excess_gbp, cover_summary)
                   VALUES (:id, :tenant, :customer_id, :product, :status, :excess_gbp, :cover_summary)""",
                row,
            )
        for row in payload["claims"]:
            connection.execute(
                """INSERT INTO claims
                   (id, tenant, customer_id, policy_id, status, opened_on, summary)
                   VALUES (:id, :tenant, :customer_id, :policy_id, :status, :opened_on, :summary)""",
                row,
            )
        for row in payload["bills"]:
            connection.execute(
                """INSERT INTO bills
                   (id, tenant, customer_id, policy_id, amount_gbp, due_on, status)
                   VALUES (:id, :tenant, :customer_id, :policy_id, :amount_gbp, :due_on, :status)""",
                row,
            )
        for row in payload["knowledge_passages"]:
            body = str(row["body"])
            connection.execute(
                """INSERT INTO knowledge_passages
                   (id, version, source, heading, effective_from, effective_to, applicability,
                    tenant, fingerprint, body, synthetic)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    row["id"],
                    row["version"],
                    row["source"],
                    row["heading"],
                    row["effective_from"],
                    row["effective_to"],
                    row["applicability"],
                    row["tenant"],
                    fingerprint(body),
                    body,
                ),
            )
            connection.execute(
                "INSERT INTO knowledge_fts (id, heading, body) VALUES (?, ?, ?)",
                (row["id"], row["heading"], body),
            )

    def customer(self, customer_id: str) -> Customer:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, tenant, name FROM customers WHERE id = ?",
                (customer_id,),
            ).fetchone()
        if row is None:
            raise KeyError(customer_id)
        return Customer(id=row["id"], tenant=row["tenant"], name=row["name"])

    def snapshot(self, customer_id: str) -> Snapshot:
        customer = self.customer(customer_id)
        with self.connect() as connection:
            policies = tuple(
                PolicyRecord(**dict(row))
                for row in connection.execute(
                    """SELECT id, tenant, customer_id, product, status, excess_gbp, cover_summary
                       FROM policies WHERE customer_id = ?""",
                    (customer_id,),
                )
            )
            claims = tuple(
                ClaimRecord(**dict(row))
                for row in connection.execute(
                    """SELECT id, tenant, customer_id, policy_id, status, opened_on, summary
                       FROM claims WHERE customer_id = ?""",
                    (customer_id,),
                )
            )
            bills = tuple(
                BillRecord(**dict(row))
                for row in connection.execute(
                    """SELECT id, tenant, customer_id, policy_id, amount_gbp, due_on, status
                       FROM bills WHERE customer_id = ?""",
                    (customer_id,),
                )
            )
        return Snapshot(customer=customer, policies=policies, claims=claims, bills=bills)

    def get_policy(self, customer_id: str, policy_id: str) -> PolicyRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id, tenant, customer_id, product, status, excess_gbp, cover_summary
                   FROM policies WHERE id = ? AND customer_id = ?""",
                (policy_id, customer_id),
            ).fetchone()
        return None if row is None else PolicyRecord(**dict(row))

    def get_claim(self, customer_id: str, claim_id: str) -> ClaimRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id, tenant, customer_id, policy_id, status, opened_on, summary
                   FROM claims WHERE id = ? AND customer_id = ?""",
                (claim_id, customer_id),
            ).fetchone()
        return None if row is None else ClaimRecord(**dict(row))

    def get_bill(self, customer_id: str, bill_id: str) -> BillRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id, tenant, customer_id, policy_id, amount_gbp, due_on, status
                   FROM bills WHERE id = ? AND customer_id = ?""",
                (bill_id, customer_id),
            ).fetchone()
        return None if row is None else BillRecord(**dict(row))

    def list_policies(self, customer_id: str) -> tuple[PolicyRecord, ...]:
        return self.snapshot(customer_id).policies

    def list_claims(self, customer_id: str) -> tuple[ClaimRecord, ...]:
        return self.snapshot(customer_id).claims

    def list_bills(self, customer_id: str) -> tuple[BillRecord, ...]:
        return self.snapshot(customer_id).bills

    def search_passages(self, tenant: str, query: str) -> SearchResult:
        """Tenant filter is applied in SQL before FTS ranking. Retrieved body is untrusted data."""
        match = fts_query(query)
        if not match:
            return SearchResult(status="no_evidence")
        try:
            with self.connect() as connection:
                rows = connection.execute(
                    """SELECT p.id, p.version, p.source, p.heading, p.effective_from, p.effective_to,
                              p.applicability, p.tenant, p.fingerprint, p.body, p.synthetic
                       FROM knowledge_passages p
                       WHERE p.tenant = ?
                         AND p.id IN (SELECT id FROM knowledge_fts WHERE knowledge_fts MATCH ?)""",
                    (tenant, match),
                ).fetchall()
        except sqlite3.Error as exc:
            return SearchResult(status="retrieval_unavailable", detail=type(exc).__name__)
        passages = tuple(
            Passage(
                id=row["id"],
                version=row["version"],
                source=row["source"],
                heading=row["heading"],
                effective_from=row["effective_from"],
                effective_to=row["effective_to"],
                applicability=row["applicability"],
                tenant=row["tenant"],
                fingerprint=row["fingerprint"],
                body=row["body"],
                synthetic=bool(row["synthetic"]),
            )
            for row in rows
        )
        if not passages:
            return SearchResult(status="no_evidence")
        return SearchResult(status="ok", passages=passages)

    def open_ticket(self, customer_id: str, correlation_id: str, reason: str, user_turn: str) -> str:
        ticket_id = f"tkt-{uuid.uuid4().hex[:12]}"
        with self.connect(write=True) as connection:
            connection.execute(
                """INSERT INTO handoff_tickets
                   (id, customer_id, correlation_id, reason, user_turn, status, opened_at)
                   VALUES (?, ?, ?, ?, ?, 'open', ?)""",
                (ticket_id, customer_id, correlation_id, reason, user_turn, utc_now()),
            )
        return ticket_id

    def open_tickets(self, customer_id: str) -> tuple[str, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM handoff_tickets WHERE customer_id = ? AND status = 'open'",
                (customer_id,),
            ).fetchall()
        return tuple(row["id"] for row in rows)

    def record_audit(
        self,
        correlation_id: str,
        customer_id: str,
        user_turn: str,
        lane: str | None,
        terminal_reason: str,
        model_calls: int,
    ) -> None:
        with self.connect(write=True) as connection:
            connection.execute(
                """INSERT INTO audit_turns
                   (id, correlation_id, customer_id, user_turn, lane, terminal_reason, model_calls, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"aud-{uuid.uuid4().hex[:12]}",
                    correlation_id,
                    customer_id,
                    user_turn,
                    lane,
                    terminal_reason,
                    model_calls,
                    utc_now(),
                ),
            )
