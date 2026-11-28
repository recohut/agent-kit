"""Host-owned mutation detector. Runs before any model call."""

from __future__ import annotations

import re

_PAYMENT = re.compile(
    r"^(take|process|collect|charge)\b.{0,48}\b(payment|premium|bill)\b",
    re.I | re.S,
)
_PAY = re.compile(r"^pay\b.{0,48}\b(premium|bill|balance|payment)\b", re.I | re.S)
_FILE_CLAIM = re.compile(r"^(file|open|submit|lodge)\b.{0,40}\bclaim\b", re.I | re.S)
_CHANGE_POLICY = re.compile(
    r"^(change|cancel|amend|increase|decrease)\b.{0,40}\b(policy|cover|coverage|limit|excess)\b",
    re.I | re.S,
)


def mutation_request(text: str) -> bool:
    """True when the turn asks this desk to pay, file, or change cover."""
    stripped = text.strip()
    return any(
        pattern.search(stripped)
        for pattern in (_PAYMENT, _PAY, _FILE_CLAIM, _CHANGE_POLICY)
    )
