"""The four host checks named in the placement article. No model call."""

from __future__ import annotations

import unittest
from pathlib import Path

from desk import Cited, Desk, DeskRefused

ROOT = Path(__file__).resolve().parent


def northline() -> Desk:
    text = (ROOT / "fixtures/northline/statement_of_fact.txt").read_text(encoding="utf-8")
    desk = Desk({"sof": text})
    desk.apply(Cited("occupancy", "warehouse", "sof", "warehouse"))
    desk.apply(Cited("territory", "United Kingdom", "sof", "United Kingdom"))
    desk.apply(Cited("tiv", "8400000", "sof", "£8,400,000"))
    return desk


class DeskRules(unittest.TestCase):
    def test_missing_span_is_rejected(self) -> None:
        text = (ROOT / "fixtures/northline/statement_of_fact.txt").read_text(encoding="utf-8")
        desk = Desk({"sof": text})
        with self.assertRaises(DeskRefused):
            desk.apply(Cited("occupancy", "warehouse", "sof", "this span is not in the statement"))
        self.assertEqual(desk.fields, {})

    def test_excluded_occupancy_cannot_send(self) -> None:
        text = (ROOT / "fixtures/excluded_occupancy/statement_of_fact.txt").read_text(encoding="utf-8")
        desk = Desk({"sof": text})
        desk.apply(Cited("occupancy", "tyre recycling", "sof", "tyre recycling"))
        desk.apply(Cited("territory", "United Kingdom", "sof", "United Kingdom"))
        desk.apply(Cited("tiv", "3200000", "sof", "£3,200,000"))
        self.assertEqual(desk.appetite().decision, "decline")
        desk.prepare("meridian", "should not send")
        with self.assertRaises(DeskRefused):
            desk.propose_send("send-1", "meridian")
        self.assertEqual(desk.outbox, {})

    def test_unapproved_send_is_empty_and_approval_is_once(self) -> None:
        desk = northline()
        desk.prepare("meridian", "Northline warehouse, United Kingdom, TIV 8400000")
        self.assertEqual(desk.propose_send("send-1", "meridian"), "waiting")
        self.assertEqual(desk.outbox, {})
        row = desk.approve("send-1")
        again = desk.approve("send-1")
        self.assertEqual(list(desk.outbox), ["send-1"])
        self.assertIs(row, again)

    def test_pending_send_keeps_the_same_action(self) -> None:
        desk = northline()
        desk.prepare("meridian", "Northline warehouse")
        desk.propose_send("send-9", "meridian")
        self.assertEqual(desk.pending["send-9"], "meridian")
        self.assertNotIn("send-9", desk.outbox)


if __name__ == "__main__":
    unittest.main()
