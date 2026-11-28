"""CLI chat for the bound synthetic customer. Desktop terminal only."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from insurance_desk.config import CUSTOMER_ID, RuntimeSettings
from insurance_desk.model import grok_model
from insurance_desk.session import result_payload, run_turn, snapshot_line
from insurance_desk.store import Store


def _store(settings: RuntimeSettings, data_dir: str | None) -> Store:
    root = Path(data_dir) if data_dir else settings.data_dir
    store = Store(root / "desk.sqlite")
    store.initialize()
    return store


async def _ask(store: Store, message: str, settings: RuntimeSettings) -> int:
    model = grok_model(settings)
    result = await run_turn(store, message, router_model=model, specialist_model=model)
    print(json.dumps(result_payload(result, model_name=settings.grok_model), indent=2))
    return 0


async def _chat(store: Store, settings: RuntimeSettings) -> int:
    model = grok_model(settings)
    print(snapshot_line(store.snapshot(CUSTOMER_ID)))
    print("Read-only desk. Empty line exits.")
    while True:
        try:
            message = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not message:
            return 0
        result = await run_turn(store, message, router_model=model, specialist_model=model)
        print(json.dumps(result_payload(result, model_name=settings.grok_model), indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="One turn for the bound customer")
    ask.add_argument("message")

    sub.add_parser("chat", help="Interactive turns for the bound customer")

    args = parser.parse_args(argv)
    settings = RuntimeSettings.load()
    store = _store(settings, args.data_dir)
    command = args.command or "chat"
    if command == "ask":
        return asyncio.run(_ask(store, args.message, settings))
    return asyncio.run(_chat(store, settings))
