"""Runtime paths and the Grok route. Secrets stay in the environment and ~/.env."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_GROK_MODEL = "grok-4.7"
HOME_ENV = Path.home() / ".env"
CUSTOMER_ID = "cust-maria"
ROUTER_CONFIDENCE_FLOOR = 0.6


def load_home_env(environ: dict[str, str] | None = None) -> None:
    """Load ``~/.env`` into the process environment without overwriting existing keys."""
    target = os.environ if environ is None else environ
    if not HOME_ENV.is_file():
        return
    for raw in HOME_ENV.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        name = key.strip()
        if not name or name in target:
            continue
        target[name] = value.strip().strip("'").strip('"')


def data_directory(environ: Mapping[str, str] | None = None) -> Path:
    """SQLite lives under ``.local/`` unless ``INSURANCE_DESK_DATA_DIR`` is set."""
    env = environ if environ is not None else os.environ
    raw = env.get("INSURANCE_DESK_DATA_DIR", "").strip()
    path = Path(raw).expanduser() if raw else PROJECT_ROOT / ".local"
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_path() -> Path:
    return PROJECT_ROOT / "data" / "seed.json"


@dataclass(frozen=True)
class RuntimeSettings:
    data_dir: Path
    grok_model: str
    grok_api_key: str | None
    xai_base_url: str = XAI_BASE_URL

    @classmethod
    def load(cls, environ: Mapping[str, str] | None = None) -> RuntimeSettings:
        env = dict(os.environ if environ is None else environ)
        load_home_env(env)
        key = env.get("GROK_API_KEY", "").strip() or None
        model = env.get("GROK_MODEL", "").strip() or DEFAULT_GROK_MODEL
        return cls(data_dir=data_directory(env), grok_model=model, grok_api_key=key)
