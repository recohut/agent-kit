"""Grok via the xAI OpenAI-compatible API. Key comes from ``GROK_API_KEY``."""

from __future__ import annotations

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from insurance_desk.config import RuntimeSettings


def grok_model(settings: RuntimeSettings) -> OpenAIChatModel:
    """Build the billed Grok route. Call only when a live model run is requested."""
    if not settings.grok_api_key:
        raise RuntimeError("GROK_API_KEY is missing from the environment and ~/.env")
    return OpenAIChatModel(
        settings.grok_model,
        provider=OpenAIProvider(api_key=settings.grok_api_key, base_url=settings.xai_base_url),
    )
