"""LLM abstraction with deterministic offline fallback."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .config import settings

log = logging.getLogger("cortex.llm")


class LLMClient:
    def __init__(self) -> None:
        self.provider = self._resolve_provider()
        self._client: Any = None
        self._init_client()

    def _resolve_provider(self) -> str:
        provider = settings.llm_provider
        if provider == "auto":
            if settings.anthropic_api_key:
                return "anthropic"
            if settings.gemini_api_key:
                return "gemini"
            return "offline"
        return provider if provider in {"anthropic", "gemini", "offline"} else "offline"

    def _init_client(self) -> None:
        try:
            if self.provider == "anthropic" and settings.anthropic_api_key:
                import anthropic
                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                return
            if self.provider == "gemini" and settings.gemini_api_key:
                from google import genai
                self._client = genai.Client(api_key=settings.gemini_api_key)
                return
            self.provider = "offline"
        except Exception as exc:
            log.warning("LLM initialization failed; using offline mode: %s", exc)
            self.provider = "offline"
            self._client = None

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        if self.provider == "offline" or self._client is None:
            return ""
        try:
            if self.provider == "anthropic":
                response = self._client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(
                    block.text for block in response.content
                    if getattr(block, "type", "") == "text"
                )
            response = self._client.models.generate_content(
                model=settings.gemini_model,
                contents=f"{system}\n\n{prompt}",
            )
            return getattr(response, "text", "") or ""
        except Exception as exc:
            log.warning("LLM call failed; using offline fallback: %s", exc)
            return ""

    def complete_json(
        self, system: str, prompt: str, fallback: dict[str, Any], max_tokens: int = 1024
    ) -> dict[str, Any]:
        raw = self.complete(
            system + " Respond with ONLY valid JSON, no prose.",
            prompt,
            max_tokens,
        )
        parsed = _extract_json(raw)
        return parsed if parsed is not None else fallback


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


_singleton: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton
