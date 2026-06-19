"""LLM abstraction with graceful offline fallback.

Cortex must run on localhost with zero external keys. So every agent that
"reasons" goes through this layer, which:
  * uses Anthropic if ANTHROPIC_API_KEY is set,
  * else Gemini if GEMINI_API_KEY is set,
  * else a deterministic offline stub so the full loop still runs.

`complete_json` always returns a dict; on any failure it falls back so the
research loop never crashes because of the LLM.
"""
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
        p = settings.llm_provider.lower()
        if p == "auto":
            if settings.anthropic_api_key:
                return "anthropic"
            if settings.gemini_api_key:
                return "gemini"
            return "offline"
        return p

    def _init_client(self) -> None:
        try:
            if self.provider == "anthropic" and settings.anthropic_api_key:
                import anthropic

                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            elif self.provider == "gemini" and settings.gemini_api_key:
                import google.generativeai as genai

                genai.configure(api_key=settings.gemini_api_key)
                self._client = genai.GenerativeModel(settings.gemini_model)
            else:
                self.provider = "offline"
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("LLM init failed (%s); using offline mode", exc)
            self.provider = "offline"

    # ------------------------------------------------------------------ #
    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        if self.provider == "offline" or self._client is None:
            return ""
        try:
            if self.provider == "anthropic":
                msg = self._client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            if self.provider == "gemini":
                resp = self._client.generate_content(f"{system}\n\n{prompt}")
                return resp.text or ""
        except Exception as exc:
            log.warning("LLM call failed (%s); offline fallback", exc)
        return ""

    def complete_json(
        self, system: str, prompt: str, fallback: dict[str, Any], max_tokens: int = 1024
    ) -> dict[str, Any]:
        raw = self.complete(
            system + " Respond with ONLY valid minified JSON, no prose.",
            prompt,
            max_tokens,
        )
        parsed = _extract_json(raw)
        if parsed is None:
            return fallback
        return parsed


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


_singleton: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton
