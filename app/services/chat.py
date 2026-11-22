"""AI chat service: swappable Anthropic/OpenAI providers with SSE streaming.

The system prompt forces educational framing, injects live market context,
and forbids profit promises. The API key is read from .env only.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Meas Pro Assistant, an educational gold-market analyst. "
    "You discuss XAU/USD (gold) analysis: technical indicators, macro drivers "
    "(DXY, US Treasury yields, equities), the app's current signal, and its "
    "detected trading concepts (FVG, order block, BOS, CHoCH, liquidity sweeps, "
    "premium/discount, OTE, killzones, market structure, price-action patterns, "
    "multi-timeframe bias and macro context). When asked about a concept, explain "
    "it plainly using the injected detections and their levels; never invent a "
    "concept that is not in the context.\n\n"
    "STRICT RULES:\n"
    "1. Never promise or guarantee profit, returns, or outcomes.\n"
    "2. Always mention that trading gold is risky and signals can be wrong.\n"
    "3. End substantive answers with: 'Educational analysis only. Not financial "
    "advice. Never risk money you can't afford to lose.'\n"
    "4. Use the injected live market context provided in the user message; if "
    "context is missing or stale, say so explicitly.\n"
    "5. Be concise, concrete, and use numbers from the context (price, RSI, "
    "signal, grade, confidence, DXY change, detected concept levels).\n"
    "6. If asked to 'buy now' or for a guaranteed trade, decline to give "
    "personalized financial advice and restate the risk disclaimer."
)


def build_context(snapshot: dict[str, Any]) -> str:
    """Serialize live market context for injection into the prompt."""
    try:
        return json.dumps(snapshot, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(snapshot)


def _model_name() -> str:
    if config.LLM_MODEL:
        return config.LLM_MODEL
    if config.LLM_PROVIDER == "anthropic":
        return config.ANTHROPIC_DEFAULT_MODEL
    return config.OPENAI_DEFAULT_MODEL


async def stream_chat(message: str, context: dict[str, Any],
                      history: list[dict[str, str]] | None = None
                      ) -> AsyncIterator[str]:
    """Yield text deltas from the configured provider.

    Raises ``RuntimeError`` when no provider/key is configured so the API
    layer can return a friendly error to the UI.
    """
    if not config.llm_configured():
        raise RuntimeError(
            "LLM not configured. Set LLM_PROVIDER (anthropic|openai) and "
            "LLM_API_KEY in backend/.env to enable AI chat."
        )

    user_content = (
        f"<live_market_context>\n{build_context(context)}\n</live_market_context>\n\n"
        f"User question: {message}"
    )
    messages: list[dict[str, Any]] = []
    for h in (history or [])[-8:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_content})

    if config.LLM_PROVIDER == "anthropic":
        async for chunk in _stream_anthropic(messages):
            yield chunk
    else:
        async for chunk in _stream_openai(messages):
            yield chunk


async def _stream_anthropic(messages: list[dict[str, Any]]) -> AsyncIterator[str]:
    """Stream from Anthropic Messages API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": config.LLM_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": _model_name(),
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:500]
                raise RuntimeError(f"Anthropic API error {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")
                elif obj.get("type") == "error":
                    raise RuntimeError(f"Anthropic stream error: {obj}")


async def _stream_openai(messages: list[dict[str, Any]]) -> AsyncIterator[str]:
    """Stream from OpenAI Chat Completions API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _model_name(),
        "max_tokens": 1024,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:500]
                raise RuntimeError(f"OpenAI API error {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
