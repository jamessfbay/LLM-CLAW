from __future__ import annotations

import hashlib
import os

from llm_claw.models import AcquisitionTask, ProviderName


def _configured_model(provider: ProviderName) -> str:
    if provider == "openai_web_search":
        return os.getenv("LLM_CLAW_OPENAI_MODEL", "gpt-5.4-mini")
    if provider == "gemini":
        return os.getenv("LLM_CLAW_GEMINI_MODEL", "gemini-3.8-flash")
    return "provider-default"


def build_cache_key(
    task: AcquisitionTask,
    query: str,
    provider: ProviderName,
    date_window: str | None = None,
    model: str | None = None,
) -> str:
    parts = [
        task.domain,
        task.entity.display_name,
        query,
        date_window or task.freshness,
        provider,
        model or _configured_model(provider),
    ]
    raw = "|".join(part.strip().lower() for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"{task.domain}:{provider}:{digest}"
