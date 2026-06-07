from __future__ import annotations

import hashlib

from llm_claw.models import AcquisitionTask, ProviderName


def build_cache_key(task: AcquisitionTask, query: str, provider: ProviderName, date_window: str | None = None) -> str:
    parts = [
        task.domain,
        task.entity.display_name,
        query,
        date_window or task.freshness,
        provider,
    ]
    raw = "|".join(part.strip().lower() for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"{task.domain}:{provider}:{digest}"
