from __future__ import annotations

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, ProviderName


class ProviderRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def select_providers(self, task: AcquisitionTask) -> list[ProviderName]:
        allowed = set(task.provider_policy.allowed_providers) & set(self.settings.provider_allowlist)
        selected: list[ProviderName] = []

        if "search_api" in allowed:
            selected.append("search_api")
        if "openai_web_search" in allowed:
            selected.append("openai_web_search")
        if "mock" in allowed:
            selected.append("mock")

        if any("staff report" in need.lower() or "ceqa" in need.lower() for need in task.data_needed):
            if "claude" in allowed:
                selected.append("claude")
        if task.source_policy.require_raw_source_fetch and "crawler" in allowed:
            selected.append("crawler")

        if not selected and "mock" in self.settings.provider_allowlist:
            selected.append("mock")
        return _dedupe(selected)


def _dedupe(values: list[ProviderName]) -> list[ProviderName]:
    seen: set[str] = set()
    result: list[ProviderName] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
