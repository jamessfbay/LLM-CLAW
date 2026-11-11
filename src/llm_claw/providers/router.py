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
        if "aisa_reddit" in allowed and self.settings.aisa_reddit_max_queries > 0:
            selected.append("aisa_reddit")
        if (
            task.task_type == "candidate_buyer_discovery"
            and "aisa_twitter" in allowed
            and self.settings.aisa_twitter_max_queries > 0
        ):
            selected.append("aisa_twitter")
        if (
            task.task_type == "candidate_buyer_discovery"
            and "aisa_apollo_people" in allowed
            and self.settings.aisa_apollo_people_max_queries > 0
        ):
            selected.append("aisa_apollo_people")
        if "aisa_tavily" in allowed and self.settings.aisa_tavily_max_queries > 0:
            selected.append("aisa_tavily")
        if "aisa_scholar" in allowed and self.settings.aisa_scholar_max_queries > 0 and task.task_type == "commercial_deep_research":
            selected.append("aisa_scholar")
        if "github" in allowed and self.settings.github_max_queries > 0:
            selected.append("github")
        if "hacker_news" in allowed and self.settings.hacker_news_max_queries > 0:
            selected.append("hacker_news")
        if "stack_exchange" in allowed and self.settings.stack_exchange_max_queries > 0:
            selected.append("stack_exchange")
        if "gemini" in allowed:
            selected.append("gemini")
        if "mock" in allowed:
            selected.append("mock")

        if "claude" in allowed:
            selected.append("claude")
        if task.source_policy.require_raw_source_fetch and "crawler" in allowed:
            selected.append("crawler")

        return _dedupe(selected)


def _dedupe(values: list[ProviderName]) -> list[ProviderName]:
    seen: set[str] = set()
    result: list[ProviderName] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
