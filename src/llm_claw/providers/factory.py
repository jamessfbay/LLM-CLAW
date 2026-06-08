from __future__ import annotations

from llm_claw.config import Settings
from llm_claw.models import ProviderName
from llm_claw.providers.llm_disabled import DisabledLLMProvider
from llm_claw.providers.gemini import GeminiProvider
from llm_claw.providers.mock import MockProvider
from llm_claw.providers.openai_web_search import OpenAIWebSearchProvider
from llm_claw.providers.search_api import SearchAPIProvider


def build_provider(name: ProviderName, settings: Settings):
    if name == "mock":
        return MockProvider()
    if name == "search_api":
        return SearchAPIProvider(settings)
    if name == "gemini":
        return GeminiProvider(settings)
    if name == "claude":
        return DisabledLLMProvider(name, settings, "ANTHROPIC_API_KEY")
    if name == "perplexity":
        return DisabledLLMProvider(name, settings, "PERPLEXITY_API_KEY")
    if name == "openai_web_search":
        return OpenAIWebSearchProvider(settings)
    if name == "government_api":
        return DisabledLLMProvider(name, settings, "government API configuration")
    if name == "crawler":
        return DisabledLLMProvider(name, settings, "crawler is invoked after source discovery")
    raise ValueError(f"Unknown provider: {name}")


def list_provider_statuses(settings: Settings) -> list[dict[str, str | bool]]:
    providers: list[ProviderName] = [
        "mock",
        "search_api",
        "crawler",
        "claude",
        "gemini",
        "perplexity",
        "openai_web_search",
        "government_api",
    ]
    statuses = []
    for provider in providers:
        enabled = provider in settings.provider_allowlist
        configured = {
            "mock": True,
            "search_api": bool(settings.search_api_key),
            "crawler": True,
            "claude": bool(settings.anthropic_api_key),
            "gemini": bool(settings.gemini_api_key),
            "perplexity": bool(settings.perplexity_api_key),
            "openai_web_search": bool(settings.openai_api_key),
            "government_api": False,
        }[provider]
        statuses.append({"provider": provider, "enabled": enabled, "configured": configured})
    return statuses
