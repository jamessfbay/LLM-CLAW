from pathlib import Path

from llm_claw.cache import build_cache_key
from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask
from llm_claw.providers.router import ProviderRouter
from llm_claw.planning import QueryPlanner


def _task() -> AcquisitionTask:
    return AcquisitionTask.model_validate(
        {
            "entity": {"project_name": "Example Housing Project", "city": "San Jose"},
            "data_needed": ["planning status", "staff report", "CEQA status"],
            "provider_policy": {"allowed_providers": ["mock", "search_api", "crawler", "claude"]},
        }
    )


def test_provider_router_selects_search_mock_claude_and_crawler(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, provider_allowlist=["mock", "search_api", "crawler", "claude"])
    providers = ProviderRouter(settings).select_providers(_task())

    assert providers == ["search_api", "mock", "claude", "crawler"]


def test_provider_router_selects_gemini_when_allowed(tmp_path: Path) -> None:
    task = AcquisitionTask.model_validate(
        {
            "entity": {"project_name": "Example Housing Project", "city": "San Jose"},
            "data_needed": ["planning status"],
            "provider_policy": {"allowed_providers": ["gemini", "crawler"]},
        }
    )
    settings = Settings(workspace=tmp_path, provider_allowlist=["gemini", "crawler"])

    assert ProviderRouter(settings).select_providers(task) == ["gemini", "crawler"]


def test_cache_key_contains_stable_task_provider_dimensions() -> None:
    task = _task()

    first = build_cache_key(task, "Example query", "mock", "latest")
    second = build_cache_key(task, "Example query", "mock", "latest")
    different_provider = build_cache_key(task, "Example query", "search_api", "latest")

    assert first == second
    assert first.startswith("generic:mock:")
    assert first != different_provider


def test_commercial_discovery_preserves_targeted_queries_without_real_estate_terms() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "commercial_discovery",
            "task_type": "market_signal_research",
            "entity": {"name": "discovery run"},
            "data_needed": ["site:reddit.com AI cooling pain", "site:github.com AI cooling issue"],
            "provider_policy": {"allowed_providers": ["openai_web_search", "gemini", "crawler"]},
        }
    )

    queries = QueryPlanner().plan(task, task.data_needed)

    assert [query.text for query in queries] == task.data_needed
    assert all("CEQA" not in query.text and "planning commission" not in query.text for query in queries)
