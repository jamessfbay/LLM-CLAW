from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, PlannedQuery
from llm_claw.providers.openai_web_search import OpenAIWebSearchProvider
from llm_claw.providers.router import ProviderRouter


def test_openai_provider_disables_without_key(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, openai_api_key=None)
    provider = OpenAIWebSearchProvider(settings)

    assert provider.available() is False
    assert "openai_web_search" in ProviderRouter(settings).select_providers(
        AcquisitionTask.model_validate(
            {
                "entity": {"project_name": "Example"},
                "data_needed": ["current status"],
                "provider_policy": {"allowed_providers": ["openai_web_search", "crawler"]},
            }
        )
    )


def test_openai_provider_records_bounded_model_usage(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, openai_api_key="test-key", openai_model="gpt-test")
    provider = OpenAIWebSearchProvider(settings)
    provider._responses_request = lambda prompt: {  # type: ignore[method-assign]
        "output_text": "[]",
        "usage": {"input_tokens": 90, "output_tokens": 10, "total_tokens": 100},
    }
    task = AcquisitionTask.model_validate(
        {"entity": {"name": "Example"}, "data_needed": ["current status"]}
    )

    _, trace = provider.discover(
        task,
        PlannedQuery(text="Example status", data_need="current status"),
    )

    assert trace.model == "gpt-test"
    assert trace.prompt_version == "source-discovery/2"
    assert trace.usage == {"input_tokens": 90, "output_tokens": 10, "total_tokens": 100}
