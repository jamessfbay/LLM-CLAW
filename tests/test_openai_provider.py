from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask
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
                "provider_policy": {"allowed_providers": ["openai_web_search", "crawler"]},
            }
        )
    )
