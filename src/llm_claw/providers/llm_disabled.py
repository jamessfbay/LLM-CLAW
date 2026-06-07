from __future__ import annotations

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderName, ProviderTrace


class DisabledLLMProvider:
    def __init__(self, name: ProviderName, settings: Settings, env_name: str) -> None:
        self.name = name
        self.settings = settings
        self.env_name = env_name

    def available(self) -> bool:
        return False

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        return [], ProviderTrace(
            provider=self.name,
            status="disabled",
            query=query.text,
            message=f"{self.env_name} is not configured or provider is not implemented in this MVP.",
        )
