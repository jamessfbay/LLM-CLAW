from __future__ import annotations

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace


class SearchAPIProvider:
    name = "search_api"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return True

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        if not self.settings.search_api_key:
            return [], ProviderTrace(
                provider=self.name,
                status="disabled",
                query=query.text,
                message="SEARCH_API_KEY is not configured; provider skipped.",
            )
        return [], ProviderTrace(
            provider=self.name,
            query=query.text,
            message="Real Search API integration is not implemented in this MVP.",
        )
