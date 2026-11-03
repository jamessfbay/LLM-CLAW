from __future__ import annotations

from urllib.parse import urlparse

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace
from llm_claw.providers.aisa_client import AIsaClient


class AIsaScholarProvider:
    name = "aisa_scholar"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AIsaClient(settings)

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        if not self.settings.aisa_api_key:
            return [], ProviderTrace(provider=self.name, status="disabled", query=query.text, message="AISA_API_KEY is not configured")
        try:
            response = self.client.post(
                "/apis/v1/scholar/search/scholar", params={"query": query.text, "max_num_results": min(task.source_policy.max_sources, 5)}
            )
            rows = response.get("results")
            if not isinstance(rows, list):
                raise ValueError("Scholar response has no results list")
        except Exception as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"Scholar search failed: {type(exc).__name__}")
        candidates: list[CandidateSource] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get("link") or "").strip()
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                continue
            candidates.append(CandidateSource(
                query_id=query.id, provider=self.name, title=str(row.get("title") or url)[:300], url=url,
                snippet=str(row.get("snippet") or "")[:500], publisher=str(row.get("journal") or parsed.hostname),
                confidence=0.7, is_official=False,
                analysis_note="Scholar discovery only; paper URL must be fetched and checked before evidence use.",
            ))
        return candidates, ProviderTrace(provider=self.name, query=query.text, candidate_count=len(candidates), message=f"Scholar returned {len(candidates)} candidate(s)")
