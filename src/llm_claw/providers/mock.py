from __future__ import annotations

from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace


class MockProvider:
    name = "mock"

    def available(self) -> bool:
        return True

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        entity = task.entity.display_name
        city = task.entity.city or "City"
        slug = "-".join(part for part in [city, entity, query.data_need] if part).lower().replace(" ", "-")
        candidates = [
            CandidateSource(
                query_id=query.id,
                provider=self.name,
                title=f"{city} {entity} {query.data_need} official record",
                url=f"mock://official/{slug}.html",
                snippet=f"Mock official source for {entity}: {query.data_need}.",
                publisher=f"City of {city}" if city != "City" else "Mock City",
                source_type="candidate",
                confidence=0.72,
                is_official=True,
            )
        ]
        return candidates, ProviderTrace(provider=self.name, query=query.text, candidate_count=len(candidates))
