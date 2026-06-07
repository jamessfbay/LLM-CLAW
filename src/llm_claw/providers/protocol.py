from __future__ import annotations

from typing import Protocol

from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderName, ProviderTrace


class ProviderAdapter(Protocol):
    name: ProviderName

    def available(self) -> bool:
        ...

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        ...
