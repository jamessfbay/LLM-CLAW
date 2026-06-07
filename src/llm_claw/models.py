from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


ProviderName = Literal[
    "mock",
    "gemini",
    "claude",
    "perplexity",
    "openai_web_search",
    "search_api",
    "crawler",
    "government_api",
]


class EntityInput(BaseModel):
    project_name: str | None = None
    type: str | None = None
    name: str | None = None
    city: str | None = None
    address: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.project_name or self.name or self.address or "unknown entity"


class SourcePolicy(BaseModel):
    prefer_official_sources: bool = True
    require_citations: bool = True
    require_raw_source_fetch: bool = True


class ProviderPolicy(BaseModel):
    allowed_providers: list[ProviderName] = Field(
        default_factory=lambda: ["mock", "search_api", "crawler", "claude"]
    )
    final_source_must_be_raw_document: bool = True


class AcquisitionTask(BaseModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    domain: str = "real_estate"
    task_type: str = "project_research"
    entity: EntityInput
    data_needed: list[str] = Field(default_factory=list)
    freshness: Literal["latest", "recent", "any"] = "latest"
    source_policy: SourcePolicy = Field(default_factory=SourcePolicy)
    provider_policy: ProviderPolicy = Field(default_factory=ProviderPolicy)
    seed_sources: list[CandidateSource] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("data_needed")
    @classmethod
    def require_data_needed(cls, value: list[str]) -> list[str]:
        return value or ["planning status", "staff report", "public comments", "CEQA status"]


class PlannedQuery(BaseModel):
    id: str = Field(default_factory=lambda: new_id("query"))
    text: str
    data_need: str
    freshness: str = "latest"


class CandidateSource(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cand"))
    query_id: str | None = None
    provider: ProviderName
    title: str
    url: str
    snippet: str = ""
    publisher: str | None = None
    source_type: Literal["candidate", "official_html", "official_pdf", "government_api", "webpage", "pdf"] = "candidate"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_official: bool = False
    analysis_note: str | None = None


class ProviderTrace(BaseModel):
    provider: ProviderName
    status: Literal["ok", "disabled", "error"] = "ok"
    query: str | None = None
    message: str | None = None
    candidate_count: int = 0
    duration_ms: int | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RawSource(BaseModel):
    id: str = Field(default_factory=lambda: new_id("src"))
    candidate_id: str | None = None
    source_url: str
    source_title: str
    publisher: str | None = None
    source_type: Literal["official_html", "official_pdf", "government_api", "webpage", "pdf", "local_html", "local_pdf"]
    retrieved_at: datetime = Field(default_factory=utc_now)
    content_hash: str
    raw_path: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedClaim(BaseModel):
    id: str = Field(default_factory=lambda: new_id("claim"))
    text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    source_id: str
    evidence_text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: Literal["active", "uncertain", "unsupported"] = "active"


class VerificationNote(BaseModel):
    provider: ProviderName = "claude"
    claim_id: str
    support_level: Literal["full", "partial", "unsupported", "unclear"]
    rationale: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EvidenceItem(BaseModel):
    claim: str
    source_title: str
    source_url: str
    publisher: str | None = None
    source_type: str
    evidence_text: str
    retrieved_at: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    verified_by: list[ProviderName] = Field(default_factory=list)
    page_number: int | None = None
    source_id: str | None = None
    claim_id: str | None = None


class EvidencePack(BaseModel):
    request_id: str
    entity: dict[str, Any]
    summary: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    provider_trace: list[ProviderTrace] = Field(default_factory=list)
    candidate_sources: list[CandidateSource] = Field(default_factory=list)
    raw_sources: list[RawSource] = Field(default_factory=list)
    verification_notes: list[VerificationNote] = Field(default_factory=list)


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["created", "running", "completed", "failed"]
    evidence_pack_path: str | None = None
    message: str | None = None


JsonLike = str | Path | dict[str, Any] | AcquisitionTask
