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
    project_id: str | None = None
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
    max_sources: int = Field(default=50, ge=1, le=200)


class ProviderPolicy(BaseModel):
    allowed_providers: list[ProviderName] = Field(
        default_factory=lambda: ["mock", "search_api", "crawler", "claude"]
    )
    final_source_must_be_raw_document: bool = True


class AcquisitionTask(BaseModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    domain: str = "generic"
    task_type: str = "evidence_acquisition"
    entity: EntityInput
    data_needed: list[str]
    question: str | None = None
    acquisition_instruction: str | None = None
    freshness: Literal["latest", "recent", "any"] = "latest"
    source_policy: SourcePolicy = Field(default_factory=SourcePolicy)
    provider_policy: ProviderPolicy = Field(default_factory=ProviderPolicy)
    seed_sources: list[CandidateSource] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("data_needed")
    @classmethod
    def require_data_needed(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("data_needed must explicitly describe the observations to acquire")
        return value


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
    source_type: Literal["candidate", "official_html", "official_pdf", "government_api", "webpage", "pdf", "youtube"] = "candidate"
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
    model: str | None = None
    prompt_version: str | None = None
    usage: dict[str, int] | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RawSource(BaseModel):
    id: str = Field(default_factory=lambda: new_id("src"))
    candidate_id: str | None = None
    source_url: str
    source_title: str
    publisher: str | None = None
    source_type: Literal["official_html", "official_pdf", "government_api", "webpage", "pdf", "local_html", "local_pdf", "youtube"]
    retrieved_at: datetime = Field(default_factory=utc_now)
    content_hash: str
    raw_path: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceFetchDiagnostic(BaseModel):
    id: str = Field(default_factory=lambda: new_id("fetch"))
    candidate_id: str | None = None
    url: str
    title: str | None = None
    status: Literal["fetched", "failed", "dropped"] = "failed"
    fetch_mode: str = "http"
    http_status: int | None = None
    final_url: str | None = None
    content_type: str | None = None
    bytes: int = 0
    text_length: int = 0
    raw_path: str | None = None
    drop_reason: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


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
    contract_version: Literal["evidence-artifact/2.0"] = "evidence-artifact/2.0"
    id: str = Field(default_factory=lambda: new_id("ev"))
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
    raw_content_hash: str | None = None
    quote_start: int | None = Field(default=None, ge=0)
    quote_end: int | None = Field(default=None, ge=0)
    quote_match: Literal["exact", "unbound"] = "unbound"
    observed_at: datetime | None = None
    extractor_version: str = "llm-claw-evidence-extractor/1"

    @property
    def decision_grade(self) -> bool:
        return bool(
            self.source_id
            and self.raw_content_hash
            and self.quote_match == "exact"
            and self.quote_start is not None
            and self.quote_end is not None
            and self.quote_end > self.quote_start
        )


class EvidencePack(BaseModel):
    contract_version: Literal["observer-output/2.0"] = "observer-output/2.0"
    capability_role: Literal["observer"] = "observer"
    request_id: str
    correlation_id: str | None = None
    decision_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    input_hash: str | None = None
    entity: dict[str, Any]
    summary: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    provider_trace: list[ProviderTrace] = Field(default_factory=list)
    candidate_sources: list[CandidateSource] = Field(default_factory=list)
    raw_sources: list[RawSource] = Field(default_factory=list)
    source_fetch_diagnostics: list[SourceFetchDiagnostic] = Field(default_factory=list)
    verification_notes: list[VerificationNote] = Field(default_factory=list)


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["created", "running", "completed", "failed"]
    evidence_pack_path: str | None = None
    message: str | None = None


JsonLike = str | Path | dict[str, Any] | AcquisitionTask
