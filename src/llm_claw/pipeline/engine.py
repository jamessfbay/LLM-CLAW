from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, EvidenceItem, EvidencePack, ProviderTrace, RawSource
from llm_claw.pipeline.extractors import ContentExtractor, EvidenceExtractor
from llm_claw.pipeline.normalizer import ConfidenceScorer, DataNormalizer
from llm_claw.pipeline.source_filter import SourceRelevanceFilter
from llm_claw.pipeline.source_fetcher import SourceFetcher
from llm_claw.planning import DataNeedPlanner, QueryPlanner
from llm_claw.providers.claude import ClaudeVerificationAgent
from llm_claw.providers.factory import build_provider
from llm_claw.providers.router import ProviderRouter


class DataAcquisitionEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_need_planner = DataNeedPlanner()
        self.query_planner = QueryPlanner()
        self.router = ProviderRouter(settings)
        self.fetcher = SourceFetcher(settings)
        self.source_filter = SourceRelevanceFilter()
        self.content_extractor = ContentExtractor()
        self.evidence_extractor = EvidenceExtractor()
        self.normalizer = DataNormalizer()
        self.scorer = ConfidenceScorer()
        self.verifier = ClaudeVerificationAgent()

    def run(self, task: AcquisitionTask) -> EvidencePack:
        data_needs = self.data_need_planner.plan(task)
        queries = self.query_planner.plan(task, data_needs)
        selected = self.router.select_providers(task)

        candidates: list[CandidateSource] = task.seed_sources[:]
        traces: list[ProviderTrace] = []
        discover_jobs = [
            (provider_name, query)
            for provider_name in selected
            if provider_name not in {"crawler", "claude"}
            for query in queries
        ]
        if discover_jobs:
            with ThreadPoolExecutor(max_workers=max(1, self.settings.provider_max_workers)) as executor:
                futures = {
                    executor.submit(_timed_discover, build_provider(provider_name, self.settings), task, query): (
                        provider_name,
                        query,
                    )
                    for provider_name, query in discover_jobs
                }
                for future in as_completed(futures):
                    found, trace = future.result()
                    candidates.extend(found)
                    traces.append(trace)

        candidates = _dedupe_candidates(candidates)
        candidates = self.source_filter.filter_candidates(task, candidates)
        raw_sources: list[RawSource] = []
        if task.source_policy.require_raw_source_fetch or self.settings.require_raw_source_fetch:
            fetch_start = time.perf_counter()
            raw_sources = self.fetcher.fetch(candidates)
            traces.append(
                ProviderTrace(
                    provider="crawler",
                    candidate_count=len(raw_sources),
                    message=f"Fetched {len(raw_sources)} raw sources.",
                    duration_ms=_duration_ms(fetch_start),
                )
            )

        extracted_sources = self.content_extractor.extract_text(task, raw_sources)
        claims = self.evidence_extractor.extract_claims(task, extracted_sources)
        verify_start = time.perf_counter()
        verification_notes, verification_trace = self.verifier.verify(claims, extracted_sources)
        verification_trace.duration_ms = _duration_ms(verify_start)
        if "claude" in selected:
            traces.append(verification_trace)

        source_by_id = {source.id: source for source in extracted_sources}
        verified_claim_ids = {note.claim_id for note in verification_notes if note.support_level == "full"}
        evidence: list[EvidenceItem] = []
        for claim in claims:
            source = source_by_id.get(claim.source_id)
            if not source:
                continue
            verified = claim.id in verified_claim_ids
            evidence.append(
                EvidenceItem(
                    claim=claim.text,
                    source_title=source.source_title,
                    source_url=source.source_url,
                    publisher=source.publisher,
                    source_type=source.source_type,
                    evidence_text=claim.evidence_text,
                    retrieved_at=source.retrieved_at,
                    confidence=self.scorer.score(claim.confidence, source.source_type, verified),
                    verified_by=["crawler", "claude"] if verified else ["crawler"],
                    source_id=source.id,
                    claim_id=claim.id,
                )
            )

        structured = self.normalizer.normalize(task, claims)
        missing = _missing_data(task, evidence)
        return EvidencePack(
            request_id=task.id,
            entity=task.entity.model_dump(mode="json"),
            summary=_summary(task, evidence, missing),
            structured_data=structured,
            evidence=evidence,
            missing_data=missing,
            recommended_next_actions=_next_actions(missing),
            provider_trace=traces,
            candidate_sources=candidates,
            raw_sources=raw_sources,
            verification_notes=verification_notes,
        )


def _timed_discover(provider, task: AcquisitionTask, query) -> tuple[list[CandidateSource], ProviderTrace]:
    start = time.perf_counter()
    found, trace = provider.discover(task, query)
    trace.duration_ms = _duration_ms(start)
    return found, trace


def _duration_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _dedupe_candidates(candidates: list[CandidateSource]) -> list[CandidateSource]:
    seen: set[str] = set()
    result: list[CandidateSource] = []
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _candidate_key(candidate: CandidateSource) -> str:
    match = re.search(r"ceqanet\.(?:lci|opr)\.ca\.gov/(?:project/)?(\d{10})(?:/\d+)?", candidate.url.lower())
    if match:
        return f"ceqanet:{match.group(1)}"
    return candidate.url.rstrip("/").lower()


def _missing_data(task: AcquisitionTask, evidence: list[EvidenceItem]) -> list[str]:
    missing: list[str] = []
    evidence_text = " ".join(item.claim.lower() for item in evidence)
    for need in task.data_needed:
        terms = [term for term in need.lower().split() if len(term) >= 4]
        if terms and not any(term in evidence_text for term in terms):
            missing.append(f"No verified raw-source evidence found for {need}.")
    if not evidence:
        missing.append("No raw-source evidence could be fetched.")
    return missing


def _next_actions(missing: list[str]) -> list[str]:
    if not missing:
        return ["Review Evidence Pack before upserting to LLM-KG."]
    return [
        "Search city council and planning commission agendas.",
        "Check permit and parcel databases.",
        "Review CEQA notice repositories and staff report PDFs.",
    ]


def _summary(task: AcquisitionTask, evidence: list[EvidenceItem], missing: list[str]) -> str:
    entity = task.entity.display_name
    if evidence:
        return f"{entity} has {len(evidence)} source-linked evidence item(s); {len(missing)} data gap(s) remain."
    return f"{entity} has no verified raw-source evidence yet."
