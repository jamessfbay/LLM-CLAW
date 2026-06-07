from __future__ import annotations

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, EvidenceItem, EvidencePack, ProviderTrace, RawSource
from llm_claw.pipeline.extractors import ContentExtractor, EvidenceExtractor
from llm_claw.pipeline.normalizer import ConfidenceScorer, DataNormalizer
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
        for provider_name in selected:
            if provider_name in {"crawler", "claude"}:
                continue
            provider = build_provider(provider_name, self.settings)
            for query in queries:
                found, trace = provider.discover(task, query)
                candidates.extend(found)
                traces.append(trace)

        candidates = _dedupe_candidates(candidates)
        raw_sources: list[RawSource] = []
        if task.source_policy.require_raw_source_fetch or self.settings.require_raw_source_fetch:
            raw_sources = self.fetcher.fetch(candidates)
            traces.append(ProviderTrace(provider="crawler", candidate_count=len(raw_sources), message=f"Fetched {len(raw_sources)} raw sources."))

        extracted_sources = self.content_extractor.extract_text(raw_sources)
        claims = self.evidence_extractor.extract_claims(task, extracted_sources)
        verification_notes, verification_trace = self.verifier.verify(claims, extracted_sources)
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


def _dedupe_candidates(candidates: list[CandidateSource]) -> list[CandidateSource]:
    seen: set[str] = set()
    result: list[CandidateSource] = []
    for candidate in candidates:
        key = candidate.url
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


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
