from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from llm_claw.config import Settings
from llm_claw.models import (
    AcquisitionTask,
    CandidateSource,
    EvidenceItem,
    EvidencePack,
    PlannedQuery,
    ProviderName,
    ProviderTrace,
    RawSource,
    SourceFetchDiagnostic,
)
from llm_claw.pipeline.extractors import ContentExtractor, EvidenceExtractor
from llm_claw.pipeline.normalizer import ConfidenceScorer, DataNormalizer
from llm_claw.pipeline.source_filter import SourceRelevanceFilter
from llm_claw.pipeline.source_fetcher import SourceFetcher
from llm_claw.planning import DataNeedPlanner, QueryPlanner
from llm_claw.providers.claude import ClaudeVerificationAgent
from llm_claw.providers.factory import build_provider
from llm_claw.providers.gemini import GeminiProvider
from llm_claw.providers.aisa_enrichment import AIsaReranker, AIsaSimilarweb
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
        self.youtube_analyzer = GeminiProvider(settings)
        self.aisa_reranker = AIsaReranker(settings)
        self.aisa_similarweb = AIsaSimilarweb(settings)

    def run(self, task: AcquisitionTask) -> EvidencePack:
        data_needs = self.data_need_planner.plan(task)
        queries = self.query_planner.plan(task, data_needs)
        selected = self.router.select_providers(task)
        scholar_query = PlannedQuery(
            text=f"{task.entity.metadata.get('topic') or task.question or task.entity.display_name} empirical evidence limitations",
            data_need="academic evidence",
            freshness=task.freshness,
        )

        candidates: list[CandidateSource] = task.seed_sources[:]
        traces: list[ProviderTrace] = []
        discover_jobs = []
        for provider_name in selected:
            if provider_name in {"crawler", "claude"}:
                continue
            provider_queries = [scholar_query] if provider_name == "aisa_scholar" else _provider_queries(
                provider_name, queries, task, self.settings,
            )
            for query in provider_queries:
                discover_jobs.append((provider_name, query))
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
                    provider_name, query = futures[future]
                    try:
                        found, trace = future.result()
                        candidates.extend(found)
                        traces.append(trace)
                    except Exception as exc:
                        traces.append(
                            ProviderTrace(
                                provider=provider_name,
                                status="error",
                                query=query.text,
                                message=f"Provider discovery failed: {type(exc).__name__}",
                            )
                        )

        candidates = _dedupe_candidates(candidates)
        candidates = self.source_filter.filter_candidates(task, candidates)
        if (
            len(candidates) > task.source_policy.max_sources
            and self.settings.aisa_api_key
            and self.settings.aisa_rerank_max_candidates > 0
            and "aisa_jina_rerank" in task.provider_policy.allowed_providers
            and self.settings.provider_enabled("aisa_jina_rerank")
        ):
            ranked, trace = self.aisa_reranker.rank(
                str(task.entity.metadata.get("topic") or task.question or task.entity.display_name),
                candidates[: self.settings.aisa_rerank_max_candidates],
            )
            candidates = ranked + candidates[self.settings.aisa_rerank_max_candidates :]
            traces.append(trace)
        candidates = candidates[: task.source_policy.max_sources]
        youtube_candidates = [candidate for candidate in candidates if _is_youtube_candidate(candidate)]
        raw_sources: list[RawSource] = []
        fetch_diagnostics: list[SourceFetchDiagnostic] = []
        if task.source_policy.require_raw_source_fetch or self.settings.require_raw_source_fetch:
            fetch_candidates = [candidate for candidate in candidates if not _is_youtube_candidate(candidate)]
            fetch_start = time.perf_counter()
            if hasattr(self.fetcher, "fetch_with_diagnostics"):
                raw_sources, fetch_diagnostics = self.fetcher.fetch_with_diagnostics(
                    fetch_candidates, allowed_providers=set(task.provider_policy.allowed_providers)
                )
            else:
                raw_sources = self.fetcher.fetch(fetch_candidates)
            traces.append(
                ProviderTrace(
                    provider="crawler",
                    candidate_count=len(raw_sources),
                    message=f"Fetched {len(raw_sources)} raw sources; {len(fetch_diagnostics)} fetch attempt(s) recorded.",
                    duration_ms=_duration_ms(fetch_start),
                )
            )
        youtube_sources, youtube_candidate_traces = self._fetch_youtube_candidates_with_gemini(task, youtube_candidates)
        raw_sources.extend(youtube_sources)
        traces.extend(youtube_candidate_traces)

        raw_sources, youtube_traces = self._analyze_youtube_sources(task, raw_sources)
        traces.extend(youtube_traces)

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
            quote_start = source.text.find(claim.evidence_text)
            quote_end = quote_start + len(claim.evidence_text) if quote_start >= 0 else None
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
                    verified_by=_source_providers(source, verified),
                    source_id=source.id,
                    claim_id=claim.id,
                    raw_content_hash=source.content_hash,
                    quote_start=quote_start if quote_start >= 0 else None,
                    quote_end=quote_end,
                    quote_match="exact" if quote_start >= 0 else "unbound",
                    observed_at=source.retrieved_at,
                )
            )

        structured = self.normalizer.normalize(task, claims)
        if (
            task.task_type == "commercial_deep_research"
            and self.settings.aisa_api_key
            and self.settings.aisa_similarweb_max_domains > 0
            and "aisa_similarweb" in task.provider_policy.allowed_providers
            and self.settings.provider_enabled("aisa_similarweb")
        ):
            domains = task.entity.metadata.get("competitor_domains")
            if isinstance(domains, list):
                snapshots, market_traces = self.aisa_similarweb.traffic(
                    [domain for domain in domains if isinstance(domain, str)],
                    self.settings.aisa_similarweb_max_domains,
                )
                if snapshots:
                    structured["market_metrics"] = snapshots
                traces.extend(market_traces)
        missing = _missing_data(task, evidence, raw_sources, fetch_diagnostics)
        return EvidencePack(
            request_id=task.id,
            entity=task.entity.model_dump(mode="json"),
            summary=_summary(task, evidence, missing),
            structured_data=structured,
            evidence=evidence,
            missing_data=missing,
            recommended_next_actions=_next_actions(task, missing),
            provider_trace=traces,
            candidate_sources=candidates,
            raw_sources=raw_sources,
            source_fetch_diagnostics=fetch_diagnostics,
            verification_notes=verification_notes,
        )

    def _fetch_youtube_candidates_with_gemini(
        self, task: AcquisitionTask, candidates: list[CandidateSource]
    ) -> tuple[list[RawSource], list[ProviderTrace]]:
        sources: list[RawSource] = []
        traces: list[ProviderTrace] = []
        for candidate in candidates:
            start = time.perf_counter()
            analyzed, trace = self.youtube_analyzer.analyze_youtube_candidate(task, candidate)
            trace.duration_ms = _duration_ms(start)
            traces.append(trace)
            if analyzed and _has_useful_youtube_analysis(analyzed.text):
                sources.append(analyzed)
        return sources, traces

    def _analyze_youtube_sources(
        self, task: AcquisitionTask, sources: list[RawSource]
    ) -> tuple[list[RawSource], list[ProviderTrace]]:
        analyzed_sources: list[RawSource] = []
        traces: list[ProviderTrace] = []
        for source in sources:
            if source.source_type != "youtube":
                analyzed_sources.append(source)
                continue
            if source.metadata.get("analysis_provider") == "gemini":
                analyzed_sources.append(source)
                continue
            start = time.perf_counter()
            analyzed, trace = self.youtube_analyzer.analyze_youtube_source(task, source)
            trace.duration_ms = _duration_ms(start)
            traces.append(trace)
            if analyzed and _has_useful_youtube_analysis(analyzed.text):
                analyzed_sources.append(analyzed)
        return analyzed_sources, traces


def _timed_discover(provider, task: AcquisitionTask, query) -> tuple[list[CandidateSource], ProviderTrace]:
    start = time.perf_counter()
    found, trace = provider.discover(task, query)
    trace.duration_ms = _duration_ms(start)
    return found, trace


def _provider_queries(
    provider: ProviderName,
    queries: list[PlannedQuery],
    task: AcquisitionTask,
    settings: Settings,
) -> list[PlannedQuery]:
    text = lambda query: query.text.lower()
    if provider == "aisa_reddit":
        return [query for query in queries if "site:reddit.com" in text(query)][: settings.aisa_reddit_max_queries]
    if provider == "aisa_twitter":
        return [query for query in queries if "site:x.com" in text(query)][: settings.aisa_twitter_max_queries]
    if provider == "aisa_apollo_people":
        return [
            query for query in queries if "site:linkedin.com/in" in text(query)
        ][: settings.aisa_apollo_people_max_queries]
    if provider == "github":
        return [query for query in queries if "site:github.com" in text(query)][: settings.github_max_queries]
    if provider == "hacker_news":
        return [query for query in queries if "site:news.ycombinator.com" in text(query)][: settings.hacker_news_max_queries]
    if provider == "stack_exchange":
        return [query for query in queries if "site:stackoverflow.com" in text(query) or "site:stackexchange.com" in text(query)][: settings.stack_exchange_max_queries]
    if provider == "aisa_tavily":
        generic = [query for query in queries if not text(query).startswith("site:")]
        fallback = queries if not generic else generic
        return fallback[: settings.aisa_tavily_max_queries]
    if task.task_type == "candidate_buyer_discovery" and provider in {"openai_web_search", "gemini", "search_api"}:
        return [query for query in queries if not text(query).startswith("site:")][:2]
    return queries


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


def _is_youtube_candidate(candidate: CandidateSource) -> bool:
    host = urlparse(candidate.url).netloc.lower()
    return candidate.source_type == "youtube" or host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _has_useful_youtube_analysis(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if "youtube.com/watch?v=" in normalized:
        return True
    if "no project-relevant youtube content found" in normalized:
        return False
    if "no requested youtube content found" in normalized:
        return False
    if normalized in {"no concrete youtube video url found.", "no concrete youtube video url found"}:
        return False
    return True


def _source_providers(source: RawSource, verified: bool) -> list[ProviderName]:
    providers: list[ProviderName] = []
    source_provider = source.metadata.get("source_provider")
    if source_provider in {"aisa_twitter", "aisa_apollo_people"}:
        providers.append(source_provider)
    if source.raw_path:
        providers.append("crawler")
    if source.metadata.get("source_fetcher") == "gemini" or source.metadata.get("analysis_provider") == "gemini":
        providers.append("gemini")
    if not providers:
        providers.append("crawler")
    if verified:
        providers.append("claude")
    return list(dict.fromkeys(providers))


def _missing_data(
    task: AcquisitionTask,
    evidence: list[EvidenceItem],
    raw_sources: list[RawSource] | None = None,
    diagnostics: list[SourceFetchDiagnostic] | None = None,
) -> list[str]:
    missing: list[str] = []
    evidence_text = " ".join(item.claim.lower() for item in evidence)
    for need in task.data_needed:
        if not _need_is_covered(need, evidence_text):
            missing.append(f"No verified raw-source evidence found for {need}.")
    if not evidence:
        if raw_sources or any(item.raw_path for item in diagnostics or []):
            missing.append("Raw source was fetched but no usable source-linked evidence was extracted.")
        else:
            missing.append("No raw source could be fetched; inspect source fetch diagnostics.")
    return missing


def _need_is_covered(need: str, evidence_text: str) -> bool:
    lower_need = need.lower()
    if "cve" in lower_need and re.search(r"\bCVE-\d{4}-\d{4,}\b", evidence_text, re.I):
        return True
    if "date" in lower_need and (
        re.search(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b",
            evidence_text,
            re.I,
        )
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", evidence_text)
    ):
        return True
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_text.lower()))
    terms = [term for term in re.findall(r"[a-z0-9]+", lower_need) if len(term) >= 3]
    return bool(terms) and any(_term_is_covered(term, evidence_tokens) for term in terms)


def _term_is_covered(term: str, tokens: set[str]) -> bool:
    if term in tokens:
        return True
    stem = term[:7] if len(term) >= 7 else term
    return len(stem) >= 5 and any(token.startswith(stem) for token in tokens)


def _next_actions(task: AcquisitionTask, missing: list[str]) -> list[str]:
    if not missing:
        return ["Review Evidence Pack before upserting to LLM-KG."]
    return [
        f"Acquire an additional authoritative source for {need}."
        for need in task.data_needed
    ]


def _summary(task: AcquisitionTask, evidence: list[EvidenceItem], missing: list[str]) -> str:
    entity = task.entity.display_name
    if evidence:
        return f"{entity} has {len(evidence)} source-linked evidence item(s); {len(missing)} data gap(s) remain."
    return f"{entity} has no verified raw-source evidence yet."
