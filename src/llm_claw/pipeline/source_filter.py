from __future__ import annotations

import re
from urllib.parse import urlparse

from llm_claw.models import AcquisitionTask, CandidateSource, RawSource


class SourceRelevanceFilter:
    def filter_candidates(self, task: AcquisitionTask, candidates: list[CandidateSource]) -> list[CandidateSource]:
        return [candidate for candidate in candidates if self.is_relevant_candidate(task, candidate)]

    def filter_sources(self, task: AcquisitionTask, sources: list[RawSource]) -> list[RawSource]:
        return [source for source in sources if self.is_relevant_source(task, source)]

    def is_relevant_candidate(self, task: AcquisitionTask, candidate: CandidateSource) -> bool:
        if candidate.url.startswith(("mock://", "file://")):
            return True
        if _is_blocked_url(candidate.url):
            return False
        if candidate.source_type == "youtube" and candidate.is_official:
            return True
        if _is_official_youtube_channel(candidate.url):
            return True

        text = " ".join(
            part
            for part in [candidate.title, candidate.url, candidate.snippet, candidate.publisher or ""]
            if part
        )
        if _has_strong_entity_anchor(task, text):
            return True

        host = urlparse(candidate.url).netloc.lower()
        if host.endswith(".gov") or "paloalto.gov" in host or "ceqanet" in host:
            return _has_project_token_overlap(task, text, minimum=2)
        return False

    def is_relevant_source(self, task: AcquisitionTask, source: RawSource) -> bool:
        if source.source_type in {"local_html", "local_pdf"}:
            return True
        if _is_blocked_url(source.source_url):
            return False

        text = " ".join([source.source_title, source.source_url, source.text[:5000]])
        if _has_strong_entity_anchor(task, text):
            return True
        return _has_project_token_overlap(task, text, minimum=3)


def _is_blocked_url(url: str) -> bool:
    lower = url.lower()
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "vertexaisearch.cloud.google.com/grounding-api-redirect" in lower:
        return True
    if host == "efile.cityofpaloalto.org" and "/public/search/campaign" in path:
        return True
    if host == "data.cityofpaloalto.org" and path in {"", "/"}:
        return True
    if host == "data.cityofpaloalto.org" and any(part in path for part in ["/dashboards/", "/dataviews/", "/datasets/"]):
        return True
    if host == "opengis.cityofpaloalto.org" and path.rstrip("/") in {"", "/opengisdata"}:
        return True
    return False


def _is_official_youtube_channel(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower().rstrip("/")
    return host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and path.startswith("/@cityofpaloalto")


def _has_strong_entity_anchor(task: AcquisitionTask, text: str) -> bool:
    normalized = _normalize(text)
    anchors = [
        task.entity.project_name or "",
        task.entity.name or "",
        task.entity.address or "",
    ]
    for anchor in anchors:
        if anchor and _normalize(anchor) in normalized:
            return True

    address = task.entity.address or ""
    match = re.match(r"\s*(\d+)\s+([A-Za-z]+)", address)
    if match:
        street_anchor = _normalize(" ".join(match.groups()))
        if street_anchor and street_anchor in normalized:
            return True
    return False


def _has_project_token_overlap(task: AcquisitionTask, text: str, minimum: int) -> bool:
    text_tokens = set(_tokens(text))
    entity_tokens = set(_tokens(" ".join([task.entity.project_name or "", task.entity.address or ""])))
    city_tokens = set(_tokens(task.entity.city or ""))
    important = {token for token in entity_tokens if len(token) >= 4 or token.isdigit()}
    if not important:
        return False
    overlap = important & text_tokens
    if city_tokens and city_tokens <= text_tokens:
        return len(overlap) >= max(1, minimum - 1)
    return len(overlap) >= minimum


def _normalize(value: str) -> str:
    return " ".join(_tokens(value))


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in re.split(r"[^A-Za-z0-9]+", value) if len(token) >= 2]
