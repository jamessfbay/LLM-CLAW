from __future__ import annotations

import re

from llm_claw.models import AcquisitionTask, ExtractedClaim, RawSource
from llm_claw.pipeline.source_filter import SourceRelevanceFilter


class ContentExtractor:
    def __init__(self) -> None:
        self.relevance_filter = SourceRelevanceFilter()

    def extract_text(self, task: AcquisitionTask, sources: list[RawSource]) -> list[RawSource]:
        non_empty = [source for source in sources if source.text.strip()]
        return self.relevance_filter.filter_sources(task, non_empty)


class EvidenceExtractor:
    def extract_claims(self, task: AcquisitionTask, sources: list[RawSource]) -> list[ExtractedClaim]:
        claims: list[ExtractedClaim] = []
        for source in sources:
            sentences = _sentences(source.text)
            for need in task.data_needed:
                evidence = _find_sentence(sentences, need)
                if not evidence:
                    continue
                claims.append(
                    ExtractedClaim(
                        text=_claim_text(task, need, evidence),
                        subject=task.entity.display_name,
                        predicate=_predicate_for_need(need),
                        object=_object_for_need(need, evidence),
                        source_id=source.id,
                        evidence_text=evidence,
                        confidence=0.62,
                    )
                )
            if source.source_type == "youtube":
                claims.extend(_youtube_city_development_claims(task, source, sentences))
        return claims


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?。])\s+", cleaned)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def _find_sentence(sentences: list[str], need: str) -> str | None:
    terms = [term for term in re.split(r"[\s/_-]+", need.lower()) if len(term) >= 4]
    if _is_city_development_need(need):
        return _find_city_development_sentence(sentences)
    if "status" in need.lower():
        for sentence in sentences:
            lower = sentence.lower()
            if (
                "under review" in lower
                or "approved" in lower
                or "pending" in lower
                or "notice of preparation" in lower
                or "draft eir" in lower
            ):
                return sentence
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in terms):
            return sentence
    return None


def _find_city_development_sentence(sentences: list[str]) -> str | None:
    for sentence in sentences:
        lower = sentence.lower()
        if _is_negative_availability_sentence(lower):
            continue
        if any(keyword in lower for keyword in _SUBSTANTIVE_CITY_DEVELOPMENT_KEYWORDS):
            return sentence
    return None


def _youtube_city_development_claims(
    task: AcquisitionTask, source: RawSource, sentences: list[str]
) -> list[ExtractedClaim]:
    claims: list[ExtractedClaim] = []
    seen: set[str] = set()
    for sentence in sentences:
        lower = sentence.lower()
        if _is_negative_availability_sentence(lower):
            continue
        if not any(keyword in lower for keyword in _SUBSTANTIVE_CITY_DEVELOPMENT_KEYWORDS):
            continue
        normalized = re.sub(r"\s+", " ", sentence).strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        claims.append(
            ExtractedClaim(
                text=f"{task.entity.city or 'The city'} has city construction or planning information mentioned in a YouTube source.",
                subject=task.entity.city or task.entity.display_name,
                predicate="has_city_development_topic",
                object="city construction and planning",
                source_id=source.id,
                evidence_text=sentence,
                confidence=0.6,
            )
        )
        if len(claims) >= 8:
            break
    return claims


def _is_negative_availability_sentence(lower_sentence: str) -> bool:
    negative_markers = [
        "no detailed",
        "no direct reference",
        "no direct references",
        "no visible information",
        "not visible",
        "not available",
        "no meeting transcript",
        "no transcript",
        "no specific transcript",
        "no specific transcripts",
        "no specific agenda",
        "were available in the provided text",
        "no specific mention",
        "no specific mentions",
        "could not be found",
        "cannot be found",
        "not found",
    ]
    return any(marker in lower_sentence for marker in negative_markers)


def _is_city_development_need(need: str) -> bool:
    lower = need.lower()
    matches = [keyword for keyword in _SUBSTANTIVE_CITY_DEVELOPMENT_KEYWORDS if keyword in lower]
    return len(matches) >= 2


def _claim_text(task: AcquisitionTask, need: str, evidence: str) -> str:
    entity = task.entity.display_name
    if "status" in need.lower():
        return f"{entity} has a planning status mentioned in the fetched source."
    if "ceqa" in need.lower():
        return f"{entity} has a CEQA-related record mentioned in the fetched source."
    if "comment" in need.lower():
        return f"{entity} has public comment information mentioned in the fetched source."
    if "staff report" in need.lower():
        return f"{entity} is mentioned in a staff report or planning record."
    return f"{entity} has source-linked information for {need}."


def _predicate_for_need(need: str) -> str:
    lower = need.lower()
    if "status" in lower:
        return "has_status"
    if "ceqa" in lower:
        return "has_ceqa_status"
    if "comment" in lower:
        return "has_public_comment_record"
    if "staff report" in lower:
        return "mentioned_in"
    return "has_source_linked_data"


def _object_for_need(need: str, evidence: str) -> str:
    lower = evidence.lower()
    if "under review" in lower:
        return "under review"
    if "ceqa" in need.lower():
        return "CEQA record"
    if "staff report" in need.lower():
        return "staff report"
    return need


_CITY_DEVELOPMENT_KEYWORDS = {
    "construction",
    "development",
    "housing",
    "zoning",
    "land use",
    "planning",
    "permit",
    "permitting",
    "infrastructure",
    "public works",
    "transportation",
    "transit",
    "ceqa",
    "builder's remedy",
    "builders remedy",
    "affordable housing",
    "density",
    "height",
    "agenda",
    "project",
}


_SUBSTANTIVE_CITY_DEVELOPMENT_KEYWORDS = _CITY_DEVELOPMENT_KEYWORDS - {"city", "project", "agenda"}
