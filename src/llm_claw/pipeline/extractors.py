from __future__ import annotations

from datetime import datetime, timezone
import json
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
            structured_claims = _structured_api_claims(task, source)
            if structured_claims:
                claims.extend(structured_claims)
                continue
            sentences = _sentences(source.text)
            if source.metadata.get("mock"):
                excluded_mock_text = {
                    _normalized_sentence(source.source_title),
                    _normalized_sentence(str(source.metadata.get("candidate_snippet", ""))),
                }
                sentences = [sentence for sentence in sentences if _normalized_sentence(sentence) not in excluded_mock_text]
            used_evidence: set[str] = set()
            for need in task.data_needed:
                evidence = _find_sentence(sentences, need, excluded=used_evidence)
                if not evidence:
                    evidence = _find_sentence(sentences, need)
                if not evidence:
                    continue
                used_evidence.add(_normalized_sentence(evidence))
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
            claims.extend(_official_planning_page_claims(task, source, sentences))
            if source.source_type == "youtube":
                claims.extend(_youtube_city_development_claims(task, source, sentences))
        return claims


def _structured_api_claims(task: AcquisitionTask, source: RawSource) -> list[ExtractedClaim]:
    if source.source_type != "government_api" and "json" not in str(source.metadata.get("content_type", "")).lower():
        return []
    try:
        payload = json.loads(source.text)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        return []

    features = payload.get("features")
    if not isinstance(features, list):
        return []
    changed_ids = {
        str(value)
        for value in task.entity.metadata.get("changed_record_ids", [])
        if value
    }
    removed_ids = {
        str(value)
        for value in task.entity.metadata.get("removed_record_ids", [])
        if value
    }
    source_updated_at = _epoch_millis_to_iso(
        payload.get("metadata", {}).get("generated")
        if isinstance(payload.get("metadata"), dict)
        else None
    )
    removal_claims = [
        ExtractedClaim(
            text=(
                f"USGS event ID {event_id} is not present in the current feed generated at "
                f"{source_updated_at}; this may indicate that it aged out of the feed window "
                f"or was removed. Official source citation: {source.source_url}"
            ),
            subject=event_id,
            predicate="absent_from_current_feed",
            object=source_updated_at,
            source_id=source.id,
            evidence_text=(
                f"The current USGS feed generated at {source_updated_at} does not contain "
                f"event ID {event_id}. Official source citation: {source.source_url}"
            ),
            confidence=0.82,
            status="uncertain",
        )
        for event_id in sorted(removed_ids)
    ]
    if removed_ids and not changed_ids:
        return removal_claims
    selected = [
        feature
        for feature in features
        if isinstance(feature, dict) and (not changed_ids or str(feature.get("id")) in changed_ids)
    ]
    if not selected:
        return []

    claims: list[ExtractedClaim] = removal_claims
    for feature in selected:
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(properties, dict) or properties.get("type") != "earthquake":
            continue
        coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if not isinstance(coordinates, list) or len(coordinates) < 3:
            coordinates = [None, None, None]
        event_id = str(feature.get("id") or properties.get("code") or "unknown")
        event_url = str(properties.get("url") or source.source_url)
        magnitude = properties.get("mag")
        place = str(properties.get("place") or "unknown location")
        status = str(properties.get("status") or "unknown")
        alert = properties.get("alert")
        tsunami = int(properties.get("tsunami") or 0)
        event_time = _epoch_millis_to_iso(properties.get("time"))
        updated_time = _epoch_millis_to_iso(properties.get("updated"))
        evidence_text = (
            f"USGS event ID {event_id}; magnitude {magnitude}; event time {event_time}; "
            f"updated time {updated_time}; place {place}; coordinates "
            f"{coordinates[0]}, {coordinates[1]}; depth {coordinates[2]} km; "
            f"review status {status}; alert {alert or 'none'}; tsunami indicator {tsunami}. "
            f"Official source citation: {event_url}"
        )
        claims.append(
            ExtractedClaim(
                text=evidence_text,
                subject=event_id,
                predicate="reported_earthquake_event",
                object=place,
                source_id=source.id,
                evidence_text=evidence_text,
                confidence=0.96,
            )
        )
    return claims


def _epoch_millis_to_iso(value: object) -> str:
    try:
        return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return "unknown"


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    lines = [
        cleaned
        for line in text.splitlines()
        if (cleaned := re.sub(r"\s+", " ", line).strip())
    ]
    for cleaned in lines:
        parts.extend(re.split(r"(?<=[.!?。])\s+", cleaned))
    # Government detail pages often render a field label and its value in
    # adjacent block elements. Preserve short windows so "Accession No." can be
    # extracted together with the identifier that follows it.
    for start in range(len(lines)):
        for width in range(2, 5):
            window = " ".join(lines[start:start + width])
            if len(window) >= 20:
                parts.append(window)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def _find_sentence(sentences: list[str], need: str, excluded: set[str] | None = None) -> str | None:
    excluded = excluded or set()
    terms = [term for term in re.split(r"[\s/_-]+", need.lower()) if len(term) >= 3]
    usable = [
        sentence
        for sentence in sentences
        if _normalized_sentence(sentence) not in excluded and not _is_boilerplate_sentence(sentence)
    ]
    lower_need = need.lower()
    if "cve" in lower_need and ("identifier" in lower_need or "id" in lower_need):
        return next((sentence for sentence in usable if re.search(r"\bCVE-\d{4}-\d{4,}\b", sentence, re.I)), None)
    if "date" in lower_need:
        labeled = next(
            (
                sentence
                for sentence in usable
                if re.search(r"\b(date issued|release date|recall date|effective date)\b", sentence, re.I)
            ),
            None,
        )
        if labeled:
            return labeled
        dated = next((sentence for sentence in usable if _contains_absolute_date(sentence)), None)
        if dated:
            return dated
    if _is_city_development_need(need):
        return _find_city_development_sentence(usable)
    if "status" in need.lower():
        for sentence in usable:
            lower = sentence.lower()
            if (
                "under review" in lower
                or "approved" in lower
                or "pending" in lower
                or "notice of preparation" in lower
                or "draft eir" in lower
            ):
                return sentence
    ranked = sorted(
        ((_sentence_match_score(sentence, terms, lower_need), sentence) for sentence in usable),
        key=lambda item: item[0],
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] > 0 else None


def _sentence_match_score(sentence: str, terms: list[str], need: str) -> int:
    tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    matched = sum(1 for term in terms if _term_matches(term, tokens))
    phrase_bonus = 3 if need in sentence.lower() else 0
    substantive_bonus = 1 if 30 <= len(sentence) <= 600 else 0
    return matched * 4 + phrase_bonus + substantive_bonus


def _term_matches(term: str, tokens: set[str]) -> bool:
    if term in tokens:
        return True
    stem = term[:7] if len(term) >= 7 else term
    return len(stem) >= 5 and any(token.startswith(stem) for token in tokens)


def _is_boilerplate_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    return any(
        marker in lower
        for marker in [
            "skip to main content",
            "an official website of",
            "official websites use .gov",
            "here's how you know",
            "here’s how you know",
            "subscribe to email updates",
        ]
    )


def _contains_absolute_date(sentence: str) -> bool:
    month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    return bool(
        re.search(rf"\b{month}\s+\d{{1,2}},\s+\d{{4}}\b", sentence, re.I)
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", sentence)
    )


def _normalized_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip().lower()


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


def _official_planning_page_claims(
    task: AcquisitionTask, source: RawSource, sentences: list[str]
) -> list[ExtractedClaim]:
    if source.source_type not in {"official_html", "webpage", "local_html", "government_api"}:
        return []
    if not (source.publisher or source.metadata.get("content_type") or source.source_url):
        return []
    if not _has_planning_or_permit_need(task):
        return []
    for sentence in sentences:
        lower = sentence.lower()
        if _is_negative_availability_sentence(lower):
            continue
        if any(keyword in lower for keyword in _OFFICIAL_PLANNING_PAGE_KEYWORDS):
            return [
                ExtractedClaim(
                    text=f"{task.entity.display_name} has official planning or permit information available in the fetched source.",
                    subject=task.entity.display_name,
                    predicate="has_source_linked_data",
                    object="official planning or permit information",
                    source_id=source.id,
                    evidence_text=sentence,
                    confidence=0.48,
                    status="uncertain",
                )
            ]
    return []


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


def _has_planning_or_permit_need(task: AcquisitionTask) -> bool:
    text = " ".join([task.question or "", task.acquisition_instruction or "", *task.data_needed]).lower()
    return any(keyword in text for keyword in _OFFICIAL_PLANNING_PAGE_KEYWORDS)


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
    return evidence


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

_OFFICIAL_PLANNING_PAGE_KEYWORDS = {
    "permit",
    "permits",
    "permitting",
    "planning",
    "entitlement",
    "entitlements",
    "development",
    "staff report",
    "agenda",
    "zoning",
    "public hearing",
    "appeal",
}
