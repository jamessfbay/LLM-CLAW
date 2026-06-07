from __future__ import annotations

import re

from llm_claw.models import AcquisitionTask, ExtractedClaim, RawSource


class ContentExtractor:
    def extract_text(self, sources: list[RawSource]) -> list[RawSource]:
        return [source for source in sources if source.text.strip()]


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
        return claims


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?。])\s+", cleaned)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def _find_sentence(sentences: list[str], need: str) -> str | None:
    terms = [term for term in re.split(r"[\s/_-]+", need.lower()) if len(term) >= 4]
    if "status" in need.lower():
        for sentence in sentences:
            lower = sentence.lower()
            if "under review" in lower or "approved" in lower or "pending" in lower:
                return sentence
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in terms):
            return sentence
    return None


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
