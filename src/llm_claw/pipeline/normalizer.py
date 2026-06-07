from __future__ import annotations

from llm_claw.models import AcquisitionTask, ExtractedClaim


class DataNormalizer:
    def normalize(self, task: AcquisitionTask, claims: list[ExtractedClaim]) -> dict[str, str]:
        structured: dict[str, str] = {}
        for claim in claims:
            if claim.predicate == "has_status" and "approval_status" not in structured:
                structured["approval_status"] = claim.object or "mentioned"
            if claim.predicate == "has_ceqa_status" and "ceqa_status" not in structured:
                structured["ceqa_status"] = claim.object or "mentioned"
            if claim.predicate == "has_public_comment_record" and "public_comments" not in structured:
                structured["public_comments"] = "mentioned"
        return structured


class ConfidenceScorer:
    def score(self, base_confidence: float, source_type: str, verified: bool) -> float:
        score = base_confidence
        if source_type in {"official_html", "official_pdf", "government_api", "local_pdf", "local_html"}:
            score += 0.18
        if verified:
            score += 0.12
        return max(0.0, min(1.0, round(score, 2)))
