from __future__ import annotations

from llm_claw.models import ExtractedClaim, ProviderTrace, RawSource, VerificationNote


class ClaudeVerificationAgent:
    name = "claude"

    def verify(self, claims: list[ExtractedClaim], sources: list[RawSource]) -> tuple[list[VerificationNote], ProviderTrace]:
        source_text_by_id = {source.id: source.text.lower() for source in sources}
        notes: list[VerificationNote] = []
        for claim in claims:
            source_text = source_text_by_id.get(claim.source_id, "")
            evidence = claim.evidence_text.lower()
            if evidence and evidence in source_text:
                support_level = "full"
                confidence = 0.82
                rationale = "The evidence quote is present in the fetched raw source."
            else:
                support_level = "unclear"
                confidence = 0.45
                rationale = "The evidence quote could not be matched exactly in the fetched source."
            notes.append(
                VerificationNote(
                    provider=self.name,
                    claim_id=claim.id,
                    support_level=support_level,
                    rationale=rationale,
                    confidence=confidence,
                )
            )
        return notes, ProviderTrace(provider=self.name, candidate_count=0, message=f"Verified {len(notes)} claims.")
