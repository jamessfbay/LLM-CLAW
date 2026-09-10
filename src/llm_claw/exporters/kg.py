from __future__ import annotations

from llm_claw.models import EvidencePack


def export_for_llm_kg(pack: EvidencePack) -> dict:
    documents = []
    evidence_records = []
    claims = []
    raw_sources = {source.id: source for source in pack.raw_sources}

    for source in pack.raw_sources:
        documents.append(
            {
                "id": source.id,
                "title": source.source_title,
                "source_path": source.raw_path or source.source_url,
                "source_type": _kg_source_type(source.source_type),
                "content": source.text,
                "hash": source.content_hash,
                "metadata": {
                    "source_url": source.source_url,
                    "publisher": source.publisher,
                    "retrieved_at": source.retrieved_at.isoformat(),
                    "llm_claw_source_type": source.source_type,
                },
            }
        )

    for item in pack.evidence:
        evidence_id = item.id
        source = raw_sources.get(item.source_id or "")
        decision_grade = bool(
            item.decision_grade
            and source
            and source.content_hash == item.raw_content_hash
            and source.text[item.quote_start:item.quote_end] == item.evidence_text
        )
        evidence_records.append(
            {
                "id": evidence_id,
                "source_id": item.source_id,
                "quote": item.evidence_text,
                "page_number": item.page_number,
                "url": item.source_url,
                "source_mode": "native_text",
                "confidence": item.confidence,
                "review_state": "auto_accepted" if decision_grade else "pending_review",
                "source_content_hash": item.raw_content_hash,
                "quote_start": item.quote_start,
                "quote_end": item.quote_end,
                "observed_at": item.observed_at.isoformat() if item.observed_at else None,
                "extractor_version": item.extractor_version,
                "governance_notes": (
                    "Exact quote bound to an immutable LLM-CLAW raw source."
                    if decision_grade
                    else "Quote is not position-bound to an immutable raw source and requires review."
                ),
            }
        )
        claims.append(
            {
                "id": item.claim_id,
                "text": item.claim,
                "source_ids": [item.source_id] if item.source_id else [],
                "evidence_ids": [evidence_id] if evidence_id else [],
                "confidence": item.confidence,
                "status": "active" if decision_grade else "uncertain",
                "review_state": "auto_accepted" if decision_grade else "pending_review",
                "observed_at": item.observed_at.isoformat() if item.observed_at else None,
                "governance_notes": "LLM provider summaries were not used as final facts.",
            }
        )

    return {
        "format": "llm-kg-import",
        "contract_version": "evidence-import/2.0",
        "request_id": pack.request_id,
        "documents": documents,
        "evidence": evidence_records,
        "claims": claims,
        "missing_data": pack.missing_data,
        "recommended_next_actions": pack.recommended_next_actions,
    }


def _kg_source_type(source_type: str) -> str:
    if "pdf" in source_type:
        return "pdf"
    if source_type in {"official_html", "webpage", "local_html", "government_api"}:
        return "md"
    return "txt"
