from __future__ import annotations

from llm_claw.models import EvidencePack


def export_for_llm_kg(pack: EvidencePack) -> dict:
    documents = []
    evidence_records = []
    claims = []

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
        evidence_id = item.claim_id.replace("claim_", "ev_") if item.claim_id else None
        evidence_records.append(
            {
                "id": evidence_id,
                "source_id": item.source_id,
                "quote": item.evidence_text,
                "page_number": item.page_number,
                "url": item.source_url,
                "source_mode": "native_text",
                "confidence": item.confidence,
                "review_state": "auto_accepted",
                "governance_notes": "Created by LLM-CLAW from raw fetched source.",
            }
        )
        claims.append(
            {
                "id": item.claim_id,
                "text": item.claim,
                "source_ids": [item.source_id] if item.source_id else [],
                "evidence_ids": [evidence_id] if evidence_id else [],
                "confidence": item.confidence,
                "status": "active" if item.evidence_text else "uncertain",
                "review_state": "auto_accepted" if item.evidence_text else "pending_review",
                "governance_notes": "LLM provider summaries were not used as final facts.",
            }
        )

    return {
        "format": "llm-kg-import",
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
