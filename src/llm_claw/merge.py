from __future__ import annotations

from typing import Callable, TypeVar

from llm_claw.models import CandidateSource, EvidenceItem, EvidencePack, ProviderTrace, RawSource, VerificationNote


T = TypeVar("T")


def merge_evidence_packs(base: EvidencePack, *others: EvidencePack) -> EvidencePack:
    merged = base.model_copy(deep=True)

    for pack in others:
        merged.structured_data.update({key: value for key, value in pack.structured_data.items() if value is not None})
        merged.evidence = _dedupe(merged.evidence + pack.evidence, _evidence_key)
        merged.missing_data = _dedupe_strings(merged.missing_data + pack.missing_data)
        merged.recommended_next_actions = _dedupe_strings(
            merged.recommended_next_actions + pack.recommended_next_actions
        )
        merged.provider_trace = _dedupe(merged.provider_trace + pack.provider_trace, _trace_key)
        merged.candidate_sources = _dedupe(merged.candidate_sources + pack.candidate_sources, _candidate_key)
        merged.raw_sources = _dedupe(merged.raw_sources + pack.raw_sources, _raw_source_key)
        merged.verification_notes = _dedupe(merged.verification_notes + pack.verification_notes, _verification_key)

    merged.summary = (
        f"{merged.entity.get('project_name') or merged.entity.get('name') or 'Entity'} has "
        f"{len(merged.evidence)} source-linked evidence item(s); "
        f"{len(merged.missing_data)} data gap(s) remain after merge."
    )
    return merged


def _dedupe(items: list[T], key_fn: Callable[[T], tuple]) -> list[T]:
    seen: set[tuple] = set()
    result: list[T] = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


def _evidence_key(item: EvidenceItem) -> tuple:
    return (item.claim.strip().lower(), item.source_url.strip().lower())


def _trace_key(item: ProviderTrace) -> tuple:
    return (item.provider, item.status, item.query, item.message)


def _candidate_key(item: CandidateSource) -> tuple:
    return (item.url.strip().lower(),)


def _raw_source_key(item: RawSource) -> tuple:
    return (item.source_url.strip().lower(), item.content_hash)


def _verification_key(item: VerificationNote) -> tuple:
    return (item.provider, item.claim_id, item.support_level)
