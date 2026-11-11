from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace
from llm_claw.providers.aisa_client import AIsaClient


class AIsaApolloPeopleProvider:
    """Find named economic buyers with public professional profile URLs."""

    name = "aisa_apollo_people"
    search_endpoint = "/apis/v1/apollo/mixed_people/api_search"
    match_endpoint = "/apis/v1/apollo/people/match"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AIsaClient(settings)

    def available(self) -> bool:
        return bool(self.settings.aisa_api_key)

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        if not self.available():
            return [], ProviderTrace(
                provider=self.name,
                status="disabled",
                query=query.text,
                message="AISA_API_KEY is not configured; provider skipped.",
            )
        keywords = _people_query(query.text)
        if not keywords:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message="Empty people query")
        try:
            payload = self.client.post(
                self.search_endpoint,
                params={
                    "q_keywords": keywords,
                    "include_similar_titles": "true",
                    "person_seniorities[]": ["owner", "founder", "c_suite", "vp", "head", "director"],
                    "page": 1,
                    "per_page": 10,
                },
            )
        except Exception as exc:
            return [], ProviderTrace(
                provider=self.name,
                status="error",
                query=query.text,
                message=f"AIsa Apollo request failed: {type(exc).__name__}",
            )

        retrieved_at = datetime.now(timezone.utc).isoformat()
        candidates: list[CandidateSource] = []
        enrichment_errors = 0
        search_people = _people_items(payload)[: self.settings.aisa_apollo_people_max_enrichments]
        for person in search_people:
            person_id = str(person.get("id") or person.get("person_id") or "").strip()
            if not person_id:
                continue
            try:
                enriched_payload = self.client.post(
                    self.match_endpoint,
                    payload={"id": person_id, "reveal_personal_emails": False},
                    params={
                        "reveal_phone_number": "false",
                        "run_waterfall_email": "false",
                        "run_waterfall_phone": "false",
                    },
                )
            except Exception:
                enrichment_errors += 1
                continue
            enriched_person = enriched_payload.get("person")
            normalized = _normalize_person(enriched_person if isinstance(enriched_person, dict) else {})
            if not normalized:
                continue
            request_id = _request_id(enriched_payload)
            metadata: dict[str, Any] = {
                "platform": "linkedin",
                "person_name": normalized["person_name"],
                "profile_url": normalized["profile_url"],
                "target_role": normalized["target_role"],
                "organization_name": normalized["organization_name"],
                "identity_kind": "public_professional_profile",
                "source_provider": "aisa_apollo_people",
                "source_label": "AISA Apollo People Enrichment",
                "source_method": "licensed_public_professional_data",
                "retrieved_at": retrieved_at,
            }
            if normalized.get("match_confidence"):
                metadata["match_confidence"] = normalized["match_confidence"]
            if request_id:
                metadata["source_request_id"] = request_id
            candidates.append(CandidateSource(
                query_id=query.id,
                provider=self.name,
                title=(
                    f"{normalized['person_name']} — {normalized['target_role']} at "
                    f"{normalized['organization_name']}"
                ),
                url=normalized["profile_url"],
                snippet=(
                    f"Public professional profile: {normalized['person_name']}, "
                    f"{normalized['target_role']} at {normalized['organization_name']}."
                ),
                publisher="LinkedIn",
                source_type="candidate",
                confidence=0.7,
                is_official=False,
                analysis_note=(
                    "Named professional returned by AIsa Apollo. Role fit is not evidence of purchase intent."
                ),
                metadata=metadata,
            ))
            if len(candidates) >= min(task.source_policy.max_sources, 10):
                break
        return candidates, ProviderTrace(
            provider=self.name,
            query=query.text,
            candidate_count=len(candidates),
            message=(
                f"AIsa Apollo verified {len(candidates)} public professional profile candidate(s) "
                f"from {len(search_people)} bounded enrichment attempt(s); {enrichment_errors} request(s) failed."
            ),
        )


def _people_query(text: str) -> str:
    value = re.sub(r"site:(?:www\.)?linkedin\.com/in\s*", "", text, flags=re.IGNORECASE)
    quoted = next((item.strip() for item in re.findall(r'"([^"]+)"', value) if item.strip()), "")
    if quoted:
        return quoted[:120]
    common_domain = re.search(
        r"\b(?:AI|coding|software|sales|support|research|autonomous|developer)\s+agents?\b",
        value,
        flags=re.IGNORECASE,
    )
    if common_domain:
        return common_domain.group(0)[:120]
    value = re.sub(
        r"\b(?:buyer|buyers|economic|founder|governance|leader|named|owner|director|head|platform|security)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return " ".join(re.sub(r"\s+", " ", value).strip().split()[:4])[:120]


def _people_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[Any] = [payload.get("people"), payload.get("contacts")]
    data = payload.get("data")
    if isinstance(data, dict):
        containers.extend([data.get("people"), data.get("contacts"), data.get("items"), data.get("results")])
    elif isinstance(data, list):
        containers.append(data)
    containers.extend([payload.get("items"), payload.get("results")])
    for container in containers:
        if isinstance(container, list):
            return [item for item in container if isinstance(item, dict)]
    return []


def _normalize_person(person: dict[str, Any]) -> dict[str, str] | None:
    match_confidence = str(person.get("match_confidence") or "").strip().lower()
    if match_confidence not in {"high", "medium"}:
        return None
    first = str(person.get("first_name") or "").strip()
    last = str(person.get("last_name") or "").strip()
    name = str(person.get("name") or " ".join(part for part in (first, last) if part)).strip()
    title = str(person.get("title") or person.get("headline") or "").strip()
    organization = person.get("organization") if isinstance(person.get("organization"), dict) else {}
    organization_name = str(
        person.get("organization_name") or person.get("company_name") or organization.get("name") or ""
    ).strip()
    profile_url = _canonical_linkedin_profile(str(
        person.get("linkedin_url") or person.get("linkedin_profile_url") or person.get("profile_url") or ""
    ).strip())
    if not name or not title or not organization_name or not _valid_linkedin_profile(profile_url):
        return None
    normalized = {
        "person_name": name[:160],
        "target_role": title[:240],
        "organization_name": organization_name[:240],
        "profile_url": profile_url,
    }
    normalized["match_confidence"] = match_confidence
    return normalized


def _canonical_linkedin_profile(value: str) -> str:
    if value.startswith("http://"):
        return f"https://{value.removeprefix('http://')}"
    return value


def _valid_linkedin_profile(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in {
        "linkedin.com", "www.linkedin.com",
    } and parsed.path.startswith("/in/")


def _request_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return str(payload.get("request_id") or payload.get("requestId") or metadata.get("request_id") or "")[:160]
