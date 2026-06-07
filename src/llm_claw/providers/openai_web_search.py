from __future__ import annotations

import json
import re
from typing import Any
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace


class OpenAIWebSearchProvider:
    name = "openai_web_search"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return bool(self.settings.openai_api_key)

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        if not self.settings.openai_api_key:
            return [], ProviderTrace(
                provider=self.name,
                status="disabled",
                query=query.text,
                message="OPENAI_API_KEY is not configured; provider skipped.",
            )

        prompt = _build_prompt(task, query)
        try:
            payload = self._responses_request(prompt)
        except Exception as exc:
            return [], ProviderTrace(
                provider=self.name,
                status="error",
                query=query.text,
                message=f"OpenAI web search failed: {type(exc).__name__}: {exc}",
            )

        candidates = _parse_candidates(payload, task, query)
        return candidates, ProviderTrace(
            provider=self.name,
            status="ok",
            query=query.text,
            candidate_count=len(candidates),
            message=f"OpenAI web search returned {len(candidates)} candidate source(s).",
        )

    def _responses_request(self, prompt: str) -> dict[str, Any]:
        body = {
            "model": self.settings.openai_model,
            "tools": [{"type": "web_search"}],
            "tool_choice": "auto",
            "input": prompt,
        }
        request = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))


def _build_prompt(task: AcquisitionTask, query: PlannedQuery) -> str:
    entity = task.entity.display_name
    address = task.entity.address or ""
    city = task.entity.city or ""
    needs = ", ".join(task.data_needed)
    return (
        "Find authoritative source URLs for a source-linked data acquisition agent.\n"
        "Return only official or high-quality candidate source pages, not a final answer.\n"
        "Prefer city/government pages, CEQA records, official PDFs, planning agendas, staff reports, and permit records.\n"
        "Return JSON only with this shape: "
        '[{"title":"...", "url":"https://...", "snippet":"...", "publisher":"...", "is_official":true}].\n'
        f"Entity: {entity}\n"
        f"Address: {address}\n"
        f"City: {city}\n"
        f"Data needed: {needs}\n"
        f"Current query: {query.text}\n"
    )


def _parse_candidates(payload: dict[str, Any], task: AcquisitionTask, query: PlannedQuery) -> list[CandidateSource]:
    text = _extract_output_text(payload)
    rows = _parse_json_rows(text)
    urls_from_sources = _extract_urls(payload)

    candidates: list[CandidateSource] = []
    for row in rows:
        url = str(row.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        candidates.append(
            CandidateSource(
                query_id=query.id,
                provider="openai_web_search",
                title=str(row.get("title") or url),
                url=url,
                snippet=str(row.get("snippet") or ""),
                publisher=str(row.get("publisher") or "") or None,
                source_type="candidate",
                confidence=0.78 if row.get("is_official", True) else 0.55,
                is_official=bool(row.get("is_official", True)),
                analysis_note="Discovered by OpenAI web search; must be fetched by crawler before becoming evidence.",
            )
        )

    known = {candidate.url for candidate in candidates}
    for url in urls_from_sources:
        if url not in known and _looks_relevant(url, task):
            candidates.append(
                CandidateSource(
                    query_id=query.id,
                    provider="openai_web_search",
                    title=url,
                    url=url,
                    snippet="URL returned in OpenAI web search sources.",
                    source_type="candidate",
                    confidence=0.62,
                    is_official=_looks_official(url),
                    analysis_note="Discovered in OpenAI web search sources; must be fetched by crawler before becoming evidence.",
                )
            )
            known.add(url)
    return candidates[:5]


def _extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []) if isinstance(item, dict) else []:
            text = content.get("text") if isinstance(content, dict) else None
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def _parse_json_rows(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _extract_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "url" and isinstance(item, str) and item.startswith(("http://", "https://")):
                urls.append(item)
            else:
                urls.extend(_extract_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_extract_urls(item))
    return list(dict.fromkeys(urls))


def _looks_official(url: str) -> bool:
    return any(domain in url.lower() for domain in [".gov", "paloalto.gov", "ceqanet"])


def _looks_relevant(url: str, task: AcquisitionTask) -> bool:
    lower = url.lower()
    terms = [task.entity.city or "", task.entity.address or "", task.entity.display_name]
    tokens = [token.lower().replace(" ", "-") for token in terms if token]
    return _looks_official(url) or any(token and token.split(",")[0] in lower for token in tokens)
