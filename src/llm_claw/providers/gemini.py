from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace
from llm_claw.providers.openai_web_search import _build_prompt, _parse_json_rows


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return bool(self.settings.gemini_api_key)

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        if not self.settings.gemini_api_key:
            return [], ProviderTrace(
                provider=self.name,
                status="disabled",
                query=query.text,
                message="GEMINI_API_KEY is not configured; provider skipped.",
            )

        try:
            payload = self._generate_content(_build_prompt(task, query))
        except Exception as exc:
            return [], ProviderTrace(
                provider=self.name,
                status="error",
                query=query.text,
                message=f"Gemini web search failed: {type(exc).__name__}: {exc}",
            )

        candidates = _parse_candidates(payload, task, query)
        return candidates, ProviderTrace(
            provider=self.name,
            status="ok",
            query=query.text,
            candidate_count=len(candidates),
            message=f"Gemini web search returned {len(candidates)} candidate source(s).",
        )

    def _generate_content(self, prompt: str) -> dict[str, Any]:
        params = urlencode({"key": self.settings.gemini_api_key or ""})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_model}:generateContent?{params}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
        }
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))


def _parse_candidates(payload: dict[str, Any], task: AcquisitionTask, query: PlannedQuery) -> list[CandidateSource]:
    text = _extract_text(payload)
    rows = _parse_json_rows(text)
    urls_from_grounding = _extract_grounding_urls(payload)

    candidates: list[CandidateSource] = []
    for row in rows:
        url = str(row.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or _is_grounding_redirect(url):
            continue
        candidates.append(
            CandidateSource(
                query_id=query.id,
                provider="gemini",
                title=str(row.get("title") or url),
                url=url,
                snippet=str(row.get("snippet") or ""),
                publisher=str(row.get("publisher") or "") or None,
                source_type="candidate",
                confidence=0.76 if row.get("is_official", True) else 0.54,
                is_official=bool(row.get("is_official", True)),
                analysis_note="Discovered by Gemini web search; must be fetched by crawler before becoming evidence.",
            )
        )

    known = {candidate.url for candidate in candidates}
    for title, url in urls_from_grounding:
        if url not in known and not _is_grounding_redirect(url) and _looks_relevant(url, task):
            candidates.append(
                CandidateSource(
                    query_id=query.id,
                    provider="gemini",
                    title=title or url,
                    url=url,
                    snippet="URL returned in Gemini grounding metadata.",
                    source_type="candidate",
                    confidence=0.62,
                    is_official=_looks_official(url),
                    analysis_note="Discovered in Gemini grounding metadata; must be fetched by crawler before becoming evidence.",
                )
            )
            known.add(url)
    return candidates[:5]


def _extract_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        for part in content.get("parts", []):
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def _extract_grounding_urls(payload: dict[str, Any]) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    for candidate in payload.get("candidates", []):
        metadata = candidate.get("groundingMetadata", {}) if isinstance(candidate, dict) else {}
        for chunk in metadata.get("groundingChunks", []):
            web = chunk.get("web") if isinstance(chunk, dict) else None
            if not isinstance(web, dict):
                continue
            uri = str(web.get("uri") or "").strip()
            title = str(web.get("title") or "").strip()
            if uri.startswith(("http://", "https://")):
                urls.append((title, uri))
    return urls


def _looks_relevant(url: str, task: AcquisitionTask) -> bool:
    haystack = url.lower()
    terms = [
        task.entity.project_name or "",
        task.entity.address or "",
        task.entity.city or "",
        "ceqanet",
        "paloalto",
        "planning",
    ]
    tokens = [token.lower() for term in terms for token in re.split(r"[^a-zA-Z0-9]+", term) if len(token) > 2]
    return any(token in haystack for token in tokens)


def _looks_official(url: str) -> bool:
    host = url.lower()
    return ".gov" in host or "paloalto.gov" in host or "ceqanet" in host


def _is_grounding_redirect(url: str) -> bool:
    return "vertexaisearch.cloud.google.com/grounding-api-redirect" in url.lower()
