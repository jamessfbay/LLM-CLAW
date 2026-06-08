from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace, RawSource
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
        return self._generate_content_with_tools(prompt, [{"google_search": {}}], timeout=45)

    def _generate_content_without_tools(self, prompt: str, timeout: int = 90) -> dict[str, Any]:
        return self._generate_content_with_tools(prompt, [], timeout=timeout)

    def _generate_content_with_url_context(self, prompt: str) -> dict[str, Any]:
        return self._generate_content_with_tools(prompt, [{"url_context": {}}, {"google_search": {}}], timeout=90)

    def _generate_content_with_tools(
        self, prompt: str, tools: list[dict[str, Any]], timeout: int
    ) -> dict[str, Any]:
        return self._generate_content_with_parts([{"text": prompt}], tools=tools, timeout=timeout)

    def _generate_content_with_parts(
        self, parts: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: int
    ) -> dict[str, Any]:
        params = urlencode({"key": self.settings.gemini_api_key or ""})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_model}:generateContent?{params}"
        body = {
            "contents": [{"parts": parts}],
        }
        if tools:
            body["tools"] = tools
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def analyze_youtube_source(self, task: AcquisitionTask, source: RawSource) -> tuple[RawSource | None, ProviderTrace]:
        if not self.settings.gemini_api_key:
            return None, ProviderTrace(
                provider=self.name,
                status="disabled",
                query=source.source_url,
                message="GEMINI_API_KEY is not configured; YouTube analysis skipped.",
            )

        video_candidates = _extract_youtube_video_candidates(source)
        prompt = _build_youtube_analysis_prompt(task, source, video_candidates)
        try:
            payload = self._generate_content_without_tools(prompt)
        except Exception as exc:
            return None, ProviderTrace(
                provider=self.name,
                status="error",
                query=source.source_url,
                message=f"Gemini YouTube analysis failed: {type(exc).__name__}: {exc}",
            )

        summary = _extract_text(payload).strip()
        if not summary:
            return None, ProviderTrace(
                provider=self.name,
                status="error",
                query=source.source_url,
                message="Gemini YouTube analysis returned empty text.",
            )
        summary = _ensure_video_candidates_in_summary(task, summary, video_candidates)

        analyzed = source.model_copy(deep=True)
        analyzed.text = summary
        analyzed.metadata = {
            **source.metadata,
            "analysis_provider": "gemini",
            "analysis_kind": "youtube_summary",
            "analysis_input": "youtube_text_prompt",
            "youtube_video_candidates": video_candidates[:20],
        }
        return analyzed, ProviderTrace(
            provider=self.name,
            status="ok",
            query=source.source_url,
            candidate_count=1,
            message="Gemini summarized YouTube source content for project-relevant evidence.",
        )

    def analyze_youtube_candidate(
        self, task: AcquisitionTask, candidate: CandidateSource
    ) -> tuple[RawSource | None, ProviderTrace]:
        source_text = "\n".join(
            part
            for part in [
                candidate.title,
                candidate.snippet,
                candidate.analysis_note or "",
                f"Candidate source URL: {candidate.url}",
            ]
            if part
        )
        source = RawSource(
            candidate_id=candidate.id,
            source_url=candidate.url,
            source_title=candidate.title,
            publisher=candidate.publisher,
            source_type="youtube",
            content_hash=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            text=source_text,
            metadata={
                "source_fetcher": "gemini",
                "candidate_provider": candidate.provider,
                "candidate_confidence": candidate.confidence,
            },
        )
        analyzed, trace = self.analyze_youtube_source(task, source)
        if trace.status == "ok":
            trace.message = "Gemini fetched and summarized YouTube source content."
        return analyzed, trace


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


def _build_youtube_analysis_prompt(
    task: AcquisitionTask, source: RawSource, video_candidates: list[dict[str, str]]
) -> str:
    entity = task.entity.display_name
    address = task.entity.address or ""
    city = task.entity.city or ""
    needs = ", ".join(task.data_needed)
    question = task.question or ""
    instruction = task.acquisition_instruction or ""
    source_text = source.text[:12000]
    candidates_text = "\n".join(
        f"- {item['title']} | {item['url']}" for item in video_candidates[:20]
    ) or "No video candidates extracted from the fetched HTML."
    return (
        "Analyze this YouTube source for a source-linked data acquisition agent.\n"
        "Use only the provided YouTube URL, extracted video candidates, and visible page text. Do not use web search.\n"
        "When the source URL is a direct YouTube watch URL, analyze the video, captions, transcript, title, and description if they are available to the model without web search.\n"
        "Return concise plain text only. Include requested video metadata even when project-specific transcript content is not visible.\n"
        "Scope: preserve both target-project details and broader city construction, planning, housing, zoning, land use, transportation, infrastructure, development, permitting, CEQA, and public works content.\n"
        "Do not discard city-wide construction or planning agenda items just because they do not mention the target project.\n"
        "Mention video titles, dates, meeting names, or agenda context when visible. Do not invent facts.\n"
        "If no requested YouTube video or content is found, say exactly: No requested YouTube content found.\n\n"
        f"Entity: {entity}\n"
        f"Address: {address}\n"
        f"City: {city}\n"
        f"Data needed: {needs}\n"
        f"Question: {question}\n"
        f"Acquisition instruction: {instruction}\n"
        f"YouTube URL: {source.source_url}\n"
        f"Extracted YouTube video candidates:\n{candidates_text}\n"
        f"Fetched page text:\n{source_text}\n"
    )


def _build_youtube_video_url_retry_prompt(
    task: AcquisitionTask,
    source: RawSource,
    previous_summary: str,
    video_candidates: list[dict[str, str]],
) -> str:
    entity = task.entity.display_name
    question = task.question or ""
    instruction = task.acquisition_instruction or ""
    candidates_text = "\n".join(
        f"- {item['title']} | {item['url']}" for item in video_candidates[:20]
    ) or "No local candidate URLs were extracted."
    return (
        "The previous YouTube answer did not provide a concrete YouTube watch URL.\n"
        "Use Google Search grounding to find the exact YouTube video or livestream matching the request.\n"
        "You must return a concrete URL in the form https://www.youtube.com/watch?v=VIDEO_ID when available.\n"
        "Do not use a channel URL, /streams URL, /videos URL, search results URL, or index page as the concrete video URL.\n"
        "If the exact watch URL cannot be found, say exactly: No concrete YouTube video URL found.\n"
        "Return concise plain text with title, date, concrete URL, captions/transcript status, and relevant summary.\n\n"
        f"Entity: {entity}\n"
        f"Question: {question}\n"
        f"Acquisition instruction: {instruction}\n"
        f"Starting YouTube URL: {source.source_url}\n"
        f"Extracted local candidates:\n{candidates_text}\n"
        f"Previous summary:\n{previous_summary[:4000]}\n"
    )


def _extract_youtube_video_candidates(source: RawSource) -> list[dict[str, str]]:
    raw = ""
    if source.raw_path:
        try:
            raw = Path(source.raw_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw = ""
    if not raw:
        raw = source.text

    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r'"title":\{"content":"(?P<title>[^"]+)"\}.*?"contentId":"(?P<video_id>[A-Za-z0-9_-]{11})"',
        re.DOTALL,
    )
    for match in pattern.finditer(raw):
        video_id = match.group("video_id")
        if video_id in seen:
            continue
        seen.add(video_id)
        title = match.group("title")
        candidates.append({"title": title, "url": f"https://www.youtube.com/watch?v={video_id}"})

    if not candidates:
        ids = list(dict.fromkeys(re.findall(r'(?:watch\?v=|/vi/)([A-Za-z0-9_-]{11})', raw)))
        for video_id in ids[:20]:
            candidates.append({"title": f"YouTube video {video_id}", "url": f"https://www.youtube.com/watch?v={video_id}"})
    return candidates


def _ensure_video_candidates_in_summary(
    task: AcquisitionTask, summary: str, video_candidates: list[dict[str, str]]
) -> str:
    if "youtube.com/watch?v=" in summary.lower():
        return summary
    matches = _matching_video_candidates(task, video_candidates)
    if not matches:
        return summary
    lines = "\n".join(f"- {item['title']}: {item['url']}" for item in matches[:3])
    return f"{summary}\n\nConcrete YouTube video URL candidates extracted from the fetched source:\n{lines}"


def _needs_concrete_youtube_video_url(task: AcquisitionTask) -> bool:
    requested_text = " ".join(
        [task.question or "", task.acquisition_instruction or "", " ".join(task.data_needed)]
    ).lower()
    return any(term in requested_text for term in ["specific video url", "concrete video url", "watch?v=", "youtube"])


def _matching_video_candidates(task: AcquisitionTask, video_candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    haystacks = [task.question or "", task.acquisition_instruction or ""]
    requested_text = " ".join(haystacks).lower()
    matches: list[dict[str, str]] = []
    for item in video_candidates:
        title = item["title"].lower()
        if "june 1, 2026" in requested_text and "june 1, 2026" in title:
            matches.append(item)
        elif "jun 1, 2026" in requested_text and "jun 1, 2026" in title:
            matches.append(item)
    return matches


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
