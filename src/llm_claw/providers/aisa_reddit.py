from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace


class AIsaRedditProvider:
    """Discover public Reddit posts; CLAW still fetches the original source."""

    name = "aisa_reddit"
    endpoint = "https://api.aisa.one/apis/v1/reddit/search"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
        search_query = _reddit_query(query.text)
        if not search_query:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message="Empty Reddit query")
        try:
            payload = self._search(search_query)
            posts = payload.get("posts")
            if payload.get("success") is not True or not isinstance(posts, list):
                raise ValueError("AIsa Reddit response did not contain a successful posts list")
        except HTTPError as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"AIsa Reddit HTTP {exc.code}")
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"AIsa Reddit request failed: {type(exc).__name__}")

        candidates: list[CandidateSource] = []
        for post in posts:
            if not isinstance(post, dict):
                continue
            url = _post_url(post)
            if not url:
                continue
            candidates.append(CandidateSource(
                query_id=query.id,
                provider=self.name,
                title=str(post.get("title") or url)[:300],
                url=url,
                snippet=str(post.get("selftext") or "")[:500],
                publisher="Reddit",
                source_type="candidate",
                confidence=0.6,
                is_official=False,
                analysis_note="Discovered through AIsa Reddit search; original post must be fetched before it becomes evidence.",
            ))
            if len(candidates) >= min(task.source_policy.max_sources, 10):
                break
        return candidates, ProviderTrace(
            provider=self.name,
            query=query.text,
            candidate_count=len(candidates),
            message=f"AIsa Reddit returned {len(candidates)} public post candidate(s).",
        )

    def _search(self, query: str) -> dict[str, Any]:
        url = f"{self.endpoint}?{urlencode({'query': query, 'limit': 10})}"
        request = Request(url, headers={"Authorization": f"Bearer {self.settings.aisa_api_key}"})
        with urlopen(request, timeout=self.settings.aisa_request_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("AIsa Reddit response is not an object")
        return payload


def _reddit_query(text: str) -> str:
    without_site = re.sub(r"site:(?:www\.)?reddit\.com\s*", "", text, flags=re.IGNORECASE)
    without_window = re.sub(r"\s*\(prefer last 30 days; extend to 90 days only if evidence is insufficient\)\s*$", "", without_site, flags=re.IGNORECASE)
    market_phrase = re.search(r'"([^"\n]{3,120})"', without_window)
    if market_phrase:
        return market_phrase.group(1).strip()
    return without_window.strip()


def _post_url(post: dict[str, Any]) -> str | None:
    raw = str(post.get("permalink") or post.get("url") or "").strip()
    if raw.startswith("/r/"):
        raw = f"https://www.reddit.com{raw}"
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in {"reddit.com", "www.reddit.com", "old.reddit.com"}:
        return None
    if not parsed.path.startswith("/r/"):
        return None
    return raw
