from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace


class HackerNewsProvider:
    """Discover public HN story and comment authors through the Algolia HN index."""

    name = "hacker_news"
    endpoint = "https://hn.algolia.com/api/v1/search_by_date"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return True

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        search_query = _hn_query(query.text)
        if not search_query:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message="Empty Hacker News query")
        try:
            payload = self._search(search_query)
            hits = payload.get("hits")
            if not isinstance(hits, list):
                raise ValueError("Hacker News response did not contain a hits list")
        except HTTPError as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"Hacker News HTTP {exc.code}")
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"Hacker News request failed: {type(exc).__name__}")

        candidates = [candidate for hit in hits if (candidate := _candidate(hit, query.id))]
        candidates = candidates[: min(task.source_policy.max_sources, 10)]
        return candidates, ProviderTrace(
            provider=self.name,
            query=query.text,
            candidate_count=len(candidates),
            message=f"Hacker News returned {len(candidates)} public author candidate(s).",
        )

    def _search(self, query: str) -> dict[str, Any]:
        params = urlencode({"query": query, "tags": "(story,comment)", "hitsPerPage": 10})
        request = Request(f"{self.endpoint}?{params}", headers={"User-Agent": self.settings.source_user_agent})
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Hacker News response is not an object")
        return payload


def _hn_query(text: str) -> str:
    value = re.sub(r"site:(?:news\.)?ycombinator\.com\s*", "", text, flags=re.IGNORECASE)
    return " ".join(value.split()).strip()


def _candidate(hit: Any, query_id: str) -> CandidateSource | None:
    if not isinstance(hit, dict):
        return None
    item_id = str(hit.get("objectID") or "").strip()
    author = str(hit.get("author") or "").strip()
    if not item_id.isdigit() or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", author):
        return None
    url = f"https://news.ycombinator.com/item?id={item_id}"
    profile_url = f"https://news.ycombinator.com/user?id={author}"
    text = str(hit.get("comment_text") or hit.get("story_text") or "")
    snippet = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text))).strip()
    published_at = _published_at(hit)
    metadata = {
        "platform": "hacker_news",
        "public_handle": author,
        "profile_url": profile_url,
        "identity_kind": "public_handle",
        "source_post_url": url,
        "community": "Hacker News",
    }
    if published_at:
        metadata["published_at"] = published_at
    return CandidateSource(
        query_id=query_id,
        provider="hacker_news",
        title=str(hit.get("title") or hit.get("story_title") or "Hacker News discussion")[:300],
        url=url,
        snippet=snippet[:700],
        publisher="Hacker News",
        confidence=0.6,
        analysis_note="Public Hacker News author; the original item must become evidence before candidate projection.",
        metadata=metadata,
    )


def _published_at(hit: dict[str, Any]) -> str | None:
    try:
        if hit.get("created_at_i") is not None:
            return datetime.fromtimestamp(float(hit["created_at_i"]), timezone.utc).isoformat()
        return datetime.fromisoformat(str(hit.get("created_at") or "").replace("Z", "+00:00")).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None
