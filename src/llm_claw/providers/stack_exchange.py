from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace


class StackExchangeProvider:
    """Find public Stack Overflow question authors with traceable profile identity."""

    name = "stack_exchange"
    endpoint = "https://api.stackexchange.com/2.3/search/advanced"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return True

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        search_query = _stack_query(query.text)
        if not search_query:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message="Empty Stack Exchange query")
        try:
            payload = self._search(search_query)
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError("Stack Exchange response did not contain an items list")
        except HTTPError as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"Stack Exchange HTTP {exc.code}")
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"Stack Exchange request failed: {type(exc).__name__}")

        candidates = [candidate for item in items if (candidate := _candidate(item, query.id))]
        candidates = candidates[: min(task.source_policy.max_sources, 10)]
        return candidates, ProviderTrace(
            provider=self.name,
            query=query.text,
            candidate_count=len(candidates),
            message=f"Stack Exchange returned {len(candidates)} public question author candidate(s).",
        )

    def _search(self, query: str) -> dict[str, Any]:
        params = urlencode({
            "site": "stackoverflow",
            "order": "desc",
            "sort": "creation",
            "q": query,
            "pagesize": 10,
            "filter": "withbody",
        })
        request = Request(f"{self.endpoint}?{params}", headers={"User-Agent": self.settings.source_user_agent})
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Stack Exchange response is not an object")
        return payload


def _stack_query(text: str) -> str:
    value = re.sub(r"site:(?:www\.)?(?:stackoverflow\.com|stackexchange\.com)\s*", "", text, flags=re.IGNORECASE)
    return " ".join(value.split()).strip()


def _candidate(item: Any, query_id: str) -> CandidateSource | None:
    if not isinstance(item, dict):
        return None
    owner = item.get("owner") if isinstance(item.get("owner"), dict) else {}
    handle = str(owner.get("display_name") or "").strip()
    profile_url = str(owner.get("link") or "").strip()
    url = str(item.get("link") or "").strip()
    if not handle or not _public_stack_url(url) or not _public_stack_url(profile_url):
        return None
    body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(item.get("body") or "")))).strip()
    published_at = _published_at(item.get("creation_date"))
    metadata = {
        "platform": "stack_overflow",
        "public_handle": handle,
        "profile_url": profile_url,
        "identity_kind": "public_handle",
        "source_post_url": url,
        "community": "Stack Overflow",
    }
    if published_at:
        metadata["published_at"] = published_at
    return CandidateSource(
        query_id=query_id,
        provider="stack_exchange",
        title=html.unescape(str(item.get("title") or url))[:300],
        url=url,
        snippet=body[:700],
        publisher="Stack Overflow",
        confidence=0.65,
        analysis_note="Public Stack Overflow question author; the original question must become evidence before candidate projection.",
        metadata=metadata,
    )


def _public_stack_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in {
        "stackoverflow.com", "www.stackoverflow.com", "stackexchange.com", "www.stackexchange.com",
    }


def _published_at(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None
