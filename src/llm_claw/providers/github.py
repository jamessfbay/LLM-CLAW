from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace


class GitHubProvider:
    """Find public issue authors whose original posts describe a concrete workflow problem."""

    name = "github"
    endpoint = "https://api.github.com/search/issues"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return True

    def discover(self, task: AcquisitionTask, query: PlannedQuery) -> tuple[list[CandidateSource], ProviderTrace]:
        search_query = _github_query(query.text)
        if not search_query:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message="Empty GitHub query")
        try:
            payload = self._search(search_query)
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError("GitHub response did not contain an items list")
        except HTTPError as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"GitHub HTTP {exc.code}")
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message=f"GitHub request failed: {type(exc).__name__}")

        candidates: list[CandidateSource] = []
        for item in items:
            candidate = _candidate(item, query.id)
            if candidate:
                candidates.append(candidate)
            if len(candidates) >= min(task.source_policy.max_sources, 10):
                break
        return candidates, ProviderTrace(
            provider=self.name,
            query=query.text,
            candidate_count=len(candidates),
            message=f"GitHub returned {len(candidates)} public issue author candidate(s).",
        )

    def _search(self, query: str) -> dict[str, Any]:
        params = urlencode({"q": f"{query} is:issue", "sort": "created", "order": "desc", "per_page": 10})
        headers = {"Accept": "application/vnd.github+json", "User-Agent": self.settings.source_user_agent}
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        request = Request(f"{self.endpoint}?{params}", headers=headers)
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("GitHub response is not an object")
        return payload


def _github_query(text: str) -> str:
    value = re.sub(r"site:(?:www\.)?github\.com(?:/[^\s]*)?\s*", "", text, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:issues?|discussions?)\b", "", value, flags=re.IGNORECASE)
    return " ".join(value.split()).strip()


def _candidate(item: Any, query_id: str) -> CandidateSource | None:
    if not isinstance(item, dict):
        return None
    url = str(item.get("html_url") or "").strip()
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    handle = str(user.get("login") or "").strip()
    profile_url = str(user.get("html_url") or "").strip()
    if not _public_github_url(url) or not handle or not _public_github_profile(profile_url):
        return None
    published_at = _iso_time(item.get("created_at"))
    metadata = {
        "platform": "github",
        "public_handle": handle,
        "profile_url": profile_url,
        "identity_kind": "public_handle",
        "source_post_url": url,
    }
    if published_at:
        metadata["published_at"] = published_at
    return CandidateSource(
        query_id=query_id,
        provider="github",
        title=str(item.get("title") or url)[:300],
        url=url,
        snippet=str(item.get("body") or "")[:700],
        publisher="GitHub",
        confidence=0.65,
        analysis_note="Public GitHub issue author; the original issue must become evidence before candidate projection.",
        metadata=metadata,
    )


def _public_github_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and parsed.hostname == "github.com" and "/issues/" in parsed.path


def _public_github_profile(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and parsed.hostname == "github.com" and len(parsed.path.strip("/").split("/")) == 1


def _iso_time(value: Any) -> str | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()
    except (TypeError, ValueError):
        return None
