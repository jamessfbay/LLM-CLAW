from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery, ProviderTrace
from llm_claw.providers.aisa_client import AIsaClient


class AIsaTwitterProvider:
    """Find public X pain authors through AIsa's licensed Twitter API gateway."""

    name = "aisa_twitter"
    endpoint = "/apis/v1/twitter/tweet/advanced_search"

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
        search_query = _twitter_query(query.text)
        if not search_query:
            return [], ProviderTrace(provider=self.name, status="error", query=query.text, message="Empty X query")
        try:
            payload = self.client.get(self.endpoint, {"query": search_query, "queryType": "Latest"})
        except Exception as exc:
            return [], ProviderTrace(
                provider=self.name,
                status="error",
                query=query.text,
                message=f"AIsa X request failed: {type(exc).__name__}",
            )

        retrieved_at = datetime.now(timezone.utc).isoformat()
        request_id = _request_id(payload)
        candidates: list[CandidateSource] = []
        for tweet in _tweet_items(payload):
            normalized = _normalize_tweet(tweet)
            if not normalized or not _is_pain_or_solution_signal(normalized["text"], search_query):
                continue
            metadata: dict[str, Any] = {
                "platform": "x",
                "public_handle": f"@{normalized['username']}",
                "profile_url": f"https://x.com/{normalized['username']}",
                "identity_kind": "public_handle",
                "source_post_url": normalized["url"],
                "source_provider": "aisa_twitter",
                "source_label": "AISA Twitter API",
                "source_method": "platform_api",
                "retrieved_at": retrieved_at,
            }
            if normalized.get("person_name"):
                metadata["person_name"] = normalized["person_name"]
            if normalized.get("published_at"):
                metadata["published_at"] = normalized["published_at"]
            if request_id:
                metadata["source_request_id"] = request_id
            candidates.append(CandidateSource(
                query_id=query.id,
                provider=self.name,
                title=f"@{normalized['username']}: {normalized['text'][:180]}",
                url=normalized["url"],
                snippet=normalized["text"][:1000],
                publisher="X",
                source_type="candidate",
                confidence=0.68,
                is_official=False,
                analysis_note=(
                    "Public post returned by AIsa's Twitter API. Identity and post text remain bound to the original X URL."
                ),
                metadata=metadata,
            ))
            if len(candidates) >= min(task.source_policy.max_sources, 15):
                break
        return candidates, ProviderTrace(
            provider=self.name,
            query=query.text,
            candidate_count=len(candidates),
            message=f"AIsa X returned {len(candidates)} public post candidate(s).",
        )


def _twitter_query(text: str) -> str:
    value = re.sub(r"site:(?:www\.)?(?:x|twitter)\.com\s*", "", text, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""
    quoted = next((item.strip() for item in re.findall(r'"([^"]+)"', value) if item.strip()), "")
    if quoted:
        topic = quoted[:100]
    else:
        generic = {
            "alternative", "buyer", "buyers", "cost", "economic", "for", "founder", "head", "leader",
            "looking", "manual", "named", "owner", "pain", "problem", "public", "seeker", "seekers",
            "solution", "tool", "workflow", "workaround",
        }
        tokens = [
            token for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", value)
            if token.lower() not in generic
        ]
        topic = " ".join(tokens[:5])[:100].strip()
    if not topic:
        return ""
    since = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
    lowered_value = value.lower()
    if "looking for" in lowered_value or "tool" in lowered_value:
        intent = '"looking for"'
    elif "manual" in lowered_value:
        intent = "manual"
    else:
        intent = "workaround"
    return f'"{topic}" {intent} since:{since} lang:en -filter:retweets'


def _is_pain_or_solution_signal(text: str, search_query: str) -> bool:
    lowered = re.sub(r"\s+", " ", text).lower()
    quoted = re.findall(r'"([^"]+)"', search_query)
    topic = quoted[0].lower() if quoted else ""
    common_domain = re.search(
        r"\b(?:ai|coding|software|sales|support|research|autonomous|developer)\s+agents?\b",
        topic,
    )
    matching_topic = common_domain.group(0) if common_domain else topic
    topic_tokens = [token for token in re.findall(r"[a-z0-9]+", matching_topic) if len(token) > 1]
    if topic_tokens and not all(token in lowered for token in topic_tokens):
        return False
    promotional_patterns = (
        r"\bwe(?:'re| are) building\b", r"\bwe(?:'re| are) launching\b", r"\bjust launched\b",
        r"\braised \$", r"\blooking for contributors\b", r"\bsign up\b", r"\bjoin (?:us|the waitlist)\b",
        r"<p class=", r"\bwp-block-",
    )
    if any(re.search(pattern, lowered) for pattern in promotional_patterns):
        return False
    direct_seeking_patterns = (
        r"\blooking for (?:a |an |some )?(?:tool|solution|alternative|way)\b",
        r"\bany (?:good )?(?:tool|solution|alternative|recommendation)\b",
        r"\b(?:does|can) anyone (?:know|recommend|help|suggest)\b",
        r"\bwhat (?:tool|solution|alternative) (?:do|are) (?:you|people|teams)\b",
    )
    if any(re.search(pattern, lowered) for pattern in direct_seeking_patterns):
        return True
    first_person_patterns = (
        r"\b(?:i|we) (?:still |currently )?(?:need|struggl\w*|have to|can't|cannot|hate|spend|waste|manually)\b",
        r"\b(?:my|our) team (?:still )?(?:needs|struggl\w*|has to|hates|spends|wastes)\b",
        r"\b(?:i|we) (?:still |currently )?\w+(?: \w+){0,6} (?:manually|by hand)\b",
        r"\b(?:my|our) (?:problem|issue|workaround|workflow)\b",
    )
    return any(re.search(pattern, lowered) for pattern in first_person_patterns)


def _tweet_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        if _looks_like_tweet(value):
            found.append(value)
            return
        for child in value.values():
            visit(child)

    visit(payload)
    unique: dict[str, dict[str, Any]] = {}
    for item in found:
        identity = str(item.get("rest_id") or item.get("id_str") or item.get("id") or id(item))
        unique[identity] = item
    return list(unique.values())


def _looks_like_tweet(value: dict[str, Any]) -> bool:
    legacy = value.get("legacy") if isinstance(value.get("legacy"), dict) else {}
    has_id = bool(value.get("rest_id") or value.get("id_str") or value.get("id") or legacy.get("id_str"))
    has_text = bool(value.get("full_text") or value.get("text") or legacy.get("full_text"))
    return has_id and has_text


def _normalize_tweet(tweet: dict[str, Any]) -> dict[str, str] | None:
    legacy = tweet.get("legacy") if isinstance(tweet.get("legacy"), dict) else {}
    text = str(tweet.get("full_text") or tweet.get("text") or legacy.get("full_text") or "").strip()
    tweet_id = str(tweet.get("rest_id") or tweet.get("id_str") or tweet.get("id") or legacy.get("id_str") or "").strip()
    user = _tweet_user(tweet)
    user_legacy = user.get("legacy") if isinstance(user.get("legacy"), dict) else {}
    username = str(
        user.get("screen_name") or user.get("username") or user.get("userName") or user_legacy.get("screen_name")
        or tweet.get("screen_name") or tweet.get("username") or tweet.get("userName") or ""
    ).strip().removeprefix("@")
    if not text or not tweet_id or not re.fullmatch(r"[A-Za-z0-9_]{1,30}", username):
        return None
    raw_url = str(tweet.get("url") or tweet.get("tweet_url") or "").strip()
    url = raw_url if _valid_x_post_url(raw_url) else f"https://x.com/{username}/status/{tweet_id}"
    published = _published_at(tweet.get("created_at") or tweet.get("createdAt") or legacy.get("created_at"))
    person_name = str(user.get("name") or user_legacy.get("name") or "").strip()
    result = {"text": text, "url": url, "username": username}
    if published:
        result["published_at"] = published
    if person_name:
        result["person_name"] = person_name[:160]
    return result


def _published_at(value: Any) -> str:
    if value in {None, ""}:
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    try:
        return datetime.strptime(text, "%a %b %d %H:%M:%S %z %Y").astimezone(timezone.utc).isoformat()
    except ValueError:
        return ""


def _tweet_user(tweet: dict[str, Any]) -> dict[str, Any]:
    for candidate in (tweet.get("author"), tweet.get("user"), tweet.get("user_info")):
        if isinstance(candidate, dict):
            return candidate
    core = tweet.get("core") if isinstance(tweet.get("core"), dict) else {}
    user_results = core.get("user_results") if isinstance(core.get("user_results"), dict) else {}
    result = user_results.get("result")
    return result if isinstance(result, dict) else {}


def _valid_x_post_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in {
        "x.com", "www.x.com", "twitter.com", "www.twitter.com",
    } and "/status/" in parsed.path


def _request_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return str(payload.get("request_id") or payload.get("requestId") or metadata.get("request_id") or "")[:160]
