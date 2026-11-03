from __future__ import annotations

import ipaddress
import re
from datetime import date
from urllib.parse import urlparse

from llm_claw.config import Settings
from llm_claw.models import CandidateSource, ProviderTrace, utc_now
from llm_claw.providers.aisa_client import AIsaClient


class AIsaReranker:
    def __init__(self, settings: Settings) -> None:
        self.client = AIsaClient(settings)

    def rank(self, question: str, candidates: list[CandidateSource]) -> tuple[list[CandidateSource], ProviderTrace]:
        try:
            payload = self.client.post("/v1/rerank", {
                "model": "jina-reranker-v3",
                "query": question,
                "documents": [f"{item.title}\n{item.snippet[:300]}" for item in candidates],
                "top_n": len(candidates),
            })
            rows = payload.get("results")
            if not isinstance(rows, list):
                raise ValueError("Rerank response has no results list")
            order: list[int] = []
            for row in rows:
                index = row.get("index") if isinstance(row, dict) else None
                if isinstance(index, int) and 0 <= index < len(candidates) and index not in order:
                    order.append(index)
            if not order:
                raise ValueError("Rerank response has no valid indices")
            order.extend(index for index in range(len(candidates)) if index not in order)
            return [candidates[index] for index in order], ProviderTrace(
                provider="aisa_jina_rerank", candidate_count=len(candidates), message=f"Reranked {len(candidates)} candidates"
            )
        except Exception as exc:
            return candidates, ProviderTrace(provider="aisa_jina_rerank", status="error", message=f"Rerank failed: {type(exc).__name__}; original order retained")


class AIsaSimilarweb:
    def __init__(self, settings: Settings) -> None:
        self.client = AIsaClient(settings)

    def traffic(self, domains: list[str], limit: int) -> tuple[list[dict[str, object]], list[ProviderTrace]]:
        snapshots: list[dict[str, object]] = []
        traces: list[ProviderTrace] = []
        for domain in domains[:limit]:
            normalized = _valid_domain(domain)
            if not normalized:
                continue
            try:
                period = _previous_month()
                response = self.client.get("/apis/v1/similarweb/website/traffic-engagement", {
                    "domain": normalized, "start_date": period, "end_date": period,
                    "metrics": "visits,pages_per_visit,average_visit_duration,bounce_rate",
                    "country": "ww", "web_source": "total",
                })
                metrics = _compact_metrics(response)
                if not metrics:
                    raise ValueError("Similarweb response has no supported numeric metrics")
                snapshots.append({"domain": normalized, "observed_at": utc_now().isoformat(), "period": period, "source": "AIsa Similarweb traffic estimate", "metrics": metrics})
                traces.append(ProviderTrace(provider="aisa_similarweb", message=f"Traffic snapshot received for {normalized}"))
            except Exception as exc:
                traces.append(ProviderTrace(provider="aisa_similarweb", status="error", message=f"Traffic snapshot failed for {normalized}: {type(exc).__name__}"))
        return snapshots, traces


def _valid_domain(value: str) -> str | None:
    value = value.strip().lower()
    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.path not in {"", "/"}:
            return None
        value = parsed.hostname or ""
    if len(value) > 253 or "." not in value or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in value.split(".")):
        return None
    if value.endswith((".local", ".internal", ".test", ".invalid")) or value in {"localhost", "reddit.com", "github.com"}:
        return None
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return value
    return None


def _compact_metrics(response: dict[str, object]) -> dict[str, object]:
    data = response.get("data")
    if isinstance(data, list):
        data = next((item for item in reversed(data) if isinstance(item, dict)), {})
    if data is None:
        data = response
    if not isinstance(data, dict):
        return {}
    allowed = {"visits", "bounce_rate", "pages_per_visit", "average_visit_duration", "date"}
    return {key: value for key, value in data.items() if key in allowed and isinstance(value, (str, int, float)) and not isinstance(value, bool)}


def _previous_month() -> str:
    first = date.today().replace(day=1)
    previous = date.fromordinal(first.toordinal() - 1)
    return previous.strftime("%Y-%m")
