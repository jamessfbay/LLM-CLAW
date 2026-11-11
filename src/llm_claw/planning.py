from __future__ import annotations

import re

from llm_claw.models import AcquisitionTask, PlannedQuery


class DataNeedPlanner:
    def plan(self, task: AcquisitionTask) -> list[str]:
        return task.data_needed[:]


class QueryPlanner:
    def plan(self, task: AcquisitionTask, data_needs: list[str]) -> list[PlannedQuery]:
        if task.domain == "commercial_discovery" and task.task_type == "candidate_buyer_discovery":
            return _candidate_buyer_queries(task)
        if task.domain == "commercial_discovery":
            return [
                PlannedQuery(text=need, data_need=need, freshness=task.freshness)
                for need in data_needs
                if need.strip()
            ][:20]

        entity = task.entity.display_name
        city = task.entity.city
        address = task.entity.address
        queries: list[PlannedQuery] = []
        for need in data_needs:
            terms = [entity, need]
            if city:
                terms.append(city)
            if address:
                terms.append(address)
            if task.source_policy.prefer_official_sources:
                terms.append("site:.gov OR city official")
            queries.append(PlannedQuery(text=" ".join(terms), data_need=need, freshness=task.freshness))

        if task.domain == "real_estate":
            base_terms = [part for part in [entity, city, address] if part]
            extras = ["planning commission agenda", "staff report PDF", "CEQA notice", "public comments"]
            for extra in extras:
                if len(queries) >= 10:
                    break
                queries.append(PlannedQuery(text=" ".join(base_terms + [extra]), data_need=extra, freshness=task.freshness))
        return queries[:10]


def _candidate_buyer_queries(task: AcquisitionTask) -> list[PlannedQuery]:
    """Build identity-aware searches instead of sending generic research needs to every provider."""
    topic = str(task.entity.metadata.get("topic") or task.question or task.entity.display_name).strip()
    topic = " ".join(topic.split())[:240]
    primary, secondary = _candidate_search_anchors(task, topic)
    exact = f'"{primary}"'
    broad = secondary or primary
    query_specs = [
        (f"{exact} problem workaround looking for tool", "public pain and active solution search"),
        (f"{broad} named leader platform engineering security governance", "named public economic buyers"),
        (f"site:reddit.com {exact} problem workaround", "Reddit pain authors"),
        (f"site:reddit.com {exact} looking for tool alternative", "Reddit solution seekers"),
        (f"site:reddit.com {broad} manual workflow cost", "Reddit concrete workflow"),
        (f"site:github.com/issues {exact} problem workaround", "GitHub issue authors"),
        (f"site:github.com/issues {broad} security approval audit", "GitHub governance issue authors"),
        (f"site:github.com/issues {exact} looking for alternative", "GitHub solution seekers"),
        (f"site:news.ycombinator.com {exact} problem workaround", "Hacker News pain authors"),
        (f"site:news.ycombinator.com {exact} ask alternative tool", "Hacker News solution seekers"),
        (f"site:news.ycombinator.com {broad} workflow cost", "Hacker News concrete workflow"),
        (f"site:stackoverflow.com {exact} problem workaround", "Stack Overflow question authors"),
        (f"site:stackoverflow.com {broad} security permission", "Stack Overflow governance questions"),
        (f"site:stackoverflow.com {exact} tool alternative", "Stack Overflow solution seekers"),
    ]
    return [PlannedQuery(text=text, data_need=need, freshness=task.freshness) for text, need in query_specs]


def _candidate_search_anchors(task: AcquisitionTask, topic: str) -> tuple[str, str]:
    context = " ".join(
        str(task.entity.metadata.get(key) or "")
        for key in ("topic", "problem", "buyer", "economic_buyer")
    )
    common_domain = re.search(
        r"\b(?:AI|coding|software|sales|support|research|autonomous|developer)\s+agents?\b",
        context,
        flags=re.IGNORECASE,
    )
    primary = common_domain.group(0) if common_domain else ""
    stopwords = {
        "about", "after", "against", "behind", "between", "current", "from", "into", "more", "that",
        "their", "this", "through", "under", "using", "with", "workflow", "workflows", "solution", "system",
    }
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", f"{topic} {context}")
        if token.lower() not in stopwords
    ]
    distinctive = " ".join(dict.fromkeys(tokens))[:100].strip()
    if not primary:
        primary = " ".join(distinctive.split()[:4]) or topic[:80]
    secondary_tokens = [token for token in distinctive.split() if token.lower() not in primary.lower().split()]
    secondary = " ".join(([primary] + secondary_tokens[:4])).strip()
    return primary[:80], secondary[:120]
