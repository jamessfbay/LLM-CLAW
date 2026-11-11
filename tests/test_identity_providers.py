from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, PlannedQuery
from llm_claw.providers.github import GitHubProvider
from llm_claw.providers.hacker_news import HackerNewsProvider
from llm_claw.providers.stack_exchange import StackExchangeProvider


def _task(provider: str) -> AcquisitionTask:
    return AcquisitionTask.model_validate({
        "domain": "commercial_discovery",
        "task_type": "candidate_buyer_discovery",
        "entity": {"name": "Coding agent governance"},
        "data_needed": ["public pain authors"],
        "provider_policy": {"allowed_providers": [provider, "crawler"]},
    })


def test_github_preserves_public_issue_author_identity(tmp_path: Path) -> None:
    provider = GitHubProvider(Settings(workspace=tmp_path))
    provider._search = lambda _query: {"items": [{
        "title": "Agent approvals are hard to audit",
        "html_url": "https://github.com/acme/agent/issues/42",
        "body": "We need a traceable approval workflow.",
        "created_at": "2026-11-01T00:00:00Z",
        "user": {"login": "operator-one", "html_url": "https://github.com/operator-one"},
    }]}  # type: ignore[method-assign]

    candidates, trace = provider.discover(
        _task("github"), PlannedQuery(text='site:github.com/issues "Coding agent governance" audit', data_need="authors"),
    )

    assert trace.candidate_count == 1
    assert candidates[0].metadata["public_handle"] == "operator-one"
    assert candidates[0].metadata["profile_url"] == "https://github.com/operator-one"


def test_hacker_news_preserves_public_comment_author_identity(tmp_path: Path) -> None:
    provider = HackerNewsProvider(Settings(workspace=tmp_path))
    provider._search = lambda _query: {"hits": [{
        "objectID": "12345",
        "author": "founder42",
        "story_title": "Coding agents in production",
        "comment_text": "Our approval process is still manual.",
        "created_at_i": 1_762_000_000,
    }]}  # type: ignore[method-assign]

    candidates, trace = provider.discover(
        _task("hacker_news"), PlannedQuery(text='site:news.ycombinator.com "Coding agent governance"', data_need="authors"),
    )

    assert trace.candidate_count == 1
    assert candidates[0].metadata["public_handle"] == "founder42"
    assert candidates[0].url == "https://news.ycombinator.com/item?id=12345"


def test_stack_exchange_preserves_public_question_author_identity(tmp_path: Path) -> None:
    provider = StackExchangeProvider(Settings(workspace=tmp_path))
    provider._search = lambda _query: {"items": [{
        "title": "How can I audit an agent action?",
        "link": "https://stackoverflow.com/questions/123/how-can-i-audit-an-agent-action",
        "body": "Our current workaround requires manual log review.",
        "creation_date": 1_762_000_000,
        "owner": {"display_name": "Dev Operator", "link": "https://stackoverflow.com/users/99/dev-operator"},
    }]}  # type: ignore[method-assign]

    candidates, trace = provider.discover(
        _task("stack_exchange"), PlannedQuery(text='site:stackoverflow.com "Coding agent governance"', data_need="authors"),
    )

    assert trace.candidate_count == 1
    assert candidates[0].metadata["public_handle"] == "Dev Operator"
    assert candidates[0].metadata["profile_url"].startswith("https://stackoverflow.com/users/")
