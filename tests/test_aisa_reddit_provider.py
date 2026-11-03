from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, PlannedQuery
from llm_claw.providers.aisa_reddit import AIsaRedditProvider, _post_url, _reddit_query


def test_aisa_reddit_normalizes_only_original_public_post_urls(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, aisa_api_key="test-key")
    provider = AIsaRedditProvider(settings)
    searches: list[str] = []

    def fake_search(search_query: str) -> dict:  # type: ignore[type-arg]
        searches.append(search_query)
        return {
            "success": True,
            "posts": [
                {"title": "Teachers describe review pain", "permalink": "/r/Teachers/comments/123/review_pain/", "selftext": "Manual review takes hours."},
                {"title": "Unrelated external link", "url": "https://other.example/article"},
                {"title": "Unsafe host", "url": "https://reddit.com.evil.example/r/test/comments/9"},
            ],
        }

    provider._search = fake_search  # type: ignore[method-assign]
    task = AcquisitionTask.model_validate({
        "domain": "commercial_discovery",
        "entity": {"name": "AI for Education"},
        "data_needed": ["site:reddit.com AI for Education pain"],
    })
    query = PlannedQuery(text='site:reddit.com "AI for Education" pain (prefer last 30 days; extend to 90 days only if evidence is insufficient)', data_need="reddit")

    candidates, trace = provider.discover(task, query)

    assert len(candidates) == 1
    assert candidates[0].url == "https://www.reddit.com/r/Teachers/comments/123/review_pain/"
    assert candidates[0].provider == "aisa_reddit"
    assert candidates[0].source_type == "candidate"
    assert trace.candidate_count == 1
    assert _reddit_query(query.text) == "AI for Education"
    assert searches == ["AI for Education"]


def test_aisa_reddit_rejects_bad_response_without_exposing_key(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, aisa_api_key="secret-test-key")
    provider = AIsaRedditProvider(settings)
    provider._search = lambda query: {"success": False, "error": "secret-test-key"}  # type: ignore[method-assign]
    task = AcquisitionTask.model_validate({"entity": {"name": "AI for Education"}, "data_needed": ["reddit"]})
    query = PlannedQuery(text="site:reddit.com AI for Education", data_need="reddit")

    candidates, trace = provider.discover(task, query)

    assert candidates == []
    assert trace.status == "error"
    assert "secret-test-key" not in (trace.message or "")
    assert _post_url({"permalink": "https://reddit.com.evil.example/r/test/comments/9"}) is None
