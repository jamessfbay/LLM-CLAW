from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, PlannedQuery
from llm_claw.pipeline import source_fetcher
from llm_claw.pipeline.source_fetcher import SourceFetcher
from llm_claw.providers.aisa_enrichment import AIsaReranker, AIsaSimilarweb, _valid_domain
from llm_claw.providers.aisa_scholar import AIsaScholarProvider
from llm_claw.providers.aisa_tavily import AIsaTavilyProvider
from llm_claw.providers.router import ProviderRouter


def _task(task_type: str = "commercial_discovery_research") -> AcquisitionTask:
    return AcquisitionTask.model_validate({
        "domain": "commercial_discovery", "task_type": task_type,
        "entity": {"name": "Education AI", "metadata": {"topic": "Education AI"}},
        "data_needed": ["teacher workflow pain"],
        "provider_policy": {"allowed_providers": ["aisa_tavily", "aisa_scholar", "aisa_jina_rerank", "aisa_similarweb", "crawler"]},
    })


def test_tavily_discovers_only_public_https_candidates(tmp_path: Path) -> None:
    provider = AIsaTavilyProvider(Settings(workspace=tmp_path, aisa_api_key="test"))
    provider.client.post = lambda path, payload: {"results": [  # type: ignore[method-assign]
        {"url": "https://school.example/report", "title": "Teacher workflow", "content": "Teachers spend hours reviewing drafts."},
        {"url": "http://private.example/report", "title": "Insecure"},
    ]}

    candidates, trace = provider.discover(_task(), PlannedQuery(text="teacher workflow pain", data_need="pain"))

    assert [item.url for item in candidates] == ["https://school.example/report"]
    assert trace.candidate_count == 1
    assert candidates[0].source_type == "candidate"


def test_scholar_is_research_only_and_papers_still_require_fetch(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, aisa_api_key="test", provider_allowlist=["aisa_scholar", "crawler"])
    assert ProviderRouter(settings).select_providers(_task()) == ["crawler"]
    assert ProviderRouter(settings).select_providers(_task("commercial_deep_research")) == ["aisa_scholar", "crawler"]
    provider = AIsaScholarProvider(settings)
    seen: list[dict] = []
    def search(path: str, payload=None, params=None):
        seen.append({"path": path, "payload": payload, "params": params})
        return {"results": [{"title": "Teaching study", "link": "https://journal.example/study", "snippet": "A controlled study"}]}
    provider.client.post = search  # type: ignore[method-assign]
    candidates, _ = provider.discover(_task("commercial_deep_research"), PlannedQuery(text="AI teaching outcomes", data_need="academic"))
    assert seen[0]["payload"] is None
    assert seen[0]["params"]["query"] == "AI teaching outcomes"
    assert candidates[0].url == "https://journal.example/study"


def test_jina_rerank_changes_order_but_never_creates_candidates(tmp_path: Path) -> None:
    reranker = AIsaReranker(Settings(workspace=tmp_path, aisa_api_key="test"))
    reranker.client.post = lambda path, payload: {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}  # type: ignore[method-assign]
    candidates = [CandidateSource(provider="aisa_tavily", title=title, url=f"https://example.com/{index}") for index, title in enumerate(["Weak", "Strong"])]
    ranked, trace = reranker.rank("teacher pain", candidates)
    assert [item.title for item in ranked] == ["Strong", "Weak"]
    assert trace.status == "ok"
    reranker.client.post = lambda path, payload: {"results": [{"index": 99}]}  # type: ignore[method-assign]
    ranked, trace = reranker.rank("teacher pain", candidates)
    assert ranked == candidates
    assert trace.status == "error"


def test_extract_and_reddit_comment_fallbacks_are_bounded(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        workspace=tmp_path, aisa_api_key="test", provider_allowlist=["aisa_tavily", "aisa_reddit"],
        aisa_extract_max_fallbacks=1, aisa_reddit_comments_max_fallbacks=1,
    )
    fetcher = SourceFetcher(settings)
    monkeypatch.setattr(source_fetcher, "_public_url_error", lambda *_args: None)
    monkeypatch.setattr(source_fetcher, "_open_url", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("blocked")))
    monkeypatch.setattr(source_fetcher, "_curl_fetch", lambda *_args: None)
    monkeypatch.setenv("CLAW_ENABLE_BROWSER_FETCH", "0")
    fetcher.aisa.post = lambda path, payload: {"results": [{"url": "https://example.com/article", "raw_content": "Teachers report a lengthy manual review workflow every day."}]}  # type: ignore[method-assign]
    def comments(path, params):
        assert params == {"url": "https://www.reddit.com/r/Teachers/comments/123/topic/"}
        return {"post": {"url": params["url"], "title": "AI classroom concerns", "selftext": "Teachers question whether this saves time."}, "comments": [{"body": "The review process still takes hours."}]}
    fetcher.aisa.get = comments  # type: ignore[method-assign]
    sources, diagnostics = fetcher.fetch_with_diagnostics([
        CandidateSource(provider="aisa_tavily", title="Article", url="https://example.com/article"),
        CandidateSource(
            provider="aisa_reddit",
            title="Discussion",
            url="https://www.reddit.com/r/Teachers/comments/123/topic/",
            metadata={"platform": "reddit", "public_handle": "u/teacher_ops"},
        ),
    ], allowed_providers={"aisa_tavily", "aisa_reddit"})
    assert [item.metadata["fetch_mode"] for item in sources] == ["aisa_tavily_extract", "aisa_reddit_comments"]
    assert [item.status for item in diagnostics] == ["fetched", "fetched"]
    assert fetcher.aisa_extract_attempts == 1
    assert fetcher.aisa_reddit_comment_attempts == 1
    assert "Teachers question" in sources[1].text
    assert sources[1].metadata["platform"] == "reddit"
    assert sources[1].metadata["public_handle"] == "u/teacher_ops"


def test_fetch_fallback_respects_task_provider_policy(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(workspace=tmp_path, aisa_api_key="test", provider_allowlist=["aisa_tavily"])
    fetcher = SourceFetcher(settings)
    monkeypatch.setattr(source_fetcher, "_open_url", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("blocked")))
    monkeypatch.setattr(source_fetcher, "_curl_fetch", lambda *_args: None)
    monkeypatch.setattr(source_fetcher, "_node_fetch", lambda *_args: None)
    monkeypatch.setenv("CLAW_ENABLE_BROWSER_FETCH", "0")
    fetcher.aisa.post = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("AIsa must not be called"))  # type: ignore[method-assign]
    _sources, diagnostics = fetcher.fetch_with_diagnostics(
        [CandidateSource(provider="crawler", title="Public article", url="https://example.com/article")],
        allowed_providers={"crawler"},
    )
    assert diagnostics[0].status == "failed"
    assert fetcher.aisa_extract_attempts == 0


def test_similarweb_requires_explicit_public_domain_and_compacts_metrics(tmp_path: Path) -> None:
    provider = AIsaSimilarweb(Settings(workspace=tmp_path, aisa_api_key="test"))
    requests: list[dict] = []
    def get(path, params):
        requests.append(params)
        return {"data": [{"date": "2026-08-01", "visits": 1234, "bounce_rate": 0.3, "private_contact": "ignored"}]}
    provider.client.get = get  # type: ignore[method-assign]
    snapshots, traces = provider.traffic(["localhost", "example.com"], limit=2)
    assert _valid_domain("https://user:pass@example.com/") is None
    assert snapshots[0]["domain"] == "example.com"
    assert snapshots[0]["metrics"] == {"date": "2026-08-01", "visits": 1234, "bounce_rate": 0.3}
    assert requests[0]["start_date"] == requests[0]["end_date"] == snapshots[0]["period"]
    assert requests[0]["metrics"] == "visits,pages_per_visit,average_visit_duration,bounce_rate"
    assert len(traces) == 1
