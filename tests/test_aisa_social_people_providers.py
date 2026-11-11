from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, PlannedQuery
from llm_claw.models import CandidateSource
from llm_claw.pipeline.source_fetcher import SourceFetcher
from llm_claw.pipeline.source_filter import SourceRelevanceFilter
from llm_claw.providers.aisa_apollo_people import AIsaApolloPeopleProvider, _people_query
from llm_claw.providers.aisa_twitter import (
    AIsaTwitterProvider,
    _is_pain_or_solution_signal,
    _normalize_tweet,
    _twitter_query,
)


def _task(provider: str) -> AcquisitionTask:
    return AcquisitionTask.model_validate({
        "domain": "commercial_discovery",
        "task_type": "candidate_buyer_discovery",
        "entity": {"name": "Agent governance", "metadata": {"topic": "AI agent governance"}},
        "data_needed": ["public candidate buyers"],
        "provider_policy": {"allowed_providers": [provider, "crawler"]},
    })


def test_aisa_twitter_preserves_public_author_and_provenance(tmp_path: Path) -> None:
    provider = AIsaTwitterProvider(Settings(workspace=tmp_path, aisa_api_key="test-key"))
    provider.client.get = lambda _path, _params: {  # type: ignore[method-assign]
        "request_id": "req-x-1",
        "data": {
            "tweets": [{
                "rest_id": "123456",
                "legacy": {
                    "full_text": "We still review every AI agent action by hand and need a safer workflow.",
                    "created_at": "Mon Nov 09 10:00:00 +0000 2026",
                },
                "core": {"user_results": {"result": {"legacy": {
                    "screen_name": "ops_builder",
                    "name": "Ops Builder",
                }}}},
            }],
        },
    }

    candidates, trace = provider.discover(
        _task("aisa_twitter"),
        PlannedQuery(text='site:x.com "AI agent governance" problem workaround', data_need="X pain authors"),
    )

    assert trace.candidate_count == 1
    assert candidates[0].url == "https://x.com/ops_builder/status/123456"
    assert candidates[0].metadata["public_handle"] == "@ops_builder"
    assert candidates[0].metadata["person_name"] == "Ops Builder"
    assert candidates[0].metadata["source_provider"] == "aisa_twitter"
    assert candidates[0].metadata["source_method"] == "platform_api"
    assert candidates[0].metadata["source_request_id"] == "req-x-1"
    assert "test-key" not in str(candidates[0].metadata)


def test_aisa_twitter_accepts_documented_passthrough_field_names() -> None:
    normalized = _normalize_tweet({
        "id": "789",
        "url": "https://x.com/platform_lead/status/789",
        "text": "Looking for a better way to audit agent actions.",
        "createdAt": "2026-11-10T12:00:00Z",
        "author": {"userName": "platform_lead", "name": "Platform Lead"},
    })

    assert normalized == {
        "text": "Looking for a better way to audit agent actions.",
        "url": "https://x.com/platform_lead/status/789",
        "username": "platform_lead",
        "published_at": "2026-11-10T12:00:00+00:00",
        "person_name": "Platform Lead",
    }


def test_aisa_twitter_uses_topic_plus_alternative_intent_terms() -> None:
    query = _twitter_query('site:x.com "AI agent governance" problem workaround looking for tool')

    assert query.startswith('"AI agent governance" "looking for"')
    assert "lang:en" in query
    assert "site:x.com" not in query


def test_aisa_twitter_rejects_generic_news_but_keeps_explicit_pain() -> None:
    query = _twitter_query('site:x.com "AI agents" problem workaround')

    assert not _is_pain_or_solution_signal("AI agents will change every industry.", query)
    assert _is_pain_or_solution_signal("We still have to review AI agents manually and need help.", query)


def test_aisa_apollo_search_uses_the_core_market_phrase() -> None:
    assert _people_query("site:linkedin.com/in AI agent startup founder platform engineering") == "AI agent"


def test_aisa_apollo_rejects_low_confidence_identity_matches(tmp_path: Path) -> None:
    provider = AIsaApolloPeopleProvider(Settings(
        workspace=tmp_path,
        aisa_api_key="test-key",
        aisa_apollo_people_max_enrichments=1,
    ))

    def fake_post(path, payload=None, params=None):
        if path == provider.search_endpoint:
            return {"people": [{"id": "person-low"}]}
        return {
            "person": {
                "name": "Possible Person",
                "title": "Founder",
                "organization": {"name": "Possible Co"},
                "linkedin_url": "https://www.linkedin.com/in/possible-person",
                "match_confidence": "low",
            },
        }

    provider.client.post = fake_post  # type: ignore[method-assign]
    candidates, trace = provider.discover(
        _task("aisa_apollo_people"),
        PlannedQuery(text="site:linkedin.com/in AI agent founder", data_need="economic buyers"),
    )

    assert candidates == []
    assert trace.candidate_count == 0


def test_aisa_apollo_keeps_only_named_people_with_public_linkedin_profiles(tmp_path: Path) -> None:
    provider = AIsaApolloPeopleProvider(Settings(
        workspace=tmp_path,
        aisa_api_key="test-key",
        aisa_apollo_people_max_enrichments=2,
    ))

    def fake_post(path, payload=None, params=None):
        if path == provider.search_endpoint:
            return {"people": [{"id": "person-1"}, {"id": "person-2"}]}
        if payload == {"id": "person-1", "reveal_personal_emails": False}:
            return {
                "request_id": "req-people-1",
                "person": {
                    "first_name": "Jordan",
                    "last_name": "Lee",
                    "title": "VP of Platform Engineering",
                    "organization": {"name": "Acme Robotics"},
                    "linkedin_url": "http://www.linkedin.com/in/jordan-lee/",
                    "match_confidence": "high",
                    "email": "private@example.com",
                    "phone_numbers": ["+1-555-private"],
                },
            }
        return {"person": {"name": "No Public Profile", "title": "Director"}}

    provider.client.post = fake_post  # type: ignore[method-assign]

    candidates, trace = provider.discover(
        _task("aisa_apollo_people"),
        PlannedQuery(
            text='site:linkedin.com/in "AI agent governance" platform engineering security governance',
            data_need="economic buyers",
        ),
    )

    assert trace.candidate_count == 1
    assert candidates[0].url == "https://www.linkedin.com/in/jordan-lee/"
    assert candidates[0].metadata["person_name"] == "Jordan Lee"
    assert candidates[0].metadata["target_role"] == "VP of Platform Engineering"
    assert candidates[0].metadata["organization_name"] == "Acme Robotics"
    assert candidates[0].metadata["source_provider"] == "aisa_apollo_people"
    assert candidates[0].metadata["source_method"] == "licensed_public_professional_data"
    assert candidates[0].metadata["match_confidence"] == "high"
    assert "email" not in candidates[0].metadata
    assert "phone" not in candidates[0].metadata
    assert SourceRelevanceFilter().is_relevant_candidate(_task("aisa_apollo_people"), candidates[0])


def test_aisa_social_results_become_traceable_api_snapshots_without_scraping_profiles(
    tmp_path: Path, monkeypatch,
) -> None:
    settings = Settings(
        workspace=tmp_path,
        aisa_api_key="test-key",
        provider_allowlist=["aisa_twitter", "crawler"],
    )
    candidate = CandidateSource(
        provider="aisa_twitter",
        title="@ops_builder: manual review is painful",
        url="https://x.com/ops_builder/status/123456",
        snippet="We still review every coding agent action by hand.",
        metadata={
            "platform": "x",
            "public_handle": "@ops_builder",
            "source_provider": "aisa_twitter",
            "source_label": "AISA Twitter API",
            "source_method": "platform_api",
        },
    )
    monkeypatch.setattr("llm_claw.pipeline.source_fetcher._public_url_error", lambda *_args: None)

    sources, diagnostics = SourceFetcher(settings).fetch_with_diagnostics(
        [candidate], allowed_providers={"aisa_twitter", "crawler"},
    )

    assert len(sources) == 1
    assert "review every coding agent action" in sources[0].text
    assert sources[0].metadata["source_provider"] == "aisa_twitter"
    assert sources[0].metadata["fetch_mode"] == "aisa_twitter_api"
    assert diagnostics[0].fetch_mode == "aisa_twitter_api"
