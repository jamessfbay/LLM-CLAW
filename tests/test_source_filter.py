from llm_claw.models import AcquisitionTask, CandidateSource, RawSource
from llm_claw.pipeline.source_filter import SourceRelevanceFilter


def _task() -> AcquisitionTask:
    return AcquisitionTask.model_validate(
        {
            "entity": {
                "project_name": "156 California Avenue Mixed-Use Project",
                "city": "Palo Alto",
                "address": "156 California Ave, Palo Alto, CA 94306",
            }
        }
    )


def test_source_filter_blocks_generic_city_data_and_campaign_pages() -> None:
    task = _task()
    filterer = SourceRelevanceFilter()
    candidates = [
        CandidateSource(
            provider="openai_web_search",
            title="City of Palo Alto Electronic Filing System",
            url="https://efile.cityofpaloalto.org/public/search/campaign?current_page=1",
            snippet="Campaign filings and statements.",
            is_official=True,
        ),
        CandidateSource(
            provider="openai_web_search",
            title="Palo Alto Development Center Permits",
            url="https://data.cityofpaloalto.org/dashboards/7712/palo-alto-development-center-permits/",
            snippet="Updated weekly permit dashboard.",
            is_official=True,
        ),
        CandidateSource(
            provider="gemini",
            title="OpenGIS data",
            url="https://opengis.cityofpaloalto.org/OpenGisData/",
            snippet="Citywide GIS data downloads.",
            is_official=True,
        ),
    ]

    assert filterer.filter_candidates(task, candidates) == []


def test_source_filter_keeps_project_specific_official_candidates() -> None:
    task = _task()
    filterer = SourceRelevanceFilter()
    candidates = [
        CandidateSource(
            provider="gemini",
            title="156 California Avenue Mixed-Use Project",
            url="https://ceqanet.lci.ca.gov/2024120754",
            snippet="Official CEQAnet record for 156 California Avenue Mixed-Use Project.",
            is_official=True,
        )
    ]

    assert filterer.filter_candidates(task, candidates) == candidates


def test_source_filter_keeps_official_city_youtube_channel_candidate() -> None:
    task = _task()
    filterer = SourceRelevanceFilter()
    candidates = [
        CandidateSource(
            provider="crawler",
            title="City of Palo Alto YouTube videos",
            url="https://www.youtube.com/@cityofpaloalto/videos",
            snippet="Official City of Palo Alto YouTube channel videos.",
            is_official=True,
        )
    ]

    assert filterer.filter_candidates(task, candidates) == candidates


def test_source_filter_keeps_official_youtube_watch_candidate() -> None:
    task = _task()
    filterer = SourceRelevanceFilter()
    candidates = [
        CandidateSource(
            provider="crawler",
            title="City Council Meeting - June 1, 2026",
            url="https://www.youtube.com/watch?v=Cczy-CGO8IE",
            source_type="youtube",
            is_official=True,
        )
    ]

    assert filterer.filter_candidates(task, candidates) == candidates


def test_source_filter_blocks_unrelated_raw_source_text() -> None:
    task = _task()
    filterer = SourceRelevanceFilter()
    source = RawSource(
        source_url="https://efile.cityofpaloalto.org/public/search/campaign",
        source_title="City of Palo Alto Electronic Filing System",
        source_type="official_html",
        content_hash="hash",
        text="Campaign committee filings and officeholder reports. Public comments are available.",
    )

    assert filterer.filter_sources(task, [source]) == []


def test_source_filter_keeps_local_fixture_sources() -> None:
    task = _task()
    filterer = SourceRelevanceFilter()
    source = RawSource(
        source_url="file:///tmp/staff_report.html",
        source_title="Planning Commission Staff Report",
        source_type="local_html",
        content_hash="hash",
        text="A local fixture that intentionally may not match the project.",
    )

    assert filterer.filter_sources(task, [source]) == [source]
