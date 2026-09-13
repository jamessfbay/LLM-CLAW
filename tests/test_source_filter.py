from llm_claw.models import AcquisitionTask, CandidateSource, RawSource
from llm_claw.pipeline.source_filter import SourceRelevanceFilter
from llm_claw.models import PlannedQuery
from llm_claw.providers.openai_web_search import _build_prompt


def _task() -> AcquisitionTask:
    return AcquisitionTask.model_validate(
        {
            "domain": "real_estate",
            "task_type": "project_research",
            "entity": {
                "project_name": "156 California Avenue Mixed-Use Project",
                "city": "Palo Alto",
                "address": "156 California Ave, Palo Alto, CA 94306",
            },
            "data_needed": ["planning status", "staff report", "public comments", "CEQA status"],
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


def test_source_filter_keeps_official_planning_seed_candidate() -> None:
    task = _task()
    filterer = SourceRelevanceFilter()
    candidates = [
        CandidateSource(
            provider="crawler",
            title="Planning & Development Services",
            url="https://www.paloalto.gov/Departments/Planning-Development-Services",
            snippet="Official planning, development, and permit services page.",
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


def test_source_filter_keeps_official_planning_source_with_operational_signal() -> None:
    task = _task()
    filterer = SourceRelevanceFilter()
    source = RawSource(
        source_url="https://www.paloalto.gov/Departments/Planning-Development-Services",
        source_title="Planning & Development Services",
        source_type="official_html",
        content_hash="hash",
        text="Planning applications, development review, permit history, zoning, and public hearing agendas.",
    )

    assert filterer.filter_sources(task, [source]) == [source]


def test_source_filter_normalizes_ampersand_in_cross_industry_entity_names() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "consumer_product_safety",
            "entity": {"name": "Fisher and Paykel Gas Ranges", "type": "consumer_product"},
            "data_needed": ["burn hazard", "repair remedy"],
        }
    )
    source = RawSource(
        source_url="https://www.cpsc.gov/Recalls/example",
        source_title="Fisher & Paykel Gas Ranges Recalled",
        source_type="official_html",
        content_hash="hash",
        text="Fisher & Paykel Gas Ranges were recalled because delayed ignition poses a burn hazard.",
    )

    assert SourceRelevanceFilter().filter_sources(task, [source]) == [source]


def test_source_filter_keeps_official_source_with_domain_task_signals() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "cybersecurity",
            "entity": {"name": "Enterprise vulnerability response", "type": "cybersecurity_alert"},
            "data_needed": ["active exploitation", "remediation requirement"],
            "question": "Which actively exploited vulnerabilities require remediation?",
        }
    )
    source = RawSource(
        source_url="https://www.cisa.gov/news-events/alerts/example",
        source_title="Known Exploited Vulnerabilities Alert",
        source_type="official_html",
        content_hash="hash",
        text="CISA identified active exploitation and requires timely remediation of catalog vulnerabilities.",
    )

    assert SourceRelevanceFilter().filter_sources(task, [source]) == [source]


def test_source_filter_keeps_commercial_candidates_by_task_terms_not_run_id() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "commercial_discovery",
            "entity": {"name": "run-uuid", "type": "discovery_run"},
            "question": "warehouse ammonia refrigeration leak compliance software",
            "data_needed": ["ammonia leak compliance pain"],
        }
    )
    candidate = CandidateSource(
        provider="openai_web_search",
        title="Ammonia refrigeration compliance",
        url="https://iiarcondenser.org/ammonia-refrigeration-compliance/",
        snippet="Warehouse operators describe ammonia leak compliance reporting.",
        is_official=False,
    )

    assert SourceRelevanceFilter().filter_candidates(task, [candidate]) == [candidate]


def test_source_filter_rejects_unrelated_commercial_candidate() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "commercial_discovery",
            "entity": {"name": "run-uuid", "type": "discovery_run"},
            "question": "warehouse ammonia refrigeration leak compliance software",
            "data_needed": ["ammonia leak compliance pain"],
        }
    )
    candidate = CandidateSource(
        provider="gemini",
        title="Consumer travel trends",
        url="https://example.com/travel",
        snippet="A report about airline vacation bookings.",
        is_official=False,
    )

    assert SourceRelevanceFilter().filter_candidates(task, [candidate]) == []


def test_commercial_topic_anchor_rejects_generic_ai_results() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "commercial_discovery",
            "entity": {"name": "run-uuid", "metadata": {"topic": "AI for Education"}},
            "question": "Find buyer pain and workflow problems in AI for Education.",
            "data_needed": ["buyer workflow pain", "pricing alternatives"],
        }
    )
    unrelated = CandidateSource(
        provider="openai_web_search",
        title="Terminal workflow built for AI agents",
        url="https://news.ycombinator.com/item?id=1",
        snippet="Developers switched from tmux to an agent terminal.",
    )
    relevant = CandidateSource(
        provider="openai_web_search",
        title="Teachers struggle to review AI-generated assignments",
        url="https://example.com/teacher-workflow",
        snippet="Schools report teacher workflow and classroom review pain.",
    )

    assert SourceRelevanceFilter().filter_candidates(task, [unrelated, relevant]) == [relevant]


def test_commercial_topic_anchor_filters_fetched_source_content() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "commercial_discovery",
            "entity": {"name": "run-uuid", "metadata": {"topic": "AI for Education"}},
            "question": "Find buyer pain and workflow problems in AI for Education.",
            "data_needed": ["buyer workflow pain"],
        }
    )
    unrelated = RawSource(
        source_url="https://example.com/terminal",
        source_title="AI agent terminal",
        source_type="webpage",
        content_hash="one",
        text="Developers describe a better terminal workflow for autonomous coding agents.",
    )
    relevant = RawSource(
        source_url="https://example.com/classroom",
        source_title="Teacher AI review workload",
        source_type="webpage",
        content_hash="two",
        text="Teachers and schools report a manual classroom workflow for reviewing student AI assignments.",
    )

    assert SourceRelevanceFilter().filter_sources(task, [unrelated, relevant]) == [relevant]


def test_web_search_prompt_requires_direct_topic_evidence() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "commercial_discovery",
            "entity": {"name": "run-uuid", "metadata": {"topic": "AI for Education"}},
            "question": "Find commercial pain in AI for Education.",
            "data_needed": ["buyer workflow pain"],
        }
    )
    prompt = _build_prompt(task, PlannedQuery(text='"AI for Education" buyer pain', data_need="buyer pain"))

    assert "Topical scope: AI for Education" in prompt
    assert "Reject generic AI" in prompt
