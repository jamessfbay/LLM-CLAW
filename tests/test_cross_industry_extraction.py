import json

from llm_claw.models import AcquisitionTask, RawSource
from llm_claw.pipeline.extractors import EvidenceExtractor
from llm_claw.pipeline.engine import _need_is_covered


def _source(text: str) -> RawSource:
    return RawSource(
        source_url="https://agency.gov/alert",
        source_title="Official alert",
        source_type="official_html",
        content_hash="hash",
        text=text,
    )


def test_extracts_cve_identifiers_without_literal_identifier_word() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "cybersecurity",
            "entity": {"name": "CISA KEV update"},
            "data_needed": ["CVE identifiers"],
        }
    )
    source = _source(
        "CISA added three vulnerabilities to the catalog. "
        "The additions are CVE-2025-31200, CVE-2025-31201, and CVE-2025-24054."
    )

    claims = EvidenceExtractor().extract_claims(task, [source])

    assert len(claims) == 1
    assert "CVE-2025-31200" in claims[0].evidence_text
    assert claims[0].text == claims[0].evidence_text


def test_prefers_substantive_evidence_over_government_header_boilerplate() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "healthcare",
            "entity": {"name": "H3 ankle replacement"},
            "data_needed": ["device failure risk", "effective date"],
        }
    )
    source = _source(
        "An official website of the United States government. "
        "Date Issued: June 3, 2026. "
        "The FDA identified an elevated long-term risk of device failure and revision surgery."
    )

    claims = EvidenceExtractor().extract_claims(task, [source])

    assert len(claims) == 2
    assert any("elevated long-term risk" in claim.evidence_text for claim in claims)
    assert any("June 3, 2026" in claim.evidence_text for claim in claims)
    assert all("official website" not in claim.evidence_text.lower() for claim in claims)


def test_missing_data_coverage_handles_identifiers_and_word_variants() -> None:
    evidence = "CVE-2025-31200 requires agencies to remediate identified vulnerabilities."

    assert _need_is_covered("CVE identifiers", evidence)
    assert _need_is_covered("remediation requirement", evidence)
    assert _need_is_covered("effective date", "Washington D.C., July 7, 2026")


def test_extracts_publication_date_as_effective_date_evidence() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "financial_services",
            "entity": {"name": "Retail Fraud Working Group"},
            "data_needed": ["effective date"],
        }
    )
    source = _source("Washington D.C., July 7, 2026 — The SEC announced a new working group.")

    claims = EvidenceExtractor().extract_claims(task, [source])

    assert len(claims) == 1
    assert "July 7, 2026" in claims[0].evidence_text


def test_extracts_changed_usgs_geojson_events_as_structured_evidence() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "emergency_management",
            "entity": {
                "name": "USGS earthquake feed",
                "metadata": {"changed_record_ids": ["us-test-1"]},
            },
            "data_needed": [
                "USGS event ID",
                "magnitude",
                "event time",
                "updated time",
                "place",
                "coordinates and depth",
                "review status",
                "alert or tsunami indicator",
                "official source citation",
            ],
        }
    )
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "us-test-1",
                "properties": {
                    "type": "earthquake",
                    "mag": 4.3,
                    "place": "8 km ESE of Cloverdale, CA",
                    "time": 1785292806730,
                    "updated": 1785295898564,
                    "status": "reviewed",
                    "alert": "green",
                    "tsunami": 0,
                    "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us-test-1",
                },
                "geometry": {"type": "Point", "coordinates": [-122.93, 38.77, 5.53]},
            },
            {
                "type": "Feature",
                "id": "unchanged-event",
                "properties": {"type": "earthquake", "mag": 1.0},
                "geometry": {"type": "Point", "coordinates": [-120, 37, 2]},
            },
        ],
    }
    source = RawSource(
        source_url="https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
        source_title="USGS All Earthquakes, Past Hour",
        source_type="government_api",
        content_hash="hash",
        text=json.dumps(payload),
        metadata={"content_type": "application/geo+json"},
    )

    claims = EvidenceExtractor().extract_claims(task, [source])

    assert len(claims) == 1
    assert "USGS event ID us-test-1" in claims[0].text
    assert "magnitude 4.3" in claims[0].text
    assert "review status reviewed" in claims[0].text
    assert "Official source citation" in claims[0].text
    assert "unchanged-event" not in claims[0].text
    for need in task.data_needed:
        assert _need_is_covered(need, claims[0].text)


def test_usgs_removed_event_does_not_reuse_unrelated_current_events() -> None:
    task = AcquisitionTask.model_validate(
        {
            "domain": "emergency_management",
            "entity": {
                "name": "USGS earthquake feed",
                "metadata": {
                    "changed_record_ids": [],
                    "removed_record_ids": ["removed-event"],
                },
            },
            "data_needed": ["USGS event ID", "magnitude", "official source citation"],
        }
    )
    payload = {
        "type": "FeatureCollection",
        "metadata": {"generated": 1785295969000},
        "features": [
            {
                "type": "Feature",
                "id": "current-event",
                "properties": {"type": "earthquake", "mag": 2.0},
                "geometry": {"type": "Point", "coordinates": [-120, 37, 2]},
            }
        ],
    }
    source = RawSource(
        source_url="https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
        source_title="USGS All Earthquakes, Past Hour",
        source_type="government_api",
        content_hash="hash",
        text=json.dumps(payload),
        metadata={"content_type": "application/geo+json"},
    )

    claims = EvidenceExtractor().extract_claims(task, [source])

    assert len(claims) == 1
    assert "removed-event is not present" in claims[0].text
    assert "current-event" not in claims[0].text
    assert not _need_is_covered("magnitude", claims[0].text)
