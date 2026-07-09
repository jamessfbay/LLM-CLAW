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
