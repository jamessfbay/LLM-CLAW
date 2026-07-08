from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import CandidateSource
from llm_claw.pipeline.source_fetcher import SourceFetcher


def test_source_fetcher_reads_local_html_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "staff_report.html"
    candidate = CandidateSource(
        provider="crawler",
        title="Planning Commission Staff Report",
        url=fixture.resolve().as_uri(),
        publisher="City of San Jose",
        is_official=True,
    )

    sources = SourceFetcher(Settings(workspace=tmp_path)).fetch([candidate])

    assert len(sources) == 1
    assert sources[0].source_type == "local_html"
    assert "planning status is under review" in sources[0].text
    assert sources[0].raw_path


def test_source_fetcher_records_diagnostics_for_local_html(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "staff_report.html"
    candidate = CandidateSource(
        provider="crawler",
        title="Planning Commission Staff Report",
        url=fixture.resolve().as_uri(),
        publisher="City of San Jose",
        is_official=True,
    )

    sources, diagnostics = SourceFetcher(Settings(workspace=tmp_path)).fetch_with_diagnostics([candidate])

    assert len(sources) == 1
    assert len(diagnostics) == 1
    assert diagnostics[0].status == "fetched"
    assert diagnostics[0].candidate_id == candidate.id
    assert diagnostics[0].raw_path
    assert diagnostics[0].text_length > 0


def test_source_fetcher_records_drop_reason_for_empty_source(tmp_path: Path) -> None:
    empty = tmp_path / "empty.html"
    empty.write_text("<html><body><script>window.app = true</script></body></html>", encoding="utf-8")
    candidate = CandidateSource(
        provider="crawler",
        title="Empty app shell",
        url=empty.resolve().as_uri(),
    )

    sources, diagnostics = SourceFetcher(Settings(workspace=tmp_path)).fetch_with_diagnostics([candidate])

    assert sources == []
    assert len(diagnostics) == 1
    assert diagnostics[0].status == "dropped"
    assert diagnostics[0].drop_reason == "empty_text"
    assert diagnostics[0].raw_path
