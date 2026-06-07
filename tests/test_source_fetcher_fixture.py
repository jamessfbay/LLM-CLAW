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
