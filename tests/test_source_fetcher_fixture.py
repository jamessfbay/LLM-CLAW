from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import CandidateSource
from llm_claw.pipeline.source_fetcher import SourceFetcher, _html_to_text
from llm_claw.pipeline.extractors import _sentences
from llm_claw.pipeline import source_fetcher


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


def test_html_text_keeps_inline_field_labels_with_values() -> None:
    text = _html_to_text(
        "<span>CIK: <a href='/company'>0001965143</a></span>"
        "<p>Type: <strong>6-K</strong></p>"
    )

    assert "CIK: 0001965143" in text
    assert "Type: 6-K" in text


def test_sentence_windows_keep_block_field_labels_with_values() -> None:
    sentences = _sentences("SEC\nAccession\nNo.\n0001213900-26-082918\nFiling Date\n2026-07-29")

    assert any("Accession No. 0001213900-26-082918" in item for item in sentences)


def test_browser_fallback_retries_a_blocked_http_response(tmp_path: Path, monkeypatch) -> None:
    candidate = CandidateSource(
        provider="crawler",
        title="Official planning projects",
        url="https://example.test/planning",
        is_official=True,
    )
    monkeypatch.setenv("CLAW_ENABLE_BROWSER_FETCH", "1")
    monkeypatch.setattr(source_fetcher, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("blocked")))
    monkeypatch.setattr(source_fetcher, "_curl_fetch", lambda _url, _user_agent: (b"<html>Access denied</html>", "text/html"))
    monkeypatch.setattr(source_fetcher, "_node_fetch", lambda _url, _user_agent: None)
    monkeypatch.setattr(
        source_fetcher,
        "_browser_fetch",
        lambda _url, _user_agent: (
            b"<html><body><h1>Planning projects</h1><p>Permit and entitlement review information is available.</p></body></html>",
            "text/html",
            candidate.url,
        ),
    )

    sources, diagnostics = SourceFetcher(Settings(workspace=tmp_path)).fetch_with_diagnostics([candidate])

    assert len(sources) == 1
    assert diagnostics[0].status == "fetched"
    assert diagnostics[0].fetch_mode == "browser"
    assert "entitlement review" in sources[0].text


def test_node_fallback_recovers_when_http_and_curl_are_blocked(tmp_path: Path, monkeypatch) -> None:
    candidate = CandidateSource(
        provider="crawler",
        title="Official planning projects",
        url="https://example.test/planning",
        is_official=True,
    )
    monkeypatch.setattr(source_fetcher, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("blocked")))
    monkeypatch.setattr(source_fetcher, "_curl_fetch", lambda _url, _user_agent: (b"<html>Access denied</html>", "text/html"))
    monkeypatch.setattr(
        source_fetcher,
        "_node_fetch",
        lambda _url, _user_agent: (
            b"<html><body><h1>Planning projects</h1><p>Permit and entitlement review information is available.</p></body></html>",
            "text/html",
        ),
    )

    sources, diagnostics = SourceFetcher(Settings(workspace=tmp_path)).fetch_with_diagnostics([candidate])

    assert len(sources) == 1
    assert diagnostics[0].fetch_mode == "node"
    assert diagnostics[0].status == "fetched"
