from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import CandidateSource, RawSource, SourceFetchDiagnostic


class SourceFetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.raw_dir = settings.workspace / ".llm_claw" / "raw_sources"

    def fetch(self, candidates: list[CandidateSource]) -> list[RawSource]:
        sources, _diagnostics = self.fetch_with_diagnostics(candidates)
        return sources

    def fetch_with_diagnostics(
        self, candidates: list[CandidateSource]
    ) -> tuple[list[RawSource], list[SourceFetchDiagnostic]]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        sources: list[RawSource] = []
        diagnostics: list[SourceFetchDiagnostic] = []
        for candidate in candidates:
            source, diagnostic = self._fetch_one(candidate)
            diagnostics.append(diagnostic)
            if source:
                sources.append(source)
        return sources, diagnostics

    def _fetch_one(self, candidate: CandidateSource) -> tuple[RawSource | None, SourceFetchDiagnostic]:
        if candidate.url.startswith("mock://"):
            source = self._mock_source(candidate)
            return source, _diagnostic(candidate, status="fetched", fetch_mode="mock", raw_path=source.raw_path, text_length=len(source.text), bytes=len(source.text.encode("utf-8")))
        if candidate.url.startswith("file://"):
            return self._file_source(candidate)
        if candidate.url.startswith("http://") or candidate.url.startswith("https://"):
            return self._http_source(candidate)
        return None, _diagnostic(candidate, status="failed", drop_reason="unsupported_url_scheme")

    def _mock_source(self, candidate: CandidateSource) -> RawSource:
        body = (
            f"{candidate.title}\n\n"
            f"{candidate.snippet}\n"
            f"The project is referenced in an official planning record. "
            f"Planning status appears under review. CEQA status requires source verification."
        )
        digest = _hash(body.encode("utf-8"))
        path = self.raw_dir / f"{digest}.html"
        path.write_text(f"<html><title>{candidate.title}</title><body>{body}</body></html>", encoding="utf-8")
        return RawSource(
            candidate_id=candidate.id,
            source_url=candidate.url,
            source_title=candidate.title,
            publisher=candidate.publisher,
            source_type="official_html" if candidate.is_official else "webpage",
            content_hash=digest,
            raw_path=str(path),
            text=body,
            metadata={"mock": True},
        )

    def _file_source(self, candidate: CandidateSource) -> tuple[RawSource | None, SourceFetchDiagnostic]:
        parsed = urlparse(candidate.url)
        path = Path(parsed.path)
        if not path.exists():
            return None, _diagnostic(candidate, status="failed", fetch_mode="file", drop_reason="file_not_found")
        data = path.read_bytes()
        digest = _hash(data)
        copy_path = self.raw_dir / f"{digest}{path.suffix.lower()}"
        copy_path.write_bytes(data)
        text = _extract_file_text(path, data)
        drop_reason = _drop_reason(text)
        if drop_reason:
            return None, _diagnostic(
                candidate,
                status="dropped",
                fetch_mode="file",
                bytes=len(data),
                text_length=len(text),
                raw_path=str(copy_path),
                drop_reason=drop_reason,
            )
        if _is_blocked_response(text):
            return None, _diagnostic(candidate, status="dropped", fetch_mode="file", bytes=len(data), text_length=len(text), raw_path=str(copy_path), drop_reason="blocked_response")
        source_type = "local_pdf" if path.suffix.lower() == ".pdf" else "local_html"
        source = RawSource(
            candidate_id=candidate.id,
            source_url=candidate.url,
            source_title=candidate.title,
            publisher=candidate.publisher,
            source_type=source_type,
            content_hash=digest,
            raw_path=str(copy_path),
            text=text,
            metadata={"file_path": str(path)},
        )
        return source, _diagnostic(candidate, status="fetched", fetch_mode="file", bytes=len(data), text_length=len(text), raw_path=str(copy_path))

    def _http_source(self, candidate: CandidateSource) -> tuple[RawSource | None, SourceFetchDiagnostic]:
        request = Request(candidate.url, headers={"User-Agent": "llm-claw/0.1"})
        fetch_mode = "http"
        http_status: int | None = None
        final_url: str | None = None
        error: str | None = None
        try:
            with urlopen(request, timeout=15) as response:
                data = response.read()
                content_type = response.headers.get("content-type", "")
                http_status = getattr(response, "status", None)
                final_url = response.geturl()
        except Exception as exc:
            error = str(exc)
            fetched = _curl_fetch(candidate.url)
            if not fetched:
                if os.getenv("CLAW_ENABLE_BROWSER_FETCH") == "1":
                    browser = _browser_fetch(candidate.url)
                    if browser:
                        data, content_type, final_url = browser
                        fetch_mode = "browser"
                    else:
                        return None, _diagnostic(candidate, status="failed", fetch_mode="http", drop_reason="fetch_failed", error=error)
                else:
                    return None, _diagnostic(candidate, status="failed", fetch_mode="http", drop_reason="fetch_failed", error=error)
            else:
                data, content_type = fetched
                fetch_mode = "curl"
        digest = _hash(data)
        suffix = ".pdf" if "pdf" in content_type or candidate.url.lower().endswith(".pdf") else ".html"
        path = self.raw_dir / f"{digest}{suffix}"
        path.write_bytes(data)
        text = _extract_file_text(path, data)
        drop_reason = _drop_reason(text)
        if drop_reason:
            return None, _diagnostic(
                candidate,
                status="dropped",
                fetch_mode=fetch_mode,
                http_status=http_status,
                final_url=final_url,
                content_type=content_type,
                bytes=len(data),
                text_length=len(text),
                raw_path=str(path),
                drop_reason=drop_reason,
                error=error,
            )
        source_type = (
            "youtube"
            if _is_youtube_url(candidate.url)
            else "official_pdf"
            if suffix == ".pdf" and candidate.is_official
            else "pdf"
            if suffix == ".pdf"
            else "official_html"
            if candidate.is_official
            else "webpage"
        )
        source = RawSource(
            candidate_id=candidate.id,
            source_url=candidate.url,
            source_title=candidate.title,
            publisher=candidate.publisher,
            source_type=source_type,
            content_hash=digest,
            raw_path=str(path),
            text=text,
            metadata={"content_type": content_type, "fetch_mode": fetch_mode, "final_url": final_url},
        )
        return source, _diagnostic(
            candidate,
            status="fetched",
            fetch_mode=fetch_mode,
            http_status=http_status,
            final_url=final_url,
            content_type=content_type,
            bytes=len(data),
            text_length=len(text),
            raw_path=str(path),
            error=error,
        )


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_file_text(path: Path, data: bytes) -> str:
    if path.suffix.lower() == ".pdf":
        if data.lstrip().lower().startswith((b"<html", b"<!doctype html")):
            return _html_to_text(data.decode("utf-8", errors="ignore"))
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(f"[Page {index + 1}]\n{text}" for index, text in enumerate(pages))
        except Exception:
            return ""
    text = data.decode("utf-8", errors="ignore")
    return _html_to_text(text)


def _html_to_text(text: str) -> str:
    parser = _ReadableHTMLParser()
    try:
        parser.feed(text)
        readable = parser.text()
        if readable:
            return readable
    except Exception:
        pass
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self._chunks: list[str] = []
        self._link_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1
            return
        if tag == "a":
            self._link_href = dict(attrs).get("href")
        if tag in {"title", "h1", "h2", "h3", "p", "li", "tr", "th", "td", "button", "label", "option", "div", "section"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1
        if tag == "a":
            self._link_href = None
        if tag in {"title", "h1", "h2", "h3", "p", "li", "tr", "th", "td", "button", "label", "option", "div", "section"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        value = re.sub(r"\s+", " ", data).strip()
        if not value:
            return
        if self._link_href and self._link_href.startswith(("http://", "https://")):
            value = f"{value} ({self._link_href})"
        self._chunks.append(value)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", "\n".join(self._chunks))).strip()


def _curl_fetch(url: str) -> tuple[bytes, str] | None:
    try:
        result = subprocess.run(
            ["curl", "-L", "-sS", "-A", "Mozilla/5.0", url],
            check=False,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    content_type = "application/pdf" if url.lower().endswith(".pdf") else "text/html"
    return result.stdout, content_type


def _browser_fetch(url: str) -> tuple[bytes, str, str | None] | None:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            response = page.goto(url, wait_until="networkidle", timeout=30000)
            html = page.content()
            final_url = page.url
            content_type = response.headers.get("content-type", "text/html") if response else "text/html"
            browser.close()
            return html.encode("utf-8"), content_type, final_url
    except Exception:
        return None


def _is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _is_blocked_response(text: str) -> bool:
    lower = text.lower()
    blocked_markers = [
        "access denied",
        "you don't have permission to access",
        "request blocked",
    ]
    return any(marker in lower for marker in blocked_markers)


def _drop_reason(text: str) -> str | None:
    if not text.strip():
        return "empty_text"
    if _is_blocked_response(text):
        return "blocked_response"
    return None


def _diagnostic(
    candidate: CandidateSource,
    *,
    status: str,
    fetch_mode: str = "http",
    http_status: int | None = None,
    final_url: str | None = None,
    content_type: str | None = None,
    bytes: int = 0,
    text_length: int = 0,
    raw_path: str | None = None,
    drop_reason: str | None = None,
    error: str | None = None,
) -> SourceFetchDiagnostic:
    return SourceFetchDiagnostic(
        candidate_id=candidate.id,
        url=candidate.url,
        title=candidate.title,
        status=status,  # type: ignore[arg-type]
        fetch_mode=fetch_mode,
        http_status=http_status,
        final_url=final_url,
        content_type=content_type,
        bytes=bytes,
        text_length=text_length,
        raw_path=raw_path,
        drop_reason=drop_reason,
        error=error,
    )
