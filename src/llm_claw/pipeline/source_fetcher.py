from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from llm_claw.config import Settings
from llm_claw.models import CandidateSource, RawSource


class SourceFetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.raw_dir = settings.workspace / ".llm_claw" / "raw_sources"

    def fetch(self, candidates: list[CandidateSource]) -> list[RawSource]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        sources: list[RawSource] = []
        for candidate in candidates:
            source = self._fetch_one(candidate)
            if source:
                sources.append(source)
        return sources

    def _fetch_one(self, candidate: CandidateSource) -> RawSource | None:
        if candidate.url.startswith("mock://"):
            return self._mock_source(candidate)
        if candidate.url.startswith("file://"):
            return self._file_source(candidate)
        if candidate.url.startswith("http://") or candidate.url.startswith("https://"):
            return self._http_source(candidate)
        return None

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

    def _file_source(self, candidate: CandidateSource) -> RawSource | None:
        parsed = urlparse(candidate.url)
        path = Path(parsed.path)
        if not path.exists():
            return None
        data = path.read_bytes()
        digest = _hash(data)
        copy_path = self.raw_dir / f"{digest}{path.suffix.lower()}"
        copy_path.write_bytes(data)
        text = _extract_file_text(path, data)
        if _is_blocked_response(text):
            return None
        source_type = "local_pdf" if path.suffix.lower() == ".pdf" else "local_html"
        return RawSource(
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

    def _http_source(self, candidate: CandidateSource) -> RawSource | None:
        request = Request(candidate.url, headers={"User-Agent": "llm-claw/0.1"})
        try:
            with urlopen(request, timeout=15) as response:
                data = response.read()
                content_type = response.headers.get("content-type", "")
        except Exception:
            fetched = _curl_fetch(candidate.url)
            if not fetched:
                return None
            data, content_type = fetched
        digest = _hash(data)
        suffix = ".pdf" if "pdf" in content_type or candidate.url.lower().endswith(".pdf") else ".html"
        path = self.raw_dir / f"{digest}{suffix}"
        path.write_bytes(data)
        text = _extract_file_text(path, data)
        if not text.strip() or _is_blocked_response(text):
            return None
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
        return RawSource(
            candidate_id=candidate.id,
            source_url=candidate.url,
            source_title=candidate.title,
            publisher=candidate.publisher,
            source_type=source_type,
            content_hash=digest,
            raw_path=str(path),
            text=text,
            metadata={"content_type": content_type},
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
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
