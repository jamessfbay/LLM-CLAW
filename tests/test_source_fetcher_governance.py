from llm_claw.pipeline.source_fetcher import _extract_file_text, _is_blocked_response


def test_pdf_url_returning_html_access_denied_is_detected() -> None:
    data = b"<HTML><body>Access Denied You don't have permission to access this server.</body></HTML>"

    text = _extract_file_text(type("PathLike", (), {"suffix": ".pdf"})(), data)

    assert _is_blocked_response(text)
