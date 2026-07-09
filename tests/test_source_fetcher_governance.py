from llm_claw.pipeline.source_fetcher import _drop_reason, _extract_file_text, _is_blocked_response


def test_pdf_url_returning_html_access_denied_is_detected() -> None:
    data = b"<HTML><body>Access Denied You don't have permission to access this server.</body></HTML>"

    text = _extract_file_text(type("PathLike", (), {"suffix": ".pdf"})(), data)

    assert _is_blocked_response(text)


def test_sec_rate_limit_page_is_classified() -> None:
    text = "SEC.gov | Request Rate Threshold Exceeded Automated access to our sites must comply with SEC.gov policy."

    assert _drop_reason(text) == "rate_limited"


def test_faa_404_template_is_classified_from_raw_html() -> None:
    raw = b'<html data-headerstatus="404"><title>Federal Aviation Administration</title></html>'

    assert _drop_reason("Federal Aviation Administration", raw_data=raw) == "http_404"
