from llm_claw.models import AcquisitionTask, EvidenceItem


def test_generic_task_requires_explicit_observation_needs() -> None:
    try:
        AcquisitionTask(entity={"name": "service"})
    except ValueError as exc:
        assert "data_needed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("generic acquisition must not invent domain observations")


def test_evidence_is_decision_grade_only_when_bound_to_raw_content() -> None:
    evidence = EvidenceItem(
        claim="Latency is 120 ms.",
        source_title="Telemetry",
        source_url="https://example.test/telemetry",
        source_type="government_api",
        evidence_text="Latency is 120 ms.",
        retrieved_at="2026-09-10T00:00:00Z",
        confidence=0.9,
        source_id="source-1",
        raw_content_hash="abc",
        quote_start=4,
        quote_end=22,
        quote_match="exact",
    )

    assert evidence.contract_version == "evidence-artifact/2.0"
    assert evidence.decision_grade is True
