from pathlib import Path

from llm_claw.api import run_task
from llm_claw.models import AcquisitionTask
from llm_claw.pipeline import DataAcquisitionEngine
from llm_claw.config import Settings
from llm_claw.models import CandidateSource
from llm_claw.pipeline.engine import _dedupe_candidates


def test_pipeline_outputs_evidence_pack_from_mock_raw_sources(tmp_path: Path) -> None:
    task = AcquisitionTask.model_validate(
        {
            "entity": {"project_name": "Example Housing Project", "city": "San Jose", "address": "123 Main St"},
            "data_needed": ["planning status", "staff report", "public comments", "CEQA status"],
            "provider_policy": {"allowed_providers": ["mock", "search_api", "crawler", "claude"]},
        }
    )

    pack = DataAcquisitionEngine(Settings(workspace=tmp_path)).run(task)

    assert pack.request_id == task.id
    assert pack.candidate_sources
    assert pack.raw_sources
    assert pack.evidence
    assert pack.structured_data["approval_status"] == "under review"
    assert any(trace.provider == "crawler" for trace in pack.provider_trace)


def test_governance_final_evidence_requires_raw_source_fields(tmp_path: Path) -> None:
    pack = run_task(Path("examples/project_research.json"), workspace=tmp_path)

    assert pack.evidence
    for item in pack.evidence:
        assert item.source_url
        assert item.retrieved_at
        assert item.source_type
        assert item.evidence_text
        assert item.confidence >= 0
        assert "crawler" in item.verified_by

    llm_candidates = [candidate for candidate in pack.candidate_sources if candidate.provider == "mock"]
    assert llm_candidates
    assert all(candidate.snippet not in [item.evidence_text for item in pack.evidence] for candidate in llm_candidates)


def test_dedupe_candidates_collapses_ceqanet_mirrors() -> None:
    candidates = [
        CandidateSource(
            provider="crawler",
            title="LCI record",
            url="https://ceqanet.lci.ca.gov/2024120754",
        ),
        CandidateSource(
            provider="gemini",
            title="OPR record",
            url="https://ceqanet.opr.ca.gov/Project/2024120754",
        ),
        CandidateSource(
            provider="openai_web_search",
            title="OPR posting",
            url="https://ceqanet.opr.ca.gov/2024120754/2",
        ),
    ]

    assert _dedupe_candidates(candidates) == [candidates[0]]
