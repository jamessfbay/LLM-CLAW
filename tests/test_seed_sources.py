from pathlib import Path

from llm_claw.models import AcquisitionTask
from llm_claw.pipeline import DataAcquisitionEngine
from llm_claw.config import Settings


def test_pipeline_fetches_seed_sources_before_provider_discovery(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "staff_report.html"
    task = AcquisitionTask.model_validate(
        {
            "entity": {"project_name": "156 California Avenue", "city": "Palo Alto"},
            "data_needed": ["planning status"],
            "provider_policy": {"allowed_providers": ["crawler", "claude"]},
            "seed_sources": [
                {
                    "provider": "crawler",
                    "title": "Planning Commission Staff Report",
                    "url": fixture.resolve().as_uri(),
                    "publisher": "City of Palo Alto",
                    "is_official": True,
                }
            ],
        }
    )

    pack = DataAcquisitionEngine(Settings(workspace=tmp_path, provider_allowlist=["crawler", "claude"])).run(task)

    assert len(pack.candidate_sources) == 1
    assert len(pack.raw_sources) == 1
    assert pack.evidence
