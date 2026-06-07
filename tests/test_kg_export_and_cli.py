import json
from pathlib import Path

from llm_claw.api import export_for_llm_kg, run_task
from llm_claw.cli import main


def test_kg_export_maps_pack_to_documents_evidence_and_claims(tmp_path: Path) -> None:
    pack = run_task(Path("examples/project_research.json"), workspace=tmp_path)
    payload = export_for_llm_kg(pack)

    assert payload["format"] == "llm-kg-import"
    assert payload["documents"]
    assert payload["evidence"]
    assert payload["claims"]
    assert payload["claims"][0]["evidence_ids"]
    assert payload["claims"][0]["review_state"] == "auto_accepted"


def test_cli_run_writes_valid_evidence_pack(tmp_path: Path) -> None:
    output = tmp_path / "pack.json"

    main(["run", "examples/project_research.json", "--workspace", str(tmp_path), "--output", str(output)])

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["request_id"]
    assert data["evidence"]


def test_cli_export_kg_writes_payload(tmp_path: Path) -> None:
    pack = run_task(Path("examples/project_research.json"), workspace=tmp_path)
    pack_path = tmp_path / "pack.json"
    kg_path = tmp_path / "kg.json"
    pack_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")

    main(["export-kg", str(pack_path), "--output", str(kg_path)])

    data = json.loads(kg_path.read_text(encoding="utf-8"))
    assert data["format"] == "llm-kg-import"
