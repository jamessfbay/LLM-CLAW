import json
from pathlib import Path

from llm_claw.api import export_for_llm_kg, run_task
from llm_claw.cli import main


def test_kg_export_maps_pack_to_documents_evidence_and_claims(tmp_path: Path) -> None:
    pack = run_task(Path("examples/project_research.json"), workspace=tmp_path)
    payload = export_for_llm_kg(pack)

    assert payload["format"] == "llm-kg-import"
    assert payload["contract_version"] == "evidence-import/2.0"
    assert payload["documents"]
    assert payload["evidence"]
    assert payload["claims"]
    assert payload["claims"][0]["evidence_ids"]
    assert payload["claims"][0]["review_state"] == "auto_accepted"
    assert payload["evidence"][0]["source_content_hash"]
    assert payload["evidence"][0]["quote_end"] > payload["evidence"][0]["quote_start"]


def test_kg_export_downgrades_tampered_quote_binding(tmp_path: Path) -> None:
    pack = run_task(Path("examples/project_research.json"), workspace=tmp_path)
    pack.evidence[0].quote_start = 0
    pack.evidence[0].quote_end = len(pack.evidence[0].evidence_text)

    payload = export_for_llm_kg(pack)

    assert payload["evidence"][0]["review_state"] == "pending_review"
    assert payload["claims"][0]["status"] == "uncertain"


def test_cli_run_writes_valid_evidence_pack(tmp_path: Path) -> None:
    output = tmp_path / "pack.json"

    main(["run", "examples/project_research.json", "--workspace", str(tmp_path), "--output", str(output)])

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["request_id"]
    assert data["evidence"]
    assert (tmp_path / ".llm_claw" / "evidence_packs" / f"{data['request_id']}.json").exists()


def test_cli_export_kg_writes_payload(tmp_path: Path) -> None:
    pack = run_task(Path("examples/project_research.json"), workspace=tmp_path)
    pack_path = tmp_path / "pack.json"
    kg_path = tmp_path / "kg.json"
    pack_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")

    main(["export-kg", str(pack_path), "--output", str(kg_path)])

    data = json.loads(kg_path.read_text(encoding="utf-8"))
    assert data["format"] == "llm-kg-import"


def test_cli_export_kg_persists_canonical_artifact(tmp_path: Path, capsys) -> None:
    pack = run_task(Path("examples/project_research.json"), workspace=tmp_path)
    pack_path = tmp_path / ".llm_claw" / "evidence_packs" / f"{pack.request_id}.json"

    main(["export-kg", str(pack_path), "--workspace", str(tmp_path)])

    data = json.loads(capsys.readouterr().out)
    artifact_path = Path(data["artifact_path"])
    assert artifact_path == tmp_path / ".llm_claw" / "kg_exports" / f"{pack.request_id}.json"
    assert artifact_path.exists()
