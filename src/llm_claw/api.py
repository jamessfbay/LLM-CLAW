from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_claw.config import Settings
from llm_claw.exporters.kg import export_for_llm_kg as _export_for_llm_kg
from llm_claw.models import AcquisitionTask, EvidencePack, JsonLike
from llm_claw.pipeline import DataAcquisitionEngine


def create_task(payload: JsonLike, workspace: Path | None = None) -> AcquisitionTask:
    task = _load_task(payload)
    settings = Settings.from_env(workspace)
    task_dir = settings.workspace / ".llm_claw" / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"{task.id}.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
    return task


def run_task(task_id_or_payload: JsonLike, workspace: Path | None = None) -> EvidencePack:
    settings = Settings.from_env(workspace)
    task = _load_task_or_id(task_id_or_payload, settings.workspace)
    engine = DataAcquisitionEngine(settings)
    pack = engine.run(task)
    out_dir = settings.workspace / ".llm_claw" / "evidence_packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{pack.request_id}.json").write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    return pack


def export_for_llm_kg(evidence_pack: EvidencePack | dict[str, Any] | str | Path) -> dict[str, Any]:
    pack = _load_pack(evidence_pack)
    return _export_for_llm_kg(pack)


def _load_task_or_id(value: JsonLike, workspace: Path) -> AcquisitionTask:
    if isinstance(value, str) and not value.strip().startswith("{"):
        candidate = workspace / ".llm_claw" / "tasks" / f"{value}.json"
        if candidate.exists():
            return _load_task(candidate)
    return _load_task(value)


def _load_task(value: JsonLike) -> AcquisitionTask:
    if isinstance(value, AcquisitionTask):
        return value
    if isinstance(value, dict):
        return AcquisitionTask.model_validate(value)
    path = Path(value)
    data = json.loads(path.read_text(encoding="utf-8"))
    return AcquisitionTask.model_validate(data)


def _load_pack(value: EvidencePack | dict[str, Any] | str | Path) -> EvidencePack:
    if isinstance(value, EvidencePack):
        return value
    if isinstance(value, dict):
        return EvidencePack.model_validate(value)
    path = Path(value)
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvidencePack.model_validate(data)
