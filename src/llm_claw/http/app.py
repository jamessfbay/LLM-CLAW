from __future__ import annotations

from pathlib import Path

from llm_claw.api import create_task, run_task
from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask


def create_app():
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install llm-claw[http] to use the HTTP API.") from exc

    app = FastAPI(title="LLM-CLAW")

    @app.post("/data-acquisition/tasks")
    def create_data_acquisition_task(payload: AcquisitionTask):
        task = create_task(payload)
        return {"task_id": task.id, "status": "created"}

    @app.get("/data-acquisition/tasks/{task_id}")
    def get_data_acquisition_task(task_id: str):
        settings = Settings.from_env()
        path = settings.workspace / ".llm_claw" / "tasks" / f"{task_id}.json"
        if not path.exists():
            return {"task_id": task_id, "status": "missing"}
        pack_path = settings.workspace / ".llm_claw" / "evidence_packs" / f"{task_id}.json"
        return {"task_id": task_id, "status": "completed" if pack_path.exists() else "created"}

    @app.get("/data-acquisition/tasks/{task_id}/evidence-pack")
    def get_evidence_pack(task_id: str):
        settings = Settings.from_env()
        path = settings.workspace / ".llm_claw" / "evidence_packs" / f"{task_id}.json"
        if not path.exists():
            pack = run_task(task_id)
            return pack.model_dump(mode="json")
        return path.read_text(encoding="utf-8")

    return app
