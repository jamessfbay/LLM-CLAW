import json
from io import StringIO
from pathlib import Path

import pytest

from llm_claw.runtime_protocol import (
    OperationReceiptStore,
    RuntimeCommand,
    RuntimeEmitter,
)


def test_runtime_fixture_and_ndjson_event(tmp_path: Path):
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    command = RuntimeCommand.model_validate(payload)
    stream = StringIO()
    event = RuntimeEmitter(command, stream).emit("step.started", "running", payload={"phase": "test"})
    assert event.protocol_version == "1.0"
    emitted = json.loads(stream.getvalue())
    assert emitted["run_id"] == "run_contract"
    assert emitted["tenant_context"] == {"organization_id": "local", "workspace_id": "default"}


def test_runtime_event_propagates_explicit_tenant_context():
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    payload["tenant_context"] = {"organization_id": "org_1", "workspace_id": "workspace_1", "actor_id": "user_1"}
    command = RuntimeCommand.model_validate(payload)
    stream = StringIO()

    RuntimeEmitter(command, stream).emit("step.started", "running")

    assert json.loads(stream.getvalue())["tenant_context"] == payload["tenant_context"]


def test_operation_receipt_is_idempotent_and_rejects_hash_conflict(tmp_path: Path):
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    command = RuntimeCommand.model_validate(payload)
    store = OperationReceiptStore(tmp_path, ".llm_claw")
    receipt = store.begin(command)
    store.complete(receipt, "succeeded", {"artifact_path": "/tmp/result.json"})
    assert store.begin(command).output["artifact_path"] == "/tmp/result.json"

    conflicting = command.model_copy(update={"input_hash": "different"})
    with pytest.raises(ValueError, match="different input hash"):
        store.begin(conflicting)
