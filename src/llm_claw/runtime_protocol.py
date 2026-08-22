from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "1.0"
RuntimeStatus = Literal["planned", "running", "succeeded", "insufficient", "degraded", "failed", "interrupted", "skipped"]


class PermissionProfile(BaseModel):
    network_access: bool = False
    writable_roots: list[str] = Field(default_factory=list)
    readable_roots: list[str] = Field(default_factory=list)
    environment_allowlist: list[str] = Field(default_factory=list)
    mutates: bool = False
    approval_required: bool = False


class RuntimeCommand(BaseModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    command_id: str
    correlation_id: str
    decision_id: str
    run_id: str
    step_id: str
    engine: Literal["llm-claw", "llm-kg", "llm-kee"]
    operation: str
    idempotency_key: str
    input_hash: str
    deadline_at: str | None = None
    permission_profile: PermissionProfile
    input: dict[str, Any] = Field(default_factory=dict)


class RuntimeError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class RuntimeEvent(BaseModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}")
    sequence: int
    occurred_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    correlation_id: str
    decision_id: str
    run_id: str
    step_id: str
    engine: Literal["nox", "llm-claw", "llm-kg", "llm-kee", "human"]
    operation: str
    kind: Literal[
        "run.started", "run.completed", "run.failed",
        "step.started", "step.progress", "step.completed", "step.failed",
        "approval.requested", "approval.resolved", "artifact.created",
        "heartbeat", "migration.snapshot",
    ]
    status: RuntimeStatus
    references: dict[str, str | list[str]] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    error: RuntimeError | None = None


class OperationReceipt(BaseModel):
    idempotency_key: str
    operation: str
    input_hash: str
    status: RuntimeStatus
    output: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    completed_at: str | None = None


class RuntimeEmitter:
    def __init__(self, command: RuntimeCommand, stream) -> None:
        self.command = command
        self.stream = stream
        self.sequence = 0

    def emit(
        self,
        kind: str,
        status: RuntimeStatus,
        *,
        references: dict[str, str | list[str]] | None = None,
        payload: dict[str, Any] | None = None,
        error: RuntimeError | None = None,
    ) -> RuntimeEvent:
        self.sequence += 1
        event = RuntimeEvent(
            sequence=self.sequence,
            correlation_id=self.command.correlation_id,
            decision_id=self.command.decision_id,
            run_id=self.command.run_id,
            step_id=self.command.step_id,
            engine=self.command.engine,
            operation=self.command.operation,
            kind=kind,
            status=status,
            references=references or {},
            payload=payload or {},
            error=error,
        )
        self.stream.write(event.model_dump_json() + "\n")
        self.stream.flush()
        return event


class OperationReceiptStore:
    def __init__(self, workspace: Path, store_name: str) -> None:
        self.root = workspace / store_name / "operations"
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, idempotency_key: str) -> OperationReceipt | None:
        path = self._path(idempotency_key)
        if not path.exists():
            return None
        return OperationReceipt.model_validate_json(path.read_text(encoding="utf-8"))

    def begin(self, command: RuntimeCommand) -> OperationReceipt:
        existing = self.get(command.idempotency_key)
        if existing:
            if existing.input_hash != command.input_hash:
                raise ValueError("idempotency key was already used with a different input hash")
            return existing
        receipt = OperationReceipt(
            idempotency_key=command.idempotency_key,
            operation=command.operation,
            input_hash=command.input_hash,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )
        self._write(receipt)
        return receipt

    def complete(self, receipt: OperationReceipt, status: RuntimeStatus, output: dict[str, Any]) -> OperationReceipt:
        receipt.status = status
        receipt.output = output
        receipt.completed_at = datetime.now(UTC).isoformat()
        self._write(receipt)
        return receipt

    def interrupt(self, receipt: OperationReceipt, message: str) -> OperationReceipt:
        return self.complete(receipt, "interrupted", {"error": message})

    def _path(self, key: str) -> Path:
        return self.root / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def _write(self, receipt: OperationReceipt) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root, delete=False) as handle:
            handle.write(receipt.model_dump_json(indent=2))
            temporary = Path(handle.name)
        temporary.replace(self._path(receipt.idempotency_key))


def load_runtime_command(path: str | Path | None) -> RuntimeCommand | None:
    if not path:
        return None
    return RuntimeCommand.model_validate_json(Path(path).read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
