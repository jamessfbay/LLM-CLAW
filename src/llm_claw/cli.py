from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from llm_claw.api import create_task, export_for_llm_kg, run_task
from llm_claw.config import Settings
from llm_claw.merge import merge_evidence_packs
from llm_claw.models import EvidencePack
from llm_claw.providers import list_provider_statuses
from llm_claw.runtime_protocol import (
    OperationReceiptStore,
    RuntimeEmitter,
    RuntimeError,
    load_runtime_command,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm-claw")
    subparsers = parser.add_subparsers(dest="command", required=True)

    providers_parser = subparsers.add_parser("providers")
    providers_sub = providers_parser.add_subparsers(dest="providers_command", required=True)
    providers_sub.add_parser("list")

    task_parser = subparsers.add_parser("task")
    task_sub = task_parser.add_subparsers(dest="task_command", required=True)
    task_create = task_sub.add_parser("create")
    task_create.add_argument("task_json")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("task_json")
    run_parser.add_argument("--output", "-o")
    run_parser.add_argument("--workspace")
    _add_runtime_arguments(run_parser)

    export_parser = subparsers.add_parser("export-kg")
    export_parser.add_argument("evidence_pack_json")
    export_parser.add_argument("--workspace")
    export_parser.add_argument("--output", "-o")
    _add_runtime_arguments(export_parser)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("base_pack_json")
    merge_parser.add_argument("other_pack_json", nargs="+")
    merge_parser.add_argument("--output", "-o")

    args = parser.parse_args(argv)

    if args.command == "providers":
        settings = Settings.from_env()
        _print_json(list_provider_statuses(settings))
        return

    if args.command == "task":
        task = create_task(args.task_json)
        _print_json(task.model_dump(mode="json"))
        return

    if args.command == "run":
        workspace = Path(args.workspace) if args.workspace else None
        if args.event_stream:
            return _run_streamed(args, workspace or Path.cwd())
        pack = run_task(args.task_json, workspace=workspace)
        payload = pack.model_dump(mode="json")
        if workspace is not None:
            payload["artifact_path"] = str(
                workspace.resolve() / ".llm_claw" / "evidence_packs" / f"{pack.request_id}.json"
            )
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)
        return

    if args.command == "export-kg":
        workspace = Path(args.workspace) if args.workspace else None
        if args.event_stream:
            return _export_streamed(args, workspace or Path.cwd())
        payload = export_for_llm_kg(args.evidence_pack_json, workspace=workspace)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)
        return

    if args.command == "merge":
        base = _load_pack(args.base_pack_json)
        others = [_load_pack(path) for path in args.other_pack_json]
        merged = merge_evidence_packs(base, *others)
        text = merged.model_dump_json(indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)
        return


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--event-stream", action="store_true", help="Emit RuntimeEvent v1 NDJSON")
    parser.add_argument("--runtime-context", help="Path to a RuntimeCommand v1 JSON file")
    parser.add_argument("--idempotency-key", help="Override the RuntimeCommand idempotency key")


def _runtime(args, workspace: Path):
    command = load_runtime_command(args.runtime_context)
    if command is None:
        raise ValueError("--runtime-context is required with --event-stream")
    if args.idempotency_key:
        command.idempotency_key = args.idempotency_key
    if command.engine != "llm-claw":
        raise ValueError(f"Runtime command engine mismatch: {command.engine}")
    return command, RuntimeEmitter(command, sys.stdout), OperationReceiptStore(workspace.resolve(), ".llm_claw")


def _run_streamed(args, workspace: Path) -> None:
    command, emitter, receipts = _runtime(args, workspace)
    receipt = receipts.begin(command)
    if receipt.status in {"succeeded", "insufficient"}:
        emitter.emit("step.completed", receipt.status, references=_string_references(receipt.output), payload={"replayed": True})
        return
    emitter.emit("step.started", "running", payload={"phase": "acquisition"})
    try:
        pack = run_task(args.task_json, workspace=workspace)
        pack.correlation_id = command.correlation_id
        pack.decision_id = command.decision_id
        pack.run_id = command.run_id
        pack.step_id = command.step_id
        pack.input_hash = command.input_hash
        artifact = workspace.resolve() / ".llm_claw" / "evidence_packs" / f"{pack.request_id}.json"
        artifact.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
        status = "succeeded" if pack.evidence else "insufficient"
        output = {
            "request_id": pack.request_id,
            "artifact_path": str(artifact),
            "evidence_count": len(pack.evidence),
            "fetch_attempt_ids": [item.id for item in pack.source_fetch_diagnostics],
            "raw_source_paths": [item.raw_path for item in pack.raw_sources if item.raw_path],
        }
        receipts.complete(receipt, status, output)
        emitter.emit("artifact.created", status, references=_string_references(output), payload={"evidence_count": len(pack.evidence)})
        emitter.emit("step.completed", status, references=_string_references(output), payload={"evidence_count": len(pack.evidence), "missing_data_count": len(pack.missing_data)})
    except KeyboardInterrupt:
        receipts.interrupt(receipt, "CLAW acquisition interrupted")
        emitter.emit("step.failed", "interrupted", error=RuntimeError(code="interrupted", message="CLAW acquisition interrupted", retryable=True))
    except Exception as exc:
        receipts.complete(receipt, "failed", {"error": str(exc)})
        emitter.emit("step.failed", "failed", error=RuntimeError(code="claw_failed", message=str(exc), retryable=True))
        raise


def _export_streamed(args, workspace: Path) -> None:
    command, emitter, receipts = _runtime(args, workspace)
    receipt = receipts.begin(command)
    if receipt.status == "succeeded":
        emitter.emit("step.completed", "succeeded", references=_string_references(receipt.output), payload={"replayed": True})
        return
    emitter.emit("step.started", "running", payload={"phase": "kg_export"})
    try:
        payload = export_for_llm_kg(args.evidence_pack_json, workspace=workspace)
        output = {
            "artifact_path": str(payload.get("artifact_path") or ""),
            "request_id": str(payload.get("request_id") or ""),
            "document_count": len(payload.get("documents") or []),
            "claim_count": len(payload.get("claims") or []),
            "evidence_count": len(payload.get("evidence") or []),
        }
        receipts.complete(receipt, "succeeded", output)
        emitter.emit("artifact.created", "succeeded", references=_string_references(output), payload={key: value for key, value in output.items() if isinstance(value, int)})
        emitter.emit("step.completed", "succeeded", references=_string_references(output))
    except Exception as exc:
        receipts.complete(receipt, "failed", {"error": str(exc)})
        emitter.emit("step.failed", "failed", error=RuntimeError(code="claw_export_failed", message=str(exc), retryable=True))
        raise


def _string_references(output: dict) -> dict[str, str | list[str]]:
    return {key: value for key, value in output.items() if isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))}


def _load_pack(path: str) -> EvidencePack:
    return EvidencePack.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
