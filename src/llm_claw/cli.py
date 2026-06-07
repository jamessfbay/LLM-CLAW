from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_claw.api import create_task, export_for_llm_kg, run_task
from llm_claw.config import Settings
from llm_claw.merge import merge_evidence_packs
from llm_claw.models import EvidencePack
from llm_claw.providers import list_provider_statuses


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

    export_parser = subparsers.add_parser("export-kg")
    export_parser.add_argument("evidence_pack_json")
    export_parser.add_argument("--workspace")
    export_parser.add_argument("--output", "-o")

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
        pack = run_task(args.task_json, workspace=workspace)
        text = pack.model_dump_json(indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)
        return

    if args.command == "export-kg":
        payload = export_for_llm_kg(args.evidence_pack_json)
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


def _load_pack(path: str) -> EvidencePack:
    return EvidencePack.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
