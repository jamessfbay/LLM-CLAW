from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_claw.api import run_task


PROTOCOL_VERSION = "3.5"
PROTOCOL_SCHEMA_HASH = "8763bd987a97f020f05a220167c21052c23aae47677862a91778aedb2773f242"
ENGINE_ID = "llm-claw"
MAX_REQUEST_BYTES = 8_000_000


class CommandV35(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["3.5"]
    command_id: str
    run_id: str
    attempt_id: str | None = None
    adapter_id: str
    capability: str
    operation: str
    input: Any
    input_hash: str
    idempotency_key: str
    fencing: dict[str, int]
    lease_token: int = Field(ge=0)
    deadline: int
    authorization_permit: dict[str, Any] | None = None


class EngineRpcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["3.5"]
    protocol_schema_hash: str
    engine: Literal["llm-claw", "llm-kg", "llm-kee"]
    command: CommandV35
    payload: Any = None
    model_gateway: dict[str, str] | None = None


class ReplyV35(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["3.5"]
    command_id: str
    input_hash: str
    lease_token: int
    output: Any
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


class EngineRpcResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["3.5"]
    protocol_schema_hash: str
    engine: Literal["llm-claw", "llm-kg", "llm-kee"]
    replayed: bool
    reply: ReplyV35


class RpcError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class ReceiptStore:
    def __init__(self, workspace: Path) -> None:
        self.root = workspace / ".llm_claw" / "rpc_operations"
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def claim(self, command: CommandV35):
        digest = hashlib.sha256(command.idempotency_key.encode("utf-8")).hexdigest()
        receipt_path = self.root / f"{digest}.json"
        lock = receipt_path.with_suffix(".lock").open("a+")
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RpcError(409, "operation_in_progress", "idempotent operation is already running") from exc
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("input_hash") != command.input_hash:
                    raise RpcError(409, "idempotency_conflict", "idempotency key is bound to another input hash")
                yield EngineRpcResponse.model_validate(receipt["response"]), None
                return

            def commit(response: EngineRpcResponse) -> None:
                record = {"input_hash": command.input_hash, "response": response.model_dump(mode="json")}
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root, delete=False) as handle:
                    json.dump(record, handle, sort_keys=True, separators=(",", ":"))
                    temporary = Path(handle.name)
                temporary.replace(receipt_path)

            yield None, commit
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


def invoke(request: EngineRpcRequest, workspace: Path) -> EngineRpcResponse:
    _validate_request(request)
    receipts = ReceiptStore(workspace)
    with receipts.claim(request.command) as (cached, commit):
        if cached is not None:
            return _response(request.command, cached.reply.output).model_copy(update={"replayed": True})
        pack = run_task(request.payload if request.payload is not None else request.command.input, workspace=workspace)
        output = pack.model_dump(mode="json")
        output["artifact_path"] = str(workspace / ".llm_claw" / "evidence_packs" / f"{pack.request_id}.json")
        response = _response(request.command, output)
        assert commit is not None
        commit(response)
        return response


def _validate_request(request: EngineRpcRequest) -> None:
    command = request.command
    if request.protocol_schema_hash != PROTOCOL_SCHEMA_HASH:
        raise RpcError(409, "schema_mismatch", "engine RPC schema hash does not match")
    if request.engine != ENGINE_ID or command.adapter_id not in {"claw", ENGINE_ID}:
        raise RpcError(400, "engine_mismatch", "request is addressed to another engine")
    if command.capability != "observer" or command.operation != "outcome":
        raise RpcError(400, "unsupported_operation", "CLAW RPC only accepts observer/outcome")
    if request.model_gateway is not None:
        raise RpcError(400, "gateway_rejected", "observer cannot receive model gateway credentials")
    if command.authorization_permit is not None:
        raise RpcError(400, "authority_rejected", "observer cannot receive an authorization permit")
    if command.deadline <= int(datetime.now(UTC).timestamp() * 1000):
        raise RpcError(408, "deadline_expired", "command deadline has expired")
    if len(command.input_hash) != 64 or any(char not in "0123456789abcdef" for char in command.input_hash):
        raise RpcError(400, "invalid_input_hash", "input hash must be lowercase sha256")


def _response(command: CommandV35, output: Any) -> EngineRpcResponse:
    return EngineRpcResponse(
        protocol_version=PROTOCOL_VERSION,
        protocol_schema_hash=PROTOCOL_SCHEMA_HASH,
        engine=ENGINE_ID,
        replayed=False,
        reply=ReplyV35(
            protocol_version=PROTOCOL_VERSION,
            command_id=command.command_id,
            input_hash=command.input_hash,
            lease_token=command.lease_token,
            output=output,
            metadata={"transport": "http", "engine": ENGINE_ID},
        ),
    )


def serve(host: str, port: int, workspace: Path, token: str, handler: Callable[[EngineRpcRequest, Path], EngineRpcResponse] = invoke) -> HTTPServer:
    if host != "127.0.0.1":
        raise ValueError("engine RPC must bind to 127.0.0.1")
    if len(token) < 32:
        raise ValueError("NOX_ENGINE_RPC_TOKEN must contain at least 32 characters")

    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if not self._authorized():
                return self._problem(401, "unauthorized", "valid bearer token required")
            if self.path != "/healthz":
                return self._problem(404, "not_found", "route not found")
            self._json(200, {"status": "ready", "engine": ENGINE_ID, "action_authority": False, "protocol_schema_hash": PROTOCOL_SCHEMA_HASH, "supported_protocol_versions": [PROTOCOL_VERSION]})

        def do_POST(self) -> None:
            if not self._authorized():
                return self._problem(401, "unauthorized", "valid bearer token required")
            if self.path != "/v3.5/invoke":
                return self._problem(404, "not_found", "route not found")
            try:
                length = int(self.headers.get("content-length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise RpcError(413, "request_size", "request body is empty or exceeds the limit")
                request = EngineRpcRequest.model_validate_json(self.rfile.read(length))
                self._json(200, handler(request, workspace).model_dump(mode="json"))
            except RpcError as exc:
                self._problem(exc.status, exc.code, str(exc))
            except Exception:
                self._problem(400, "invalid_request", "request validation or engine operation failed")

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _authorized(self) -> bool:
            return hmac.compare_digest(self.headers.get("authorization", ""), f"Bearer {token}")

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _problem(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"code": code, "message": message}})

    return HTTPServer((host, port), RequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-claw-rpc")
    parser.add_argument("--host", default=os.getenv("NOX_ENGINE_RPC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("NOX_ENGINE_RPC_PORT", "7401")))
    parser.add_argument("--workspace", type=Path, default=Path(os.getenv("LLM_CLAW_WORKSPACE", ".")))
    args = parser.parse_args()
    token = os.getenv("NOX_ENGINE_RPC_TOKEN", "")
    server = serve(args.host, args.port, args.workspace.expanduser().resolve(), token)
    server.serve_forever()


if __name__ == "__main__":
    main()
