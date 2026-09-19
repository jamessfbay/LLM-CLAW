from datetime import UTC, datetime
import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from llm_claw.rpc import (
    ENGINE_ID,
    PROTOCOL_SCHEMA_HASH,
    EngineRpcRequest,
    RpcError,
    _response,
    _validate_request,
    serve,
)


def rpc_request(**overrides):
    command = {
        "protocol_version": "3.5", "command_id": "cmd-1", "run_id": "run-1",
        "adapter_id": "claw", "capability": "observer", "operation": "outcome",
        "input": {}, "input_hash": "a" * 64, "idempotency_key": "once",
        "fencing": {}, "lease_token": 1,
        "deadline": int(datetime.now(UTC).timestamp() * 1000) + 60_000,
    }
    command.update(overrides.pop("command", {}))
    return EngineRpcRequest(protocol_version="3.5", protocol_schema_hash=PROTOCOL_SCHEMA_HASH, engine=ENGINE_ID, command=command, **overrides)


def test_v35_contract_rejects_legacy_and_authority():
    payload = rpc_request().model_dump(mode="json")
    payload["protocol_version"] = "1.0"
    with pytest.raises(Exception):
        EngineRpcRequest.model_validate(payload)
    with pytest.raises(RpcError, match="authorization permit"):
        _validate_request(rpc_request(command={"authorization_permit": {"id": "permit"}}))
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve("0.0.0.0", 0, Path("."), "x" * 32)


def test_authenticated_long_running_rpc(tmp_path: Path):
    token = "rpc-test-token-" + "x" * 32
    server = serve("127.0.0.1", 0, tmp_path, token, lambda request, _workspace: _response(request.command, {"ok": True}))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(base + "/healthz")
        assert unauthorized.value.code == 401
        health = Request(base + "/healthz", headers={"authorization": f"Bearer {token}"})
        assert json.load(urlopen(health))["supported_protocol_versions"] == ["3.5"]
        body = rpc_request().model_dump_json().encode()
        invoke = Request(base + "/v3.5/invoke", data=body, headers={"authorization": f"Bearer {token}", "content-type": "application/json"})
        result = json.load(urlopen(invoke))
        assert result["reply"]["output"] == {"ok": True}
    finally:
        server.shutdown()
        server.server_close()
