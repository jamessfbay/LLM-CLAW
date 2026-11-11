from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from llm_claw.config import Settings


class AIsaClient:
    base_url = "https://api.aisa.one"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def post(self, path: str, payload: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, payload=payload, params=params)

    def _request(
        self, method: str, path: str, *, payload: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self.settings.aisa_api_key:
            raise ValueError("AISA_API_KEY is not configured")
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("AIsa API path must be absolute and local to the configured origin")
        query = f"?{urlencode(params, doseq=True)}" if params else ""
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.base_url}{path}{query}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.aisa_api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        with urlopen(request, timeout=self.settings.aisa_request_timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("AIsa response is not an object")
        return parsed
