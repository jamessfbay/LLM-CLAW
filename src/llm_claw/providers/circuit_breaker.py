from __future__ import annotations

import json
import time
from pathlib import Path

from llm_claw.models import ProviderName


class ProviderCircuitBreaker:
    """Small durable circuit breaker scoped to one CLAW workspace."""

    def __init__(self, workspace: Path, failure_threshold: int = 3, cooldown_seconds: int = 120) -> None:
        self.path = workspace / ".llm_claw" / "provider_circuits.json"
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(1, cooldown_seconds)

    def is_open(self, provider: ProviderName) -> bool:
        state = self._read()
        entry = state.get(provider, {})
        opened_at = float(entry.get("opened_at") or 0)
        if not opened_at:
            return False
        if time.time() - opened_at < self.cooldown_seconds:
            return True
        state.pop(provider, None)
        self._write(state)
        return False

    def record_success(self, provider: ProviderName) -> None:
        state = self._read()
        if provider in state:
            state.pop(provider, None)
            self._write(state)

    def record_failure(self, provider: ProviderName) -> None:
        state = self._read()
        entry = state.get(provider, {})
        failures = int(entry.get("consecutive_failures") or 0) + 1
        state[provider] = {
            "consecutive_failures": failures,
            "opened_at": time.time() if failures >= self.failure_threshold else None,
        }
        self._write(state)

    def _read(self) -> dict[str, dict[str, float | int | None]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write(self, state: dict[str, dict[str, float | int | None]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
