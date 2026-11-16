from pathlib import Path

from llm_claw.providers.circuit_breaker import ProviderCircuitBreaker


def test_provider_circuit_opens_and_resets(tmp_path: Path) -> None:
    breaker = ProviderCircuitBreaker(tmp_path, failure_threshold=2, cooldown_seconds=120)
    breaker.record_failure("openai_web_search")
    assert breaker.is_open("openai_web_search") is False
    breaker.record_failure("openai_web_search")
    assert breaker.is_open("openai_web_search") is True
    breaker.record_success("openai_web_search")
    assert breaker.is_open("openai_web_search") is False
