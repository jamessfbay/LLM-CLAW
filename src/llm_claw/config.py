from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from llm_claw.models import ProviderName


class Settings(BaseModel):
    workspace: Path = Field(default_factory=lambda: Path.cwd())
    provider_allowlist: list[ProviderName] = Field(
        default_factory=lambda: ["mock", "search_api", "openai_web_search", "crawler", "claude"]
    )
    require_raw_source_fetch: bool = True
    cache_dir: Path | None = None

    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    perplexity_api_key: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    gemini_model: str = "gemini-3.5-flash"
    search_api_key: str | None = None
    provider_max_workers: int = 4

    @classmethod
    def from_env(cls, workspace: Path | None = None) -> "Settings":
        resolved_workspace = workspace or Path(os.getenv("LLM_CLAW_WORKSPACE", Path.cwd()))
        _load_dotenv(resolved_workspace / ".env")
        allowlist = os.getenv("LLM_CLAW_PROVIDER_ALLOWLIST")
        providers = [item.strip() for item in allowlist.split(",") if item.strip()] if allowlist else None
        cache_dir = os.getenv("LLM_CLAW_CACHE_DIR")
        require_raw = os.getenv("LLM_CLAW_REQUIRE_RAW_SOURCE_FETCH", "true").lower() not in {"0", "false", "no"}
        return cls(
            workspace=resolved_workspace,
            provider_allowlist=providers or ["mock", "search_api", "openai_web_search", "crawler", "claude"],
            require_raw_source_fetch=require_raw,
            cache_dir=Path(cache_dir) if cache_dir else resolved_workspace / ".llm_claw" / "cache",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("LLM_CLAW_OPENAI_MODEL", "gpt-4.1-mini"),
            gemini_model=os.getenv("LLM_CLAW_GEMINI_MODEL", "gemini-3.5-flash"),
            search_api_key=os.getenv("SEARCH_API_KEY"),
            provider_max_workers=int(os.getenv("LLM_CLAW_PROVIDER_MAX_WORKERS", "4")),
        )

    def provider_enabled(self, provider: ProviderName) -> bool:
        return provider in self.provider_allowlist


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
