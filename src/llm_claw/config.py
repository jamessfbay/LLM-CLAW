from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

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
    github_token: str | None = None
    github_max_queries: int = Field(default=3, ge=0, le=10)
    hacker_news_max_queries: int = Field(default=3, ge=0, le=10)
    stack_exchange_max_queries: int = Field(default=3, ge=0, le=10)
    aisa_api_key: str | None = None
    aisa_request_timeout_seconds: int = Field(default=30, ge=1, le=120)
    aisa_reddit_max_queries: int = Field(default=2, ge=0, le=10)
    aisa_tavily_max_queries: int = Field(default=2, ge=0, le=10)
    aisa_scholar_max_queries: int = Field(default=1, ge=0, le=5)
    aisa_extract_max_fallbacks: int = Field(default=3, ge=0, le=20)
    aisa_reddit_comments_max_fallbacks: int = Field(default=1, ge=0, le=5)
    aisa_rerank_max_candidates: int = Field(default=40, ge=0, le=100)
    aisa_similarweb_max_domains: int = Field(default=1, ge=0, le=5)
    provider_max_workers: int = 4
    source_user_agent: str = "llm-claw/0.1"
    source_allowed_hosts: list[str] = Field(default_factory=list)
    source_file_roots: list[Path] = Field(default_factory=list)
    source_max_bytes: int = Field(default=10_000_000, ge=1, le=100_000_000)

    @model_validator(mode="after")
    def default_file_root_to_workspace(self) -> "Settings":
        if not self.source_file_roots:
            self.source_file_roots = [self.workspace.resolve()]
        return self

    @classmethod
    def from_env(cls, workspace: Path | None = None) -> "Settings":
        resolved_workspace = workspace or Path(os.getenv("LLM_CLAW_WORKSPACE", Path.cwd()))
        _load_dotenv(resolved_workspace / ".env")
        allowlist = os.getenv("LLM_CLAW_PROVIDER_ALLOWLIST")
        providers = [item.strip() for item in allowlist.split(",") if item.strip()] if allowlist else None
        cache_dir = os.getenv("LLM_CLAW_CACHE_DIR")
        require_raw = os.getenv("LLM_CLAW_REQUIRE_RAW_SOURCE_FETCH", "true").lower() not in {"0", "false", "no"}
        allowed_hosts = [item.strip().lower() for item in os.getenv("LLM_CLAW_SOURCE_ALLOWED_HOSTS", "").split(",") if item.strip()]
        file_roots = [Path(item).expanduser().resolve() for item in os.getenv("LLM_CLAW_SOURCE_FILE_ROOTS", "").split(os.pathsep) if item]
        return cls(
            workspace=resolved_workspace,
            provider_allowlist=providers or ["mock", "search_api", "openai_web_search", "crawler", "claude"],
            require_raw_source_fetch=require_raw,
            cache_dir=Path(cache_dir) if cache_dir else resolved_workspace / ".llm_claw" / "cache",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("LLM_CLAW_OPENAI_MODEL", "gpt-5.4-mini"),
            gemini_model=os.getenv("LLM_CLAW_GEMINI_MODEL", "gemini-3.8-flash"),
            search_api_key=os.getenv("SEARCH_API_KEY"),
            github_token=os.getenv("GITHUB_TOKEN"),
            github_max_queries=int(os.getenv("LLM_CLAW_GITHUB_MAX_QUERIES", "3")),
            hacker_news_max_queries=int(os.getenv("LLM_CLAW_HACKER_NEWS_MAX_QUERIES", "3")),
            stack_exchange_max_queries=int(os.getenv("LLM_CLAW_STACK_EXCHANGE_MAX_QUERIES", "3")),
            aisa_api_key=os.getenv("AISA_API_KEY"),
            aisa_request_timeout_seconds=int(os.getenv("AISA_REQUEST_TIMEOUT_SECONDS", "30")),
            aisa_reddit_max_queries=int(os.getenv("LLM_CLAW_AISA_REDDIT_MAX_QUERIES", "2")),
            aisa_tavily_max_queries=int(os.getenv("LLM_CLAW_AISA_TAVILY_MAX_QUERIES", "2")),
            aisa_scholar_max_queries=int(os.getenv("LLM_CLAW_AISA_SCHOLAR_MAX_QUERIES", "1")),
            aisa_extract_max_fallbacks=int(os.getenv("LLM_CLAW_AISA_EXTRACT_MAX_FALLBACKS", "3")),
            aisa_reddit_comments_max_fallbacks=int(os.getenv("LLM_CLAW_AISA_REDDIT_COMMENTS_MAX_FALLBACKS", "1")),
            aisa_rerank_max_candidates=int(os.getenv("LLM_CLAW_AISA_RERANK_MAX_CANDIDATES", "40")),
            aisa_similarweb_max_domains=int(os.getenv("LLM_CLAW_AISA_SIMILARWEB_MAX_DOMAINS", "1")),
            provider_max_workers=int(os.getenv("LLM_CLAW_PROVIDER_MAX_WORKERS", "4")),
            source_user_agent=(
                os.getenv("CLAW_SOURCE_USER_AGENT")
                or os.getenv("NOX_SEC_USER_AGENT")
                or "llm-claw/0.1"
            ),
            source_allowed_hosts=allowed_hosts,
            source_file_roots=file_roots or [resolved_workspace.resolve()],
            source_max_bytes=int(os.getenv("LLM_CLAW_SOURCE_MAX_BYTES", "10000000")),
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
