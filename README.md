# LLM-CLAW

LLM-CLAW is a Source Linked Data Acquisition Agent for evidence-backed agent systems. It discovers candidate sources, fetches raw HTML/PDF/government API records, extracts evidence, and exports auditable Evidence Packs for LLM-KG.

```text
data need
  -> query planning
  -> provider routing
  -> candidate source discovery
  -> raw source fetch
  -> content and evidence extraction
  -> normalization and confidence scoring
  -> Evidence Pack
```

The core governance rule is strict: LLM providers can create candidates, analysis notes, or verification notes, but final facts must be tied to raw fetched sources.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## CLI

```bash
llm-claw providers list
llm-claw task create examples/project_research.json
llm-claw run examples/project_research.json --output tmp/evidence_pack.json
llm-claw export-kg tmp/evidence_pack.json --workspace ../LLM-KG --output tmp/llm_kg_import.json
```

## Environment

- `LLM_CLAW_WORKSPACE`: workspace path; defaults to current directory.
- `LLM_CLAW_PROVIDER_ALLOWLIST`: comma-separated provider names.
- `LLM_CLAW_REQUIRE_RAW_SOURCE_FETCH`: defaults to `true`.
- `LLM_CLAW_CACHE_DIR`: defaults to `.llm_claw/cache`.
- `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `PERPLEXITY_API_KEY`, `OPENAI_API_KEY`, `SEARCH_API_KEY`: enable optional providers.

Copy `.env.example` to `.env` for local use. Do not commit `.env`; it is ignored because it contains API keys.

## Python API

```python
from llm_claw import run_task, export_for_llm_kg

pack = run_task("examples/project_research.json")
kg_payload = export_for_llm_kg(pack)
```
