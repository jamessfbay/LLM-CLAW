from pathlib import Path

from llm_claw.config import Settings
from llm_claw.models import AcquisitionTask, CandidateSource, ProviderTrace, RawSource
from llm_claw.pipeline import DataAcquisitionEngine
from llm_claw.providers.gemini import GeminiProvider


def test_gemini_youtube_analysis_updates_source_text_and_metadata(tmp_path: Path) -> None:
    task = AcquisitionTask.model_validate(
        {
            "entity": {
                "project_name": "156 California Avenue Mixed-Use Project",
                "city": "Palo Alto",
                "address": "156 California Ave, Palo Alto, CA 94306",
            },
            "data_needed": ["planning status", "public comments"],
            "question": "getting data from City Council Meeting - June 1, 2026 on youtube",
            "acquisition_instruction": "Find the specific video URL and summarize public comments.",
        }
    )
    source = RawSource(
        source_url="https://www.youtube.com/@cityofpaloalto/videos",
        source_title="City of Palo Alto YouTube videos",
        source_type="youtube",
        content_hash="hash",
        text="Official channel video listing page.",
    )
    raw_path = tmp_path / "youtube.html"
    raw_path.write_text(
        '"title":{"content":"City Council Meeting - June 1, 2026"}'
        ',"contentId":"Cczy-CGO8IE"',
        encoding="utf-8",
    )
    source.raw_path = str(raw_path)
    provider = GeminiProvider(Settings(workspace=tmp_path, gemini_api_key="test-key"))
    captured_prompt = {}

    def fake_generate_content_without_tools(prompt):  # type: ignore[no-untyped-def]
        captured_prompt["text"] = prompt
        return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "156 California Avenue Mixed-Use Project appears in a City Council meeting video. "
                                "The video discusses planning status and public comments."
                            )
                        }
                    ]
                }
            }
        ]
        }

    provider._generate_content_without_tools = fake_generate_content_without_tools  # type: ignore[method-assign]

    analyzed, trace = provider.analyze_youtube_source(task, source)

    assert analyzed is not None
    assert analyzed.source_type == "youtube"
    assert "156 California Avenue" in analyzed.text
    assert analyzed.metadata["analysis_provider"] == "gemini"
    assert trace.provider == "gemini"
    assert trace.status == "ok"
    assert "City Council Meeting - June 1, 2026" in captured_prompt["text"]
    assert "https://www.youtube.com/watch?v=Cczy-CGO8IE" in captured_prompt["text"]
    assert "Find the specific video URL" in captured_prompt["text"]
    assert "broader city construction" in captured_prompt["text"]
    assert "https://www.youtube.com/watch?v=Cczy-CGO8IE" in analyzed.text


def test_engine_routes_youtube_candidates_to_gemini_regardless_of_provider_policy(tmp_path: Path) -> None:
    task = AcquisitionTask.model_validate(
        {
            "entity": {
                "project_name": "156 California Avenue Mixed-Use Project",
                "city": "Palo Alto",
                "address": "156 California Ave, Palo Alto, CA 94306",
            },
            "data_needed": ["planning status"],
            "provider_policy": {"allowed_providers": ["crawler"]},
            "seed_sources": [
                {
                    "provider": "crawler",
                    "title": "City of Palo Alto YouTube streams",
                    "url": "https://www.youtube.com/@cityofpaloalto/streams",
                    "source_type": "youtube",
                    "is_official": True,
                },
                {
                    "provider": "crawler",
                    "title": "156 California planning page",
                    "url": "mock://planning-page",
                    "is_official": True,
                },
            ],
        }
    )
    engine = DataAcquisitionEngine(Settings(workspace=tmp_path, provider_allowlist=["crawler"]))
    fetcher = _RecordingFetcher()
    engine.fetcher = fetcher  # type: ignore[assignment]
    engine.youtube_analyzer = _FakeYoutubeAnalyzer()  # type: ignore[assignment]

    pack = engine.run(task)

    assert [candidate.url for candidate in fetcher.candidates] == ["mock://planning-page"]
    assert any(source.source_type == "youtube" for source in pack.raw_sources)
    assert any(trace.provider == "gemini" and trace.status == "ok" for trace in pack.provider_trace)
    youtube_evidence = [item for item in pack.evidence if item.source_type == "youtube"]
    assert youtube_evidence
    assert "gemini" in youtube_evidence[0].verified_by
    assert "crawler" not in youtube_evidence[0].verified_by
    assert any("city construction or planning" in item.claim for item in youtube_evidence)


def test_gemini_youtube_analysis_uses_no_web_search_tools(tmp_path: Path) -> None:
    task = AcquisitionTask.model_validate(
        {
            "entity": {
                "project_name": "156 California Avenue Mixed-Use Project",
                "city": "Palo Alto",
                "address": "156 California Ave, Palo Alto, CA 94306",
            },
            "data_needed": ["specific video URL"],
            "question": "getting data from City Council Meeting - June 1, 2026 on youtube",
            "acquisition_instruction": "Return the concrete video URL.",
        }
    )
    source = RawSource(
        source_url="https://www.youtube.com/@cityofpaloalto/streams",
        source_title="City of Palo Alto YouTube Streams",
        source_type="youtube",
        content_hash="hash",
        text="Official stream listing.",
    )
    provider = GeminiProvider(Settings(workspace=tmp_path, gemini_api_key="test-key"))
    prompts: list[str] = []
    used_web_search = False

    def fake_generate_content_without_tools(prompt):  # type: ignore[no-untyped-def]
        prompts.append(prompt)
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "Concrete URL: https://www.youtube.com/watch?v=Cczy-CGO8IE. "
                                    "Title: City Council Meeting - June 1, 2026."
                                )
                            }
                        ]
                    }
                }
            ]
        }

    def fail_if_web_search_is_used(prompt):  # type: ignore[no-untyped-def]
        nonlocal used_web_search
        used_web_search = True
        raise AssertionError("YouTube analysis should not use web search")

    provider._generate_content_without_tools = fake_generate_content_without_tools  # type: ignore[method-assign]
    provider._generate_content = fail_if_web_search_is_used  # type: ignore[method-assign]
    provider._generate_content_with_url_context = fail_if_web_search_is_used  # type: ignore[method-assign]

    analyzed, trace = provider.analyze_youtube_source(task, source)

    assert trace.status == "ok"
    assert analyzed is not None
    assert len(prompts) == 1
    assert not used_web_search
    assert "Do not use web search" in prompts[0]
    assert "https://www.youtube.com/watch?v=Cczy-CGO8IE" in analyzed.text


def test_gemini_youtube_analysis_passes_direct_watch_url_in_prompt(tmp_path: Path) -> None:
    task = AcquisitionTask.model_validate(
        {
            "entity": {
                "project_name": "156 California Avenue Mixed-Use Project",
                "city": "Palo Alto",
                "address": "156 California Ave, Palo Alto, CA 94306",
            },
            "data_needed": ["source-backed summary"],
            "question": "Analyze City Council Meeting video content.",
        }
    )
    source = RawSource(
        source_url="https://www.youtube.com/watch?v=Cczy-CGO8IE",
        source_title="City Council Meeting - June 1, 2026",
        source_type="youtube",
        content_hash="hash",
        text="Official video metadata.",
    )
    provider = GeminiProvider(Settings(workspace=tmp_path, gemini_api_key="test-key"))
    captured: dict[str, object] = {}

    def fake_generate_content_without_tools(prompt):  # type: ignore[no-untyped-def]
        captured["prompt"] = prompt
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "Video content summary: the meeting discusses citywide housing, zoning, "
                                    "transportation infrastructure, and public works."
                                )
                            }
                        ]
                    }
                }
            ]
        }

    provider._generate_content_without_tools = fake_generate_content_without_tools  # type: ignore[method-assign]

    analyzed, trace = provider.analyze_youtube_source(task, source)

    assert trace.status == "ok"
    assert analyzed is not None
    assert "https://www.youtube.com/watch?v=Cczy-CGO8IE" in str(captured["prompt"])
    assert analyzed.metadata["analysis_input"] == "youtube_text_prompt"
    assert "transportation infrastructure" in analyzed.text


def test_engine_routes_youtube_to_gemini_when_raw_fetch_is_disabled(tmp_path: Path) -> None:
    task = AcquisitionTask.model_validate(
        {
            "entity": {
                "project_name": "156 California Avenue Mixed-Use Project",
                "city": "Palo Alto",
                "address": "156 California Ave, Palo Alto, CA 94306",
            },
            "data_needed": ["planning status"],
            "source_policy": {"require_raw_source_fetch": False},
            "provider_policy": {"allowed_providers": ["crawler"]},
            "seed_sources": [
                {
                    "provider": "crawler",
                    "title": "City of Palo Alto YouTube streams",
                    "url": "https://www.youtube.com/@cityofpaloalto/streams",
                    "source_type": "youtube",
                    "is_official": True,
                }
            ],
        }
    )
    engine = DataAcquisitionEngine(
        Settings(workspace=tmp_path, provider_allowlist=["crawler"], require_raw_source_fetch=False)
    )
    fetcher = _RecordingFetcher()
    engine.fetcher = fetcher  # type: ignore[assignment]
    engine.youtube_analyzer = _FakeYoutubeAnalyzer()  # type: ignore[assignment]

    pack = engine.run(task)

    assert fetcher.candidates == []
    assert any(source.source_type == "youtube" for source in pack.raw_sources)
    assert any(item.source_type == "youtube" for item in pack.evidence)


class _RecordingFetcher:
    def __init__(self) -> None:
        self.candidates: list[CandidateSource] = []

    def fetch(self, candidates: list[CandidateSource]) -> list[RawSource]:
        self.candidates = candidates
        return [
            RawSource(
                candidate_id=candidate.id,
                source_url=candidate.url,
                source_title=candidate.title,
                publisher=candidate.publisher,
                source_type="official_html",
                content_hash="mock-hash",
                text=(
                    "156 California Avenue Mixed-Use Project planning status is under review "
                    "in the official planning page."
                ),
                raw_path=str(Path("/tmp/mock.html")),
            )
            for candidate in candidates
        ]


class _FakeYoutubeAnalyzer:
    def analyze_youtube_candidate(
        self, task: AcquisitionTask, candidate: CandidateSource
    ) -> tuple[RawSource, ProviderTrace]:
        return (
            RawSource(
                candidate_id=candidate.id,
                source_url=candidate.url,
                source_title="City Council Meeting - June 1, 2026",
                publisher="City of Palo Alto",
                source_type="youtube",
                content_hash="youtube-hash",
                text=(
                    "156 California Avenue Mixed-Use Project planning status is under review "
                    "in the City Council Meeting - June 1, 2026 YouTube video. "
                    "The meeting also discusses citywide housing, zoning, public works, and transportation infrastructure."
                ),
                metadata={"source_fetcher": "gemini", "analysis_provider": "gemini"},
            ),
            ProviderTrace(
                provider="gemini",
                status="ok",
                query=candidate.url,
                candidate_count=1,
                message="Gemini fetched and summarized YouTube source content.",
            ),
        )

    def analyze_youtube_source(
        self, task: AcquisitionTask, source: RawSource
    ) -> tuple[RawSource, ProviderTrace]:
        return (
            source,
            ProviderTrace(provider="gemini", status="ok", query=source.source_url, candidate_count=1),
        )
