"""Offline tests for the ingest_file orchestration.

MinIO, ArangoDB and the vision endpoint are all replaced by fakes, while the
tree helpers (``set_parent_ids``, ``validate_tree_and_attach_source``, ...) stay
real so the produced tree is the one production would produce.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from arango_client import DataSource, SourceType

from config.settings import settings
from workflows.ingest_file import run as run_module
from workflows.ingest_file.models import FileIngestionInput, FileIngestionStatus
from workflows.ingest_file.vision_ocr import (
    PageExtraction,
    VisionBlock,
    VisionOCRConfig,
    VisionOCRError,
    VisionPage,
)

PDF_BYTES = b"%PDF-1.4 fake document"
PDF_HASH = hashlib.md5(PDF_BYTES).hexdigest()

PARAGRAPH = (
    "Chuyển động của một vật là sự thay đổi vị trí của vật đó so với các vật khác "
    "theo thời gian, và được mô tả bằng quãng đường đi được trong mỗi khoảng thời gian."
)


class _FakeStorage:
    """Writes the stored bytes to the requested destination."""

    def __init__(self, payload: bytes = PDF_BYTES) -> None:
        self.payload = payload
        self.downloads: list[tuple[str, str]] = []

    async def download_file(
        self, *, bucket_name: str, object_name: str, destination: Path
    ) -> None:
        self.downloads.append((bucket_name, object_name))
        Path(destination).write_bytes(self.payload)


class _FakeSources:
    """In-memory stand-in for DataSourceHandler."""

    def __init__(self, sources: dict[UUID, DataSource]) -> None:
        self._sources = sources

    def get_by_id(self, source_id: UUID | str) -> DataSource | None:
        return self._sources.get(UUID(str(source_id)))

    def update(self, source_id: UUID | str, patch: dict[str, Any]) -> DataSource:
        key = UUID(str(source_id))
        updated = self._sources[key].model_copy(update=patch)
        self._sources[key] = updated
        return updated


class _FakeSystemAdminHandler:
    """Returns the fake source handler regardless of the database handle."""

    sources: _FakeSources

    def __init__(self, db: Any) -> None:
        self.db = db


class _FakeVisionClient:
    """Returns canned pages and records the state it was asked with.

    ``spend`` is what a failing page's requests were billed: it is handed to the
    raised :class:`VisionOCRError` exactly as the real client hands over what the
    endpoint answered with. ``unmeasured_pages`` are the pages the endpoint
    answered without stating what they cost.
    """

    def __init__(
        self,
        pages: dict[int, VisionPage],
        *,
        fail_on: int | None = None,
        spend: dict[str, Any] | None = None,
        cached_tokens: int | None = 4,
        unmeasured_pages: frozenset[int] = frozenset(),
    ) -> None:
        self.config = VisionOCRConfig(
            api_base="https://fake-vision.invalid/v1/chat/completions",
            model="fake-vision",
            api_key="super-secret",
            allowed_hosts=("fake-vision.invalid",),
        )
        self._pages = pages
        self._fail_on = fail_on
        self._spend = spend or {}
        self._cached_tokens = cached_tokens
        self._unmeasured = unmeasured_pages
        self.states: list[dict[str, Any]] = []
        self.closed = False

    async def extract_page(
        self, png_bytes: bytes, *, page_number: int, state: dict[str, Any]
    ) -> PageExtraction:
        self.states.append(dict(state))
        if page_number == self._fail_on:
            raise VisionOCRError(
                f"page {page_number}: HTTP 500 (after 3 attempt(s))", **self._spend
            )
        measured = page_number not in self._unmeasured
        return PageExtraction(
            page=self._pages[page_number],
            attempts=1,
            prompt_tokens=5 if measured else None,
            completion_tokens=3 if measured else None,
            cached_tokens=self._cached_tokens if measured else None,
        )

    async def aclose(self) -> None:
        self.closed = True


class _FakeArangoClient:
    db = object()


def _page(page: int, *blocks: VisionBlock) -> VisionPage:
    return VisionPage(page=page, blocks=list(blocks))


def _make_source(**overrides: Any) -> DataSource:
    config = {
        "minio_bucket": "ingest",
        "minio_path": "raw/book.pdf",
        "original_filename": "book.pdf",
    }
    defaults: dict[str, Any] = {
        "id": uuid4(),
        "name": "Vật lí 12",
        "type": SourceType.PDF,
        "config": config,
        "hash_content": PDF_HASH,
        "status": "processing",
    }
    defaults.update(overrides)
    return DataSource(**defaults)


@pytest.fixture
def sources() -> dict[UUID, DataSource]:
    return {}


@pytest.fixture
def handler(monkeypatch: pytest.MonkeyPatch, sources: dict[UUID, DataSource]) -> _FakeSources:
    fake_sources = _FakeSources(sources)

    def _build(db: Any) -> _FakeSystemAdminHandler:
        instance = _FakeSystemAdminHandler(db)
        instance.sources = fake_sources
        return instance

    monkeypatch.setattr(run_module, "SystemAdminHandler", _build)
    return fake_sources


class _FakeRenderer:
    """Yields one canned page per requested page number and records the call."""

    def __init__(self, page_numbers: list[int]) -> None:
        self.page_numbers = page_numbers
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, path: Path, *, dpi: int, page_offset: int):
        self.calls.append((path.name, dpi, page_offset))
        for number in self.page_numbers:
            yield number, b"png-bytes"


def _patch_renderer(monkeypatch: pytest.MonkeyPatch, page_numbers: list[int]) -> _FakeRenderer:
    renderer = _FakeRenderer(page_numbers)
    monkeypatch.setattr(run_module, "render_pdf_pages", renderer)
    return renderer


async def _run_workflow(
    *,
    source: DataSource,
    vision_client: _FakeVisionClient,
    storage: _FakeStorage,
    tmp_path: Path,
) -> Any:
    workflow = run_module.FileIngestionWorkflow(
        storage=storage,
        workspace_root=tmp_path,
        vision_client=vision_client,
        arango_client=_FakeArangoClient(),
    )
    return await workflow.run(FileIngestionInput(data_source_id=str(source.id)))


class TestSuccessfulIngestion:
    """A clean run builds the tree, reports statistics and activates the source."""

    async def test_tree_and_statistics(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source()
        sources[source.id] = source
        renderer = _patch_renderer(monkeypatch, [1, 2])
        vision = _FakeVisionClient(
            {
                1: _page(
                    1,
                    VisionBlock(kind="heading", marker="1", depth=1, group=1, text="1 Chương một"),
                    VisionBlock(kind="p", group=2, text=PARAGRAPH),
                ),
                2: _page(2, VisionBlock(kind="list", group=1, text="- Ý một\n- Ý hai")),
            }
        )
        storage = _FakeStorage()

        result = await _run_workflow(
            source=source, vision_client=vision, storage=storage, tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.COMPLETE
        assert result.data_source.status == "active"
        assert storage.downloads == [("ingest", "raw/book.pdf")]
        assert renderer.calls == [("book.pdf", settings.VISION_DPI, settings.VISION_PAGE_OFFSET)]

        assert result.statistics["pages_processed"] == 2
        assert result.statistics["ocr_attempts"] == 2
        assert result.statistics["ocr_prompt_tokens"] == 10
        assert result.statistics["ocr_completion_tokens"] == 6
        assert result.statistics["ocr_cached_tokens"] == 8
        assert result.statistics["blocks_by_kind"] == {"heading": 1, "p": 1, "list": 1}
        assert result.statistics["vision_model"] == "fake-vision"
        assert result.statistics["vision_endpoint_host"] == "fake-vision.invalid"

        assert result.root_node["attributes"]["title"] == "Vật lí 12"
        assert result.root_node["source_id"] == str(source.id)
        chapter = result.root_node["children"][0]
        assert chapter["attributes"]["title"] == "1 Chương một"
        assert [child["attributes"]["text"] for child in chapter["children"]] == [
            PARAGRAPH,
            "- Ý một\n- Ý hai",
        ]

    async def test_open_headings_are_passed_to_the_next_page(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1, 2])
        vision = _FakeVisionClient(
            {
                1: _page(
                    1,
                    VisionBlock(kind="heading", marker="1", depth=1, group=1, text="1 Chương một"),
                ),
                2: _page(2, VisionBlock(kind="p", group=2, text="Đoạn hai")),
            }
        )

        await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert vision.states[0] == {
            "open_headings": [],
            "open_group": None,
            "open_group_tail": "",
        }
        assert vision.states[1] == {
            "open_headings": [{"marker": "1", "depth": 1, "text": "1 Chương một"}],
            "open_group": None,
            "open_group_tail": "",
        }
        assert vision.closed

    async def test_an_unreported_cache_share_stays_unknown(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The endpoint said nothing about the cache, so the run reports no share
        # rather than a zero that would claim every prompt token was billed full.
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1])
        vision = _FakeVisionClient(
            {1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))}, cached_tokens=None
        )

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.statistics["ocr_cached_tokens"] is None

    async def test_a_page_that_reported_nothing_makes_the_run_totals_unknown(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # One page was measured and one was not. The measured page's counts
        # cannot stand in for the run's bill, so the totals go unknown while the
        # attempts stay measured: the requests were made either way.
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1, 2])
        vision = _FakeVisionClient(
            {
                1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn một")),
                2: _page(2, VisionBlock(kind="p", group=1, text="Đoạn hai")),
            },
            unmeasured_pages=frozenset({2}),
        )

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.COMPLETE
        assert result.statistics["pages_processed"] == 2
        assert result.statistics["ocr_attempts"] == 2
        assert result.statistics["ocr_prompt_tokens"] is None
        assert result.statistics["ocr_completion_tokens"] is None
        assert result.statistics["ocr_cached_tokens"] is None

    async def test_credentials_never_reach_the_statistics(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1])
        vision = _FakeVisionClient({1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))})

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert "super-secret" not in json.dumps(result.statistics)


class TestFailedIngestion:
    """Any failure returns a FAILED result, marks the source and cleans up."""

    async def test_failed_page_aborts_the_whole_document(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1, 2])
        vision = _FakeVisionClient(
            {1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))}, fail_on=2
        )

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.FAILED
        assert result.root_node is None
        assert "HTTP 500" in result.error
        assert sources[source.id].status == "error"
        assert vision.closed
        # A failed run still reports what it spent, under the names a completed
        # one uses: one page came back and the failure carried no bill of its own.
        assert result.statistics["successful"] is False
        assert result.statistics["pages_processed"] == 1
        assert result.statistics["ocr_attempts"] == 1
        assert result.statistics["ocr_prompt_tokens"] == 5
        assert result.statistics["ocr_completion_tokens"] == 3
        assert result.statistics["ocr_cached_tokens"] == 4

    async def test_a_failed_page_reports_its_own_bill(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The page that failed was requested three times and the endpoint was
        # paid for those requests, so the run's totals include them.
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1, 2])
        vision = _FakeVisionClient(
            {1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))},
            fail_on=2,
            spend={
                "attempts": 3,
                "prompt_tokens": 7,
                "completion_tokens": 4,
                "cached_tokens": 1,
            },
        )

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.FAILED
        assert result.statistics["pages_processed"] == 1
        assert result.statistics["ocr_attempts"] == 4
        assert result.statistics["ocr_prompt_tokens"] == 12
        assert result.statistics["ocr_completion_tokens"] == 7
        assert result.statistics["ocr_cached_tokens"] == 5

    async def test_a_failed_page_that_reported_nothing_makes_the_total_unknown(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The failed page was billed but said nothing about what it cost, so the
        # totals are unknown rather than the measured pages alone: a total that
        # skipped it would price it at zero.
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1, 2])
        vision = _FakeVisionClient(
            {1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))},
            fail_on=2,
            spend={"attempts": 3},
        )

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.statistics["ocr_attempts"] == 4
        assert result.statistics["ocr_prompt_tokens"] is None
        assert result.statistics["ocr_completion_tokens"] is None
        assert result.statistics["ocr_cached_tokens"] is None

    async def test_the_last_page_failing_keeps_the_whole_runs_accounting(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Three pages came back and were billed before the fourth failed. The
        # manifest reports what the run spent, not the failed page's bill alone:
        # every page the run paid for is in the total.
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1, 2, 3, 4])
        vision = _FakeVisionClient(
            {
                1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn một")),
                2: _page(2, VisionBlock(kind="p", group=1, text="Đoạn hai")),
                3: _page(3, VisionBlock(kind="p", group=1, text="Đoạn ba")),
            },
            fail_on=4,
            spend={
                "attempts": 3,
                "prompt_tokens": 7,
                "completion_tokens": 4,
                "cached_tokens": 1,
            },
        )

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.FAILED
        # No partial tree is published as a success.
        assert result.root_node is None
        assert result.statistics["pages_processed"] == 3
        # Three measured pages (5/3/4 each, one attempt each) plus the failed page.
        assert result.statistics["ocr_attempts"] == 6
        assert result.statistics["ocr_prompt_tokens"] == 22
        assert result.statistics["ocr_completion_tokens"] == 13
        assert result.statistics["ocr_cached_tokens"] == 13

    async def test_hash_mismatch_fails_before_any_vision_call(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source(hash_content="0" * 32)
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1])
        vision = _FakeVisionClient({1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))})

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.FAILED
        assert "Integrity check failed" in result.error
        assert vision.states == []
        assert sources[source.id].status == "error"
        # Nothing was sent, so the run's spend is a measured zero, not unknown.
        assert result.statistics["pages_processed"] == 0
        assert result.statistics["ocr_attempts"] == 0
        assert result.statistics["ocr_prompt_tokens"] == 0
        assert result.statistics["ocr_completion_tokens"] == 0
        assert result.statistics["ocr_cached_tokens"] == 0

    async def test_already_active_source_keeps_its_status(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source(status="active")
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1])
        vision = _FakeVisionClient({1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))})

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.FAILED
        assert "already ingested" in result.error
        assert sources[source.id].status == "active"

    async def test_missing_minio_config_fails(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source(config={"original_filename": "book.pdf"})
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1])
        vision = _FakeVisionClient({1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))})

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.FAILED
        assert "minio_bucket" in result.error

    async def test_document_without_pages_fails(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [])
        vision = _FakeVisionClient({})

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.FAILED
        assert "no renderable pages" in result.error

    async def test_unknown_source_fails_without_a_handler_record(
        self,
        handler: _FakeSources,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source()
        _patch_renderer(monkeypatch, [1])
        vision = _FakeVisionClient({1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))})

        result = await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert result.status == FileIngestionStatus.FAILED
        assert result.data_source.name == "Unknown"
        assert result.data_source.status == "error"


class TestWorkspace:
    """The temporary job directory never survives a run."""

    async def test_job_directory_is_removed(
        self,
        handler: _FakeSources,
        sources: dict[UUID, DataSource],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _make_source()
        sources[source.id] = source
        _patch_renderer(monkeypatch, [1])
        vision = _FakeVisionClient({1: _page(1, VisionBlock(kind="p", group=1, text="Đoạn"))})

        await _run_workflow(
            source=source, vision_client=vision, storage=_FakeStorage(), tmp_path=tmp_path
        )

        assert list(tmp_path.iterdir()) == []
