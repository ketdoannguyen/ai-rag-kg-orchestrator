"""Offline tests for the vision OCR transport layer.

Every request goes through ``httpx.MockTransport``, so nothing here touches the
network, a PDF file, or a vision endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from workflows.ingest_file import vision_ocr
from workflows.ingest_file.vision_ocr import (
    VisionBlock,
    VisionOCRClient,
    VisionOCRConfig,
    VisionOCRError,
    VisionPage,
    render_pdf_pages,
)

ENDPOINT = "https://vision.test/v1/chat/completions"
API_KEY = "secret-api-key"

STATE: dict[str, Any] = {
    "open_headings": [{"marker": "1", "depth": 1, "text": "Bài 1"}],
    "open_group": 3,
    "open_group_tail": "đoạn đang mở",
}


def _document(page: int = 1, blocks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if blocks is None:
        blocks = [{"kind": "p", "marker": None, "depth": None, "group": 1, "text": "Nội dung trang"}]
    return {"page": page, "running": "", "blocks": blocks}


def _response(
    document: dict[str, Any] | None = None,
    *,
    finish_reason: str = "stop",
    content: str | None = None,
) -> httpx.Response:
    if content is None:
        content = json.dumps(document if document is not None else _document())
    return httpx.Response(
        200,
        json={
            "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


class _Transport:
    """MockTransport handler that records requests and replays canned responses."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self._responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self._responses[min(len(self.requests) - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def calls(self) -> int:
        return len(self.requests)


@pytest.fixture
def sleeps() -> list[float]:
    return []


def _client(transport: _Transport, sleeps: list[float], **overrides: Any) -> VisionOCRClient:
    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    options: dict[str, Any] = {
        "api_base": ENDPOINT,
        "model": "test-model",
        "api_key": API_KEY,
        "allowed_hosts": ("vision.test",),
    }
    options.update(overrides)
    config = VisionOCRConfig(**options)
    return VisionOCRClient(config, transport=httpx.MockTransport(transport), sleep=_sleep)


async def _extract(client: VisionOCRClient, *, page_number: int = 1):
    return await client.extract_page(b"png-bytes", page_number=page_number, state=STATE)


class TestAcceptedPages:
    """A clean page is accepted once, with usage accounted."""

    async def test_clean_page_is_not_retried(self, sleeps: list[float]) -> None:
        transport = _Transport(_response())
        client = _client(transport, sleeps)
        try:
            result = await _extract(client)
        finally:
            await client.aclose()

        assert result.page.page == 1
        assert result.attempts == 1
        assert (result.prompt_tokens, result.completion_tokens) == (11, 7)
        assert transport.calls == 1
        assert sleeps == []

    async def test_fenced_json_is_accepted(self, sleeps: list[float]) -> None:
        fenced = f"```json\n{json.dumps(_document())}\n```"
        transport = _Transport(_response(content=fenced))
        client = _client(transport, sleeps)
        try:
            result = await _extract(client)
        finally:
            await client.aclose()

        assert result.page.blocks[0].text == "Nội dung trang"


class TestRetries:
    """Transient failures retry with backoff; permanent ones do not."""

    @pytest.mark.parametrize(
        "failure",
        [
            httpx.Response(500, text="server error"),
            httpx.Response(429, text="slow down"),
            httpx.ConnectTimeout("timed out"),
            _response(finish_reason="length"),
            _response(content="not json at all"),
            _response(document={"page": 1, "running": "", "blocks": [], "extra": 1}),
            _response(document=_document(page=99)),
            # A block without a group would silently fall back to per-block
            # chunks, so the page is rejected and retried instead.
            _response(
                document=_document(
                    blocks=[{"kind": "p", "marker": None, "depth": None, "text": "một"}]
                )
            ),
            _response(
                document=_document(
                    blocks=[{"kind": "p", "marker": None, "depth": None, "group": -1, "text": "một"}]
                )
            ),
            _response(
                document=_document(
                    blocks=[{"kind": "p", "marker": None, "depth": None, "group": 0, "text": "một"}]
                )
            ),
        ],
        ids=[
            "http-500",
            "http-429",
            "timeout",
            "truncated",
            "bad-json",
            "schema",
            "page-mismatch",
            "missing-group",
            "negative-group",
            "zero-group",
        ],
    )
    async def test_transient_failure_then_success(
        self, failure: httpx.Response | Exception, sleeps: list[float]
    ) -> None:
        transport = _Transport(failure, _response())
        client = _client(transport, sleeps)
        try:
            result = await _extract(client)
        finally:
            await client.aclose()

        assert result.attempts == 2
        assert transport.calls == 2
        assert sleeps == [1]

    async def test_exhausted_retries_raise_with_diagnostics(self, sleeps: list[float]) -> None:
        transport = _Transport(httpx.Response(503, text="unavailable"))
        client = _client(transport, sleeps, max_retries=2)
        try:
            with pytest.raises(VisionOCRError) as excinfo:
                await _extract(client)
        finally:
            await client.aclose()

        assert transport.calls == 3
        assert sleeps == [1, 2]
        assert "page 1" in str(excinfo.value)
        assert "3 attempt(s)" in str(excinfo.value)

    async def test_too_many_blocks_is_rejected(self, sleeps: list[float]) -> None:
        blocks = [
            {"kind": "p", "marker": None, "depth": None, "group": 1, "text": "một"},
            {"kind": "p", "marker": None, "depth": None, "group": 2, "text": "hai"},
        ]
        transport = _Transport(_response(_document(blocks=blocks)), _response())
        client = _client(transport, sleeps, max_blocks_per_page=1)
        try:
            result = await _extract(client)
        finally:
            await client.aclose()

        assert transport.calls == 2
        assert result.attempts == 2

    async def test_client_error_fails_without_retry(self, sleeps: list[float]) -> None:
        transport = _Transport(httpx.Response(400, text="bad request"))
        client = _client(transport, sleeps)
        try:
            with pytest.raises(VisionOCRError) as excinfo:
                await _extract(client)
        finally:
            await client.aclose()

        assert transport.calls == 1
        assert sleeps == []
        assert "HTTP 400" in str(excinfo.value)

    async def test_non_object_content_fails_without_retry(self, sleeps: list[float]) -> None:
        transport = _Transport(_response(content="[1, 2, 3]"))
        client = _client(transport, sleeps)
        try:
            with pytest.raises(VisionOCRError):
                await _extract(client)
        finally:
            await client.aclose()

        assert transport.calls == 1

    async def test_non_object_response_envelope_is_retried(self, sleeps: list[float]) -> None:
        malformed = httpx.Response(200, json=[{"error": "temporary gateway response"}])
        transport = _Transport(malformed, _response())
        client = _client(transport, sleeps)
        try:
            result = await _extract(client)
        finally:
            await client.aclose()

        assert result.attempts == 2
        assert transport.calls == 2
        assert sleeps == [1]


class TestCredentials:
    """The key never leaves the allowlist, and never lands in a payload or error."""

    async def test_missing_key_fails_before_any_request(self, sleeps: list[float]) -> None:
        transport = _Transport(_response())
        client = _client(transport, sleeps, api_key=None)
        try:
            with pytest.raises(VisionOCRError, match="VISION_API_KEY"):
                await _extract(client)
        finally:
            await client.aclose()

        assert transport.calls == 0

    async def test_host_outside_allowlist_fails_before_any_request(
        self, sleeps: list[float]
    ) -> None:
        transport = _Transport(_response())
        client = _client(transport, sleeps, allowed_hosts=("other.test",))
        try:
            with pytest.raises(VisionOCRError, match="refusing to send"):
                await _extract(client)
        finally:
            await client.aclose()

        assert transport.calls == 0

    async def test_key_is_sent_as_bearer_and_never_in_the_body(self, sleeps: list[float]) -> None:
        transport = _Transport(_response())
        client = _client(transport, sleeps, headers={"X-Trace": "abc", "Authorization": "nope"})
        try:
            await _extract(client)
        finally:
            await client.aclose()

        request = transport.requests[0]
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        assert request.headers["X-Trace"] == "abc"
        assert API_KEY not in request.content.decode("utf-8")

    async def test_error_message_does_not_leak_the_key(self, sleeps: list[float]) -> None:
        transport = _Transport(httpx.Response(500, text=f"upstream said {API_KEY}"))
        client = _client(transport, sleeps)
        try:
            with pytest.raises(VisionOCRError) as excinfo:
                await _extract(client)
        finally:
            await client.aclose()

        assert API_KEY not in str(excinfo.value)


class TestBlockText:
    """Text is mandatory except for the one kind that may be a picture alone."""

    def test_textless_figure_is_accepted(self) -> None:
        block = VisionBlock(kind="figure", group=1, text="")

        assert block.text == ""

    @pytest.mark.parametrize(
        "kind", ["heading", "p", "list", "question", "callout", "table", "equation"]
    )
    def test_blank_text_is_rejected_for_every_other_kind(self, kind: str) -> None:
        with pytest.raises(ValidationError, match="text must not be blank"):
            VisionBlock(kind=kind, group=1, depth=1 if kind == "heading" else None, text="   ")


class TestGrouping:
    """The model labels groups; a page that omits them is rejected, not guessed."""

    def test_continues_previous_defaults_to_false(self) -> None:
        page = VisionPage.model_validate(
            {"page": 2, "running": "", "blocks": [{"kind": "p", "group": 1, "text": "Nội dung"}]}
        )

        assert page.continues_previous is False

    def test_group_is_required_on_every_block(self) -> None:
        with pytest.raises(ValidationError):
            VisionBlock(kind="p", text="Nội dung")

    @pytest.mark.parametrize("group", [-1, 0])
    def test_non_positive_group_is_rejected(self, group: int) -> None:
        with pytest.raises(ValidationError):
            VisionBlock(kind="p", group=group, text="Nội dung")

    def test_enumerator_on_a_list_item_is_tolerated(self) -> None:
        # The model often repeats the visible enumerator in `marker`, which the
        # text already carries. A redundant label must not fail the whole page.
        block = VisionBlock(
            kind="list", marker="a)", group=1, text="a) Chất khí không có hình dạng riêng."
        )

        assert block.marker == "a)"


class TestPayload:
    """Model options, page number, open headings and the image all reach the endpoint."""

    async def test_payload_carries_options_state_and_image(self, sleeps: list[float]) -> None:
        transport = _Transport(_response(_document(page=7)))
        client = _client(transport, sleeps, temperature=0.25, max_tokens=1234)
        try:
            await _extract(client, page_number=7)
        finally:
            await client.aclose()

        payload = json.loads(transport.requests[0].content.decode("utf-8"))
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.25
        assert payload["max_tokens"] == 1234
        assert payload["response_format"] == {"type": "json_object"}
        # A reasoning model that is left to reason can spend the whole budget
        # before writing any JSON, so the default silences it.
        assert payload["reasoning_effort"] == "none"

        text_part, image_part = payload["messages"][0]["content"]
        assert "PAGE=7" in text_part["text"]
        assert '"open_headings":[{"marker":"1"' in text_part["text"]
        # The open group label has to travel with the page: it is the only way
        # the model can continue that group instead of restarting numbering.
        assert '"open_group":3' in text_part["text"]
        assert '"open_group_tail":"đoạn đang mở"' in text_part["text"]
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        assert image_part["image_url"]["detail"] == "high"

    async def test_reasoning_hint_can_be_bought_back_or_omitted(self, sleeps: list[float]) -> None:
        transport = _Transport(_response())
        client = _client(transport, sleeps, reasoning_effort="low")
        try:
            await _extract(client)
        finally:
            await client.aclose()

        payload = json.loads(transport.requests[0].content.decode("utf-8"))
        assert payload["reasoning_effort"] == "low"

        omitted = _Transport(_response())
        client = _client(omitted, sleeps, reasoning_effort=None)
        try:
            await _extract(client)
        finally:
            await client.aclose()

        payload = json.loads(omitted.requests[0].content.decode("utf-8"))
        assert "reasoning_effort" not in payload


class _FakeBitmap:
    def __init__(self) -> None:
        self.image = Image.new("RGB", (4, 3))
        self.closed = False

    def to_pil(self) -> Image.Image:
        return self.image

    def close(self) -> None:
        self.closed = True


class _FakePage:
    def __init__(self) -> None:
        self.scales: list[float] = []
        self.closed = False

    def render(self, *, scale: float) -> _FakeBitmap:
        self.scales.append(scale)
        return _FakeBitmap()

    def close(self) -> None:
        self.closed = True


class _FakeDocument:
    instances: list[_FakeDocument] = []

    def __init__(self, path: str) -> None:
        self.path = path
        self.pages = [_FakePage(), _FakePage()]
        self.closed = False
        _FakeDocument.instances.append(self)

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self.pages[index]

    def close(self) -> None:
        self.closed = True


class TestRenderPdfPages:
    """Rendering yields one PNG per page with printed page numbers."""

    def test_pages_are_numbered_scaled_and_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeDocument.instances = []
        monkeypatch.setattr(vision_ocr.pypdfium2, "PdfDocument", _FakeDocument)

        rendered = list(render_pdf_pages(Path("book.pdf"), dpi=144, page_offset=5))

        assert [number for number, _ in rendered] == [6, 7]
        for _, png_bytes in rendered:
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

        document = _FakeDocument.instances[0]
        assert document.path == "book.pdf"
        assert document.closed
        assert [page.scales for page in document.pages] == [[2.0], [2.0]]
        assert all(page.closed for page in document.pages)
