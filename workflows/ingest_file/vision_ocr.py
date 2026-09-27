"""Vision-LLM OCR: render PDF pages and extract typed JSON blocks per page.

The API key comes from settings/environment only. It is never persisted in the
returned data, never logged, and never sent to a host outside the allowlist.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import pypdfium2
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from config.settings import Settings

BlockKind = Literal["heading", "p", "list", "question", "callout", "figure", "table", "equation"]

PROMPT = """Chép nguyên văn toàn bộ ảnh trang tài liệu thành JSON. Không suy luận hoặc giải thích. Chỉ xuất JSON hợp lệ.

Dạng:
{"page":10,"running":"","continues_previous":false,"blocks":[{"kind":"heading","marker":"3.1.1","depth":3,"group":1,"text":"3.1.1 Danh sách bài báo"}]}
Mỗi block có đúng kind, marker, depth, group, text theo thứ tự đọc. kind: heading, p, list, question, callout, figure, table, equation. Với heading: marker là ký hiệu đầu tiêu đề nhìn thấy (ví dụ 1, 1.1, 1.1.1, a, a.1, I), hoặc null nếu tiêu đề không có ký hiệu; depth là số tầng cấu trúc suy từ tài liệu và ngữ cảnh, không phải số dấu chấm; không cố định a/b ở một depth cụ thể. Với block khác: marker và depth là null. Số thứ tự trong câu hỏi, bước làm hoặc danh sách không phải heading. Dùng OCR_STATE để biết chuỗi heading cha đang mở; không lặp lại cha không xuất hiện như tiêu đề trên ảnh. Header chạy và số trang chỉ ghi ở page/running, không tạo block. Trường page phải bằng đúng số ở PAGE, không phải số folio in trên ảnh (hai số này có thể lệch nhau). Giữ nguyên tiếng Việt, số, đơn vị, caption và chữ trong hình; không tóm tắt, sửa, mô tả, đoán hoặc bổ sung.

GROUP: group là số nguyên >= 1 gom các block liên tiếp thuộc cùng một ý để chương trình ghép lại thành một đoạn lớn. Block mở đầu ý mới thì tăng số; các block cùng một ý dùng cùng số. Heading luôn mở ý mới. Các item của cùng một danh sách, và nhãn callout với thân callout, phải cùng group. group chỉ là nhãn: không đổi chữ, không viết lại, phần ghép do chương trình làm.
CONTINUES_PREVIOUS: true nếu block đầu trang này nối tiếp ý đang mở ở cuối trang trước (xem OCR_STATE.open_group_tail), ngược lại false. Khi true thì block nội dung đầu tiên của trang phải dùng đúng số OCR_STATE.open_group rồi các ý sau tăng dần từ đó; khi false thì đánh số lại từ 1."""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class VisionOCRError(RuntimeError):
    """Raised when a page cannot be extracted, including after all retries."""


class _RetryableAttemptError(Exception):
    """Internal signal: this attempt failed but a retry may still succeed."""


class VisionBlock(BaseModel):
    """One typed block transcribed from a page image."""

    model_config = ConfigDict(extra="forbid")

    kind: BlockKind
    marker: str | None = None
    """Visible heading symbol (``3.1``, ``a``, ``II``); used only for headings.

    The model also fills this for list items, where the enumerator already
    appears in ``text``. The parser ignores it for every non-heading kind, and
    failing the whole page over a redundant label would be a bad trade.
    """
    depth: int | None = None
    """Structural depth, heading-only; required for ``heading`` and ignored elsewhere."""
    group: int = Field(ge=1)
    """Consecutive blocks sharing a group are joined into one chunk by the parser.

    The number itself carries no meaning beyond "differs from the previous
    block", so the model never has to keep ids consistent across a page.
    """
    text: str

    @field_validator("text")
    @classmethod
    def _require_text(cls, value: str, info: ValidationInfo) -> str:
        text = value.strip()
        # A figure may be a picture with no visible text, so it is the one kind
        # allowed to come back empty. Every other kind carries its payload here.
        if not text and info.data.get("kind") != "figure":
            raise ValueError("text must not be blank")
        return text

    @field_validator("marker")
    @classmethod
    def _clean_marker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        marker = value.strip()
        if not marker:
            raise ValueError("marker must be null or non-blank")
        return marker

    @model_validator(mode="after")
    def _check_heading_fields(self) -> VisionBlock:
        if self.kind == "heading" and (not self.depth or self.depth < 1):
            raise ValueError("heading requires depth >= 1")
        return self


class VisionPage(BaseModel):
    """One accepted page of transcribed blocks."""

    model_config = ConfigDict(extra="forbid")

    page: int
    running: str = ""
    continues_previous: bool = False
    """True when the first block continues the group left open on the last page."""
    blocks: list[VisionBlock]


@dataclass(frozen=True)
class PageExtraction:
    """An accepted page plus the accounting used for run statistics."""

    page: VisionPage
    attempts: int
    prompt_tokens: int
    completion_tokens: int


class VisionOCRConfig(BaseModel):
    """Resolved runtime configuration for :class:`VisionOCRClient`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_base: str
    model: str
    api_key: str | None = None
    allowed_hosts: tuple[str, ...] = ()
    headers: dict[str, str] = Field(default_factory=dict)
    image_detail: Literal["low", "high", "auto"] = "high"
    temperature: float = 0.0
    max_tokens: int = 8000
    reasoning_effort: str | None = "none"
    max_blocks_per_page: int = 400
    max_retries: int = 2
    timeout_seconds: float = 300.0

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        max_retries: int,
        timeout_seconds: float,
    ) -> VisionOCRConfig:
        """Build the config from environment settings and per-run retry options."""
        return cls(
            api_base=settings.VISION_API_BASE,
            model=settings.VISION_MODEL,
            api_key=settings.VISION_API_KEY,
            allowed_hosts=tuple(
                host.strip().lower()
                for host in settings.VISION_ALLOWED_HOSTS.split(",")
                if host.strip()
            ),
            headers=dict(settings.VISION_HEADERS),
            image_detail=settings.VISION_IMAGE_DETAIL,
            temperature=settings.VISION_TEMPERATURE,
            max_tokens=settings.VISION_MAX_TOKENS,
            reasoning_effort=settings.VISION_REASONING_EFFORT or None,
            max_blocks_per_page=settings.VISION_MAX_BLOCKS_PER_PAGE,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

    @property
    def host(self) -> str:
        """Hostname of the configured endpoint."""
        return (urlparse(self.api_base).hostname or "").lower()

    def model_post_init(self, __context: Any) -> None:
        if not self.allowed_hosts:
            logger.warning(
                "VISION_ALLOWED_HOSTS is empty: VISION_API_KEY may be sent to any host",
                endpoint_host=self.host,
            )


class VisionOCRClient:
    """Transcribes one page image per request against an OpenAI-compatible API.

    Any page that fails after ``config.max_retries`` attempts raises, because a
    missing page would leave a hole in the document hierarchy.
    """

    def __init__(
        self,
        config: VisionOCRConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._sleep = sleep or asyncio.sleep
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    @property
    def config(self) -> VisionOCRConfig:
        return self._config

    async def aclose(self) -> None:
        await self._client.aclose()

    async def extract_page(
        self,
        png_bytes: bytes,
        *,
        page_number: int,
        state: Mapping[str, Any],
    ) -> PageExtraction:
        """Extract one page, retrying transient failures."""
        headers = self._auth_headers()
        payload = self._build_payload(png_bytes, page_number, state)
        max_attempts = self._config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return await self._request(headers, payload, page_number, attempt)
            except _RetryableAttemptError as exc:
                last_error = exc
                logger.warning(
                    "Vision OCR attempt failed",
                    page=page_number,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < max_attempts:
                    await self._sleep(min(2 ** (attempt - 1), 30))

        raise VisionOCRError(f"page {page_number}: {last_error} (after {max_attempts} attempt(s))")

    def _auth_headers(self) -> dict[str, str]:
        """Return request headers, refusing to leak the key to an unknown host."""
        api_key = self._config.api_key
        if not api_key:
            raise VisionOCRError("VISION_API_KEY is not configured")
        allowed = self._config.allowed_hosts
        if allowed and self._config.host not in allowed:
            raise VisionOCRError(
                f"refusing to send VISION_API_KEY to host '{self._config.host}'; "
                "set VISION_ALLOWED_HOSTS to allow it"
            )
        headers = {
            key: value
            for key, value in self._config.headers.items()
            if key.lower() != "authorization"
        }
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(
        self,
        png_bytes: bytes,
        page_number: int,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        image = base64.b64encode(png_bytes).decode("ascii")
        state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{PROMPT}\n\nPAGE={page_number}\nOCR_STATE={state_json}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}",
                                "detail": self._config.image_detail,
                            },
                        },
                    ],
                }
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._config.reasoning_effort is not None:
            payload["reasoning_effort"] = self._config.reasoning_effort
        return payload

    async def _request(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
        page_number: int,
        attempt: int,
    ) -> PageExtraction:
        try:
            response = await self._client.post(
                self._config.api_base, json=payload, headers=headers
            )
        except httpx.TimeoutException as exc:
            raise _RetryableAttemptError(
                f"request timed out after {self._config.timeout_seconds}s"
            ) from exc
        except httpx.TransportError as exc:
            raise _RetryableAttemptError(f"transport error: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableAttemptError(f"HTTP {response.status_code}")
        if response.status_code != 200:
            raise VisionOCRError(
                f"page {page_number}: HTTP {response.status_code}: {_shorten(response.text)}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise _RetryableAttemptError(f"response body was not JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise _RetryableAttemptError("response body was not a JSON object")

        page = self._parse_page(body, page_number)
        usage = body.get("usage") or {}
        return PageExtraction(
            page=page,
            attempts=attempt,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    def _parse_page(self, body: dict[str, Any], page_number: int) -> VisionPage:
        choice = (body.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        content = (choice.get("message") or {}).get("content")

        if finish_reason != "stop":
            raise _RetryableAttemptError(f"finish_reason={finish_reason!r}")
        if not isinstance(content, str) or not content.strip():
            raise _RetryableAttemptError("response content was empty")

        raw = _load_json_object(content)
        if not isinstance(raw, dict):
            raise VisionOCRError(f"page {page_number}: response JSON was not an object")
        try:
            page = VisionPage.model_validate(raw)
        except ValidationError as exc:
            raise _RetryableAttemptError(f"schema mismatch: {_shorten(str(exc))}") from exc

        if page.page != page_number:
            raise _RetryableAttemptError(f"page mismatch: expected {page_number}, got {page.page}")
        if len(page.blocks) > self._config.max_blocks_per_page:
            raise _RetryableAttemptError(
                f"{len(page.blocks)} blocks exceeds VISION_MAX_BLOCKS_PER_PAGE"
            )
        return page


def render_pdf_pages(
    path: Path,
    *,
    dpi: int = 200,
    page_offset: int = 0,
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(printed_page_number, png_bytes)`` one page at a time.

    Synchronous by design: rendering one page is short and peak memory stays at
    a single page bitmap instead of the whole document.
    """
    document = pypdfium2.PdfDocument(str(path))
    try:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=dpi / 72)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            yield index + 1 + page_offset, buffer.getvalue()
            image.close()
            bitmap.close()
            page.close()
    finally:
        document.close()


def _load_json_object(content: str) -> Any:
    """Decode the model's JSON payload, tolerating a fenced code block."""
    text = content.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _RetryableAttemptError(f"invalid JSON: {exc}") from exc


def _shorten(value: str, limit: int = 300) -> str:
    text = " ".join(value.split())
    return text if len(text) <= limit else f"{text[:limit]}..."
