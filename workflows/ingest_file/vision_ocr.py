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
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
import pypdfium2
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails

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
    """Raised when a page cannot be extracted, including after all retries.

    A page that failed is still a bill: every HTTP 200 the endpoint answered was
    paid for, even when the page inside it was rejected and retried. ``attempts``
    counts the requests the page cost, and the token counts are ``None`` when no
    billed response reported them — an unknown count recorded as zero would claim
    the page was free. An error raised before any request carries no attempts and
    no counts, so a caller merging this spend keys off ``attempts``.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cached_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens


class _RetryableAttemptError(Exception):
    """Internal signal: this attempt failed but a retry may still succeed."""


class _PermanentAttemptError(Exception):
    """Internal signal: this attempt failed and an identical retry cannot help."""


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
    """An accepted page plus the accounting used for run statistics.

    ``attempts`` is always measured: the client knows how many requests the page
    cost it. The token counts are the sum over every 200 the page was billed for,
    retried ones included, and are ``None`` when one of those responses did not
    report the count, because an unknown count recorded as zero would claim the
    page was free.
    """

    page: VisionPage
    attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None = None
    """Cached-prompt share the endpoint reported, or ``None`` when it stated none."""


def _count(value: Any) -> int | None:
    """A non-negative integer token count, or ``None`` when the value is unusable.

    Absent and null both mean "not reported": the count is unknown, and neither
    is ever read as zero. A value that is present but not a non-negative integer
    is a malformed envelope, and unknown for the same reason.
    """
    if value is None:
        return None
    try:
        count = int(value)
    except (ValueError, TypeError):
        return None
    return count if count >= 0 else None


def _cached_count(usage: Mapping[str, Any]) -> int | None:
    """The cached-prompt share the response reported, or ``None`` when it stated none.

    An OpenAI-compatible endpoint states it under
    ``prompt_tokens_details.cached_tokens``. A response that says nothing about
    the cache leaves the share unknown, exactly like a missing token count:
    reading it as zero would claim the endpoint accounted for the cache and found
    no hit.
    """
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return None
    return _count(details.get("cached_tokens"))


def add_measured(total: int | None, value: int | None) -> int | None:
    """Add one count to a running total, or ``None`` once a count is unknown.

    A total that skipped a count nobody reported would price that attempt at
    zero, so one unknown count makes the whole total unknown.
    """
    if total is None or value is None:
        return None
    return total + value


@dataclass
class _BilledUsage:
    """What one page's requests were billed, summed over the 200s that answered.

    Every HTTP 200 is billed, including one whose page is rejected and retried,
    so each response's counts are added to the page's totals. A count a billed
    response left unreported makes that total unknown (``None``): a sum that
    skipped it would price the attempt at zero.
    """

    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    cached_tokens: int | None = 0

    def record(self, usage: Mapping[str, Any] | None) -> None:
        """Add one billed response's counts.

        ``None`` is a response whose usage envelope could not be read at all:
        the attempt stays on the bill, with every count unknown.
        """
        prompt = completion = cached = None
        if usage is not None:
            prompt = _count(usage.get("prompt_tokens"))
            completion = _count(usage.get("completion_tokens"))
            cached = _cached_count(usage)
        self.prompt_tokens = add_measured(self.prompt_tokens, prompt)
        self.completion_tokens = add_measured(self.completion_tokens, completion)
        self.cached_tokens = add_measured(self.cached_tokens, cached)


def _sanitized_validation_error(error: ValidationError) -> ValidationError:
    """Rebuild ``error`` without the raw inputs Pydantic echoes into messages.

    A model-level failure keeps the full constructor input as the offending
    error's ``input``, so a rejected config would otherwise print the API key
    and the configured headers through ``str``, ``json()`` and ``errors()``.
    """
    return ValidationError.from_exception_data(
        error.title, cast("list[InitErrorDetails]", error.errors(include_input=False))
    )


class VisionOCRConfig(BaseModel):
    """Resolved runtime configuration for :class:`VisionOCRClient`.

    ``api_key`` and every header value are held as :class:`~pydantic.SecretStr`,
    and every constructor/validation entry point raises a rebuilt
    :class:`ValidationError` whose errors carry no raw input, so repr, dumps and
    error messages never echo a credential.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_base: str
    model: str
    api_key: SecretStr | None = None
    allowed_hosts: tuple[str, ...] = ()
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    image_detail: Literal["low", "high", "auto"] = "high"
    temperature: float = 0.0
    max_tokens: int = 8000
    reasoning_effort: str | None = "none"
    max_blocks_per_page: int = 400
    max_retries: int = 2
    timeout_seconds: float = 300.0

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise _sanitized_validation_error(exc) from None

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> VisionOCRConfig:
        try:
            return super().model_validate(obj, **kwargs)
        except ValidationError as exc:
            raise _sanitized_validation_error(exc) from None

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> VisionOCRConfig:
        try:
            return super().model_validate_json(json_data, **kwargs)
        except ValidationError as exc:
            raise _sanitized_validation_error(exc) from None

    @classmethod
    def model_validate_strings(cls, obj: Any, **kwargs: Any) -> VisionOCRConfig:
        try:
            return super().model_validate_strings(obj, **kwargs)
        except ValidationError as exc:
            raise _sanitized_validation_error(exc) from None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        max_retries: int,
        timeout_seconds: float,
    ) -> VisionOCRConfig:
        """Build the config from environment settings and per-run retry options."""
        endpoint = urlparse(settings.VISION_API_BASE)
        allowed = tuple(
            host.strip().lower()
            for host in settings.VISION_ALLOWED_HOSTS.split(",")
            if host.strip()
        )
        if (
            endpoint.scheme != "https"
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.port not in (None, 443)
            or endpoint.hostname.lower() not in allowed
        ):
            raise VisionOCRError(
                "VISION_API_BASE must be an HTTPS endpoint on an allowlisted host"
            )
        return cls(
            api_base=settings.VISION_API_BASE,
            model=settings.VISION_MODEL,
            api_key=settings.VISION_API_KEY,
            allowed_hosts=allowed,
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
        endpoint = urlparse(self.api_base)
        if (
            endpoint.scheme != "https"
            or endpoint.username is not None
            or endpoint.password is not None
        ):
            raise ValueError("Vision endpoint must use HTTPS without embedded credentials")
        if endpoint.port not in (None, 443):
            raise ValueError("Vision endpoint must use the standard HTTPS port")
        if not self.allowed_hosts or self.host not in self.allowed_hosts:
            raise ValueError("Vision endpoint host must be explicitly allowlisted")


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
            trust_env=False,
            follow_redirects=False,
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
        """Extract one page, retrying transient failures.

        A page the endpoint answered without usage is accepted with unknown
        token counts. A response truncated at ``max_tokens`` is not retried at
        all: the payload is rebuilt once and sent unchanged, so the retry would
        buy the same truncated answer a second time.
        """
        headers = self._auth_headers()
        payload = self._build_payload(png_bytes, page_number, state)
        max_attempts = self._config.max_retries + 1
        billed = _BilledUsage()
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return await self._request(headers, payload, page_number, attempt, billed)
            except _PermanentAttemptError as exc:
                raise self._page_error(page_number, exc, attempt, billed) from exc
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

        raise self._page_error(page_number, last_error, max_attempts, billed)

    def _page_error(
        self,
        page_number: int,
        error: Exception | None,
        attempts: int,
        billed: _BilledUsage,
    ) -> VisionOCRError:
        """A page failure that reports what the page's requests were billed."""
        return VisionOCRError(
            f"page {page_number}: {error} (after {attempts} attempt(s))",
            attempts=attempts,
            prompt_tokens=billed.prompt_tokens,
            completion_tokens=billed.completion_tokens,
            cached_tokens=billed.cached_tokens,
        )

    def _auth_headers(self) -> dict[str, str]:
        """Return request headers, refusing to leak the key to an unknown host."""
        api_key = self._config.api_key
        if not api_key:
            raise VisionOCRError("VISION_API_KEY is not configured")
        allowed = self._config.allowed_hosts
        if self._config.host not in allowed:
            raise VisionOCRError(
                f"refusing to send VISION_API_KEY to host '{self._config.host}'; "
                "set VISION_ALLOWED_HOSTS to allow it"
            )
        headers = {
            key: value.get_secret_value()
            for key, value in self._config.headers.items()
            if key.lower() != "authorization"
        }
        if self._config.host == "opencode.ai" and not any(
            name.lower() == "x-opencode-session" and value for name, value in headers.items()
        ):
            raise VisionOCRError("x-opencode-session is required for opencode.ai")
        headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
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
        billed: _BilledUsage,
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
            raise _RetryableAttemptError("transport error") from exc

        if response.status_code in (408, 429) or response.status_code >= 500:
            raise _RetryableAttemptError(f"HTTP {response.status_code}")
        if response.status_code != 200:
            raise self._page_error(page_number, f"HTTP {response.status_code}", attempt, billed)

        # A 200 is billed whether or not its envelope can be read, so an
        # attempt whose usage cannot be read stays on the page's bill with every
        # count unknown: a sum that skipped it would present the remaining
        # attempts as the page's total.
        try:
            body = response.json()
        except ValueError as exc:
            billed.record(None)
            raise _RetryableAttemptError("response body was not JSON") from exc
        if not isinstance(body, dict):
            billed.record(None)
            raise _RetryableAttemptError("response body was not a JSON object")

        # A 200 is billed whether or not the page inside it is usable, so what it
        # reported is recorded before the page is judged: a rejected page that is
        # retried still stays on the page's bill.
        usage = body.get("usage")
        billed.record(usage if isinstance(usage, dict) else None)
        page = self._parse_page(body, page_number)
        return PageExtraction(
            page=page,
            attempts=attempt,
            prompt_tokens=billed.prompt_tokens,
            completion_tokens=billed.completion_tokens,
            cached_tokens=billed.cached_tokens,
        )

    def _parse_page(self, body: dict[str, Any], page_number: int) -> VisionPage:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise _RetryableAttemptError("invalid response choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise _RetryableAttemptError("invalid response message")
        finish_reason = choice.get("finish_reason")
        content = message.get("content")

        if finish_reason == "length":
            # The response was cut off at max_tokens. The payload is identical on
            # every attempt, so a retry would be cut at the same place: it costs
            # another billed request and cannot produce a complete page.
            raise _PermanentAttemptError(
                "response hit max_tokens (finish_reason=length); the same payload "
                "would be truncated again"
            )
        if finish_reason != "stop":
            raise _RetryableAttemptError(f"finish_reason={finish_reason!r}")
        if not isinstance(content, str) or not content.strip():
            raise _RetryableAttemptError("response content was empty")

        raw = _load_json_object(content)
        if not isinstance(raw, dict):
            # The endpoint answered, so the attempt is bought and paid for, but
            # it is not repeated: the payload is rebuilt unchanged and the model
            # is deterministic, so an identical retry would buy the same
            # non-object answer again. Permanent, so the attempt loop stops here
            # and reports the 200 that answered on the page's bill.
            raise _PermanentAttemptError("response JSON was not an object")
        try:
            page = VisionPage.model_validate(raw)
        except ValidationError as exc:
            raise _RetryableAttemptError("schema mismatch") from exc

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
    page_indices: list[int] | None = None,
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(request_page_number, png_bytes)`` one selected page at a time.

    Synchronous by design: rendering one page is short and peak memory stays at
    a single page bitmap instead of the whole document.
    """
    document = pypdfium2.PdfDocument(str(path))
    try:
        indices = range(len(document)) if page_indices is None else page_indices
        if any(index < 0 or index >= len(document) for index in indices):
            raise ValueError("page index outside PDF")
        for index in indices:
            page = document[index]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    image = bitmap.to_pil()
                    try:
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        yield index + 1 + page_offset, buffer.getvalue()
                    finally:
                        image.close()
                finally:
                    bitmap.close()
            finally:
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
        raise _RetryableAttemptError("invalid JSON") from exc
