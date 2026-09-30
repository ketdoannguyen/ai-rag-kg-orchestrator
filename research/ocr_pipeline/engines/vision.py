"""Vision: the runtime client driven page by page, into the same neutral artifact.

This is the runtime's own engine, not a copy of it. Every page image comes from
:func:`render_pdf_pages`, every request goes through
:class:`~workflows.ingest_file.vision_ocr.VisionOCRClient`, and the prompt's
running state comes from a real
:class:`~workflows.ingest_file.vision_parser.VisionDocumentParser` fed exactly
as the runtime feeds it -- so the artifact of a run here and the artifact of the
runtime pipeline cannot drift apart in page numbering, grouping or prompt.

Two facts shape this adapter:

Paid, and not reproducible offline
    Each page costs one request to the configured endpoint. :func:`probe` reads
    configuration only and never posts anything, so a missing key or an
    unallowlisted endpoint is found before a run instead of during one.
Whose secrets these are
    The client holds the API key and the full endpoint URL, and none of that
    belongs in an artifact. ``public_config`` is an allowlist built from named
    fields -- model, host, and the request shape -- never the key, never a
    header, never the URL with credentials in it.

The vision vocabulary *is* the runtime vocabulary, so ``kind`` and
``normalized_kind`` are the same string here and :func:`..content_pages` passes
every block through. A page that fails after its retries raises and aborts the
run: an artifact with a hole in the middle is worse than no artifact, because
nothing downstream can tell the hole from a blank page.

A selection with gaps
    A page is only offered the state of the page before it when that page was
    read: at a gap the parser starts fresh and the page cannot claim to continue,
    so no group or heading is carried across a region the model was never shown.
    Every page read before the gap is kept, and the reset is recorded both as the
    ``vision_continuity_resets`` counter and as a note naming the pages.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from workflows.ingest_file.vision_ocr import (
    PROMPT,
    PageExtraction,
    VisionOCRClient,
    VisionOCRConfig,
    VisionOCRError,
    add_measured,
    render_pdf_pages,
)
from workflows.ingest_file.vision_parser import VisionDocumentParser

from . import (
    STATUS_OK,
    EngineArtifact,
    EngineBlock,
    EnginePage,
    EngineProbe,
    follows_previous,
    sha256_bytes,
    sha256_file,
)

if TYPE_CHECKING:
    import httpx

ENGINE_ID = "vision"


def probe(config: VisionOCRConfig | None = None) -> EngineProbe:
    """Report whether a vision run is possible, reading configuration only.

    No page is rendered, nothing is posted and no key is echoed. Without an
    explicit ``config`` the project settings are validated through the client's
    own :meth:`~workflows.ingest_file.vision_ocr.VisionOCRConfig.from_settings`,
    so this reports exactly the conditions a run would fail on. The retry and
    timeout arguments there are per-run knobs with no bearing on availability, so
    the probe passes zeroes and discards the result.
    """
    if config is None:
        try:
            from config.settings import get_settings

            config = VisionOCRConfig.from_settings(
                get_settings(), max_retries=0, timeout_seconds=0.0
            )
        except (VisionOCRError, ValidationError):
            return EngineProbe(
                engine_id=ENGINE_ID,
                available=False,
                version=None,
                detail=(
                    "vision endpoint is not usable: VISION_API_BASE must be an HTTPS "
                    "endpoint on a host listed in VISION_ALLOWED_HOSTS"
                ),
            )
    if not config.api_key:
        return EngineProbe(
            engine_id=ENGINE_ID,
            available=False,
            version=None,
            detail=f"endpoint {config.host} is configured but VISION_API_KEY is not set",
        )
    return EngineProbe(
        engine_id=ENGINE_ID,
        available=True,
        version=config.model,
        detail=(
            f"paid endpoint at {config.host}; one request per page, retried per run, "
            "and nothing is posted by this probe"
        ),
    )


def public_config(config: VisionOCRConfig, *, dpi: int) -> dict[str, Any]:
    """The resolved request shape, allowlisted field by field.

    Everything here is safe to keep beside an artifact: the model, the endpoint
    *host* (never the URL), and the request parameters. The API key, the
    configured headers and the full URL are deliberately absent, so an artifact
    or a report can be shared without leaking a credential.
    """
    return {
        "model": config.model,
        "endpoint_host": config.host,
        "image_detail": config.image_detail,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort,
        "max_blocks_per_page": config.max_blocks_per_page,
        "dpi": dpi,
    }


def _page(
    extraction: PageExtraction, *, index: int, page_number: int, continues: bool
) -> EnginePage:
    """One accepted page plus the accounting the client spent on it.

    ``continues`` is the run's continuity policy: a page may only claim to continue
    the page before it when that page was actually read, so the claim is dropped at
    a gap rather than letting a replay carry a group across an unread region.

    Usage is all or nothing per page: when the endpoint did not report the token
    counts, the page records no usage at all. Recording them as zero would report
    a page nobody billed as a page that cost nothing. Attempts remain in the
    artifact's ``normalization``; the provider's cached share is not tracked here.
    """
    blocks = tuple(
        EngineBlock(
            kind=block.kind,
            normalized_kind=block.kind,
            text=block.text,
            marker=block.marker,
            depth=block.depth,
            group=block.group,
        )
        for block in extraction.page.blocks
    )
    usage = None
    if extraction.prompt_tokens is not None and extraction.completion_tokens is not None:
        usage = {
            "attempts": extraction.attempts,
            "prompt_tokens": extraction.prompt_tokens,
            "completion_tokens": extraction.completion_tokens,
        }
    return EnginePage(
        page_index=index,
        request_page=page_number,
        blocks=blocks,
        running=extraction.page.running,
        continues_previous=continues and extraction.page.continues_previous,
        status=STATUS_OK,
        usage=usage,
    )


async def extract(
    pdf_path: str | Path,
    *,
    doc_id: str,
    config: VisionOCRConfig,
    client: VisionOCRClient | None = None,
    dpi: int = 200,
    page_offset: int = 0,
    page_indices: Sequence[int] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> EngineArtifact:
    """Transcribe the pages of one PDF into a neutral artifact.

    Pages are rendered and requested one at a time, in ascending index order, and
    the parser's state is handed to each request so the prompt sees the same open
    headings and group a runtime run would. ``usage`` records the tokens the
    client was told it spent, attempts included, so a retried page is visible as
    one and a rejected 200 that was retried stays counted. When the endpoint
    omitted token usage for a page, that page records no usage and the run's token
    counters are absent: an unknown count is never written as a zero somebody
    measured. The provider's cached prompt share is not recorded, so a report
    cannot mistake missing cache accounting for a measured zero.

    ``client``, ``transport`` and ``sleep`` exist for the same reason the runtime
    client exposes them: a caller that already owns a client passes it and this
    function closes only what it built, and a test can drive the real client
    through a mock transport without touching a network. A page that fails after
    its retries raises, aborting the run rather than returning a partial document.

    A selection with gaps resets the parser at each gap and refuses the next
    page's claim to continue, so nothing is carried across a region the model was
    not shown; every page read before the gap is kept and the resets are recorded.
    """
    path = Path(pdf_path)
    owned = client is None
    engine = (
        VisionOCRClient(config, transport=transport, sleep=sleep) if client is None else client
    )
    parser = VisionDocumentParser(doc_id)
    pages: list[EnginePage] = []
    resets: list[int] = []
    previous_index: int | None = None
    attempts = 0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    try:
        for page_number, png_bytes in render_pdf_pages(
            path,
            dpi=dpi,
            page_offset=page_offset,
            page_indices=None if page_indices is None else list(page_indices),
        ):
            index = page_number - 1 - page_offset
            # A page may only continue the page before it when that page was read.
            # At a gap the parser starts fresh, so the state of a page the model was
            # never shown is not offered to it as something to continue.
            continues = follows_previous(index, previous_index)
            if previous_index is not None and not continues:
                parser = VisionDocumentParser(doc_id)
                resets.append(index)
            previous_index = index
            extraction = await engine.extract_page(
                png_bytes,
                page_number=page_number,
                state=parser.state,
            )
            parser.add_page(extraction.page, page_number=page_number)
            pages.append(
                _page(extraction, index=index, page_number=page_number, continues=continues)
            )
            attempts += extraction.attempts
            prompt_tokens = add_measured(prompt_tokens, extraction.prompt_tokens)
            completion_tokens = add_measured(completion_tokens, extraction.completion_tokens)
    finally:
        # A client the caller passed in belongs to the caller.
        if owned:
            await engine.aclose()

    if not pages:
        raise ValueError(f"{path}: no renderable pages")

    counters = {
        "vision_pages": len(pages),
        "vision_blocks": sum(len(page.blocks) for page in pages),
        "vision_attempts": attempts,
        "vision_continuity_resets": len(resets),
    }
    if prompt_tokens is not None and completion_tokens is not None:
        counters["vision_prompt_tokens"] = prompt_tokens
        counters["vision_completion_tokens"] = completion_tokens
    notes = [
        "a page that fails after its retries aborts the run: the artifact is never a partial document"
    ]
    if resets:
        notes.append(
            f"page continuity was reset at physical pages {resets}: the document was not read "
            "between them, so nothing open before the gap continues across it"
        )
    return EngineArtifact(
        engine_id=ENGINE_ID,
        engine_version=config.model,
        doc_id=doc_id,
        pages=tuple(pages),
        pdf_sha256=sha256_file(path),
        render_dpi=dpi,
        render_page_offset=page_offset,
        prompt_sha256=sha256_bytes(PROMPT.encode("utf-8")),
        public_config=public_config(config, dpi=dpi),
        normalization=dict(sorted(counters.items())),
        notes=tuple(notes),
    )
