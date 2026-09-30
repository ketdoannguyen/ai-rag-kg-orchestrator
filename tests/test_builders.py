"""Offline tests for the tree builders, neutral replay and the tree signatures.

Every fixture here is synthetic and hand-written: no book, no PDF, no network and
no database. Pages mimic the neutral artifact contract (``page_index``,
``request_page``, ``running``, ``continues_previous``, ``blocks``), and blocks mimic
what an engine mapping writes into it (``kind`` for the engine's own kind,
``normalized_kind`` for the runtime kind it maps to, and ``None`` where the engine
recorded nothing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pytest
from arango_client import ParagraphNode, StructureNode, TableNode

from config.chunking import MAX_CHUNK_SIZE
from research.ocr_pipeline import profiles
from research.ocr_pipeline.engines.marker import html_text
from research.ocr_pipeline.tree_builders import (
    BuildResult,
    MarkerStructureError,
    ReplayError,
    build_atomic,
    build_marker_structure,
    build_vision_groups,
    compare_headings,
    heading_boundaries,
    marker_compatibility,
    normalize_stream,
    page_signature,
    replay_page,
    semantic_signature,
    text_fidelity,
    text_heading_level,
    validate_tree,
)
from workflows.ingest_file.vision_ocr import VisionBlock, VisionPage
from workflows.ingest_file.vision_parser import VisionDocumentParser

TITLE = "Vật lí 12"
RUNNING_HEAD = "Vật lí 12 – Trang 5"


@dataclass
class Block:
    """One neutral block, with the artifact's optional fields left nullable."""

    kind: str
    text: str
    group: int | None = None
    depth: int | None = None
    marker: str | None = None
    normalized_kind: str | None = None


@dataclass
class Page:
    """One neutral page."""

    page_index: int
    request_page: int
    blocks: list[Block] = field(default_factory=list)
    running: str = ""
    continues_previous: bool = False


def page(index: int, *blocks: Block, continues_previous: bool = False) -> Page:
    """A page whose requested number follows the renderer's ``index + 1 + offset``."""
    return Page(
        page_index=index,
        request_page=index + 5,
        blocks=list(blocks),
        continues_previous=continues_previous,
    )


def long_text(prefix: str) -> str:
    """A paragraph long enough to survive the grouped builder's fragment net."""
    return (f"{prefix} " + "nội dung diễn giải đầy đủ cho đoạn văn này. " * 5).strip()


def details(node: StructureNode) -> list[ParagraphNode]:
    """Every paragraph below a node, in document order."""
    found: list[ParagraphNode] = []
    for child in node.children:
        if isinstance(child, StructureNode):
            found.extend(details(child))
        elif isinstance(child, ParagraphNode):
            found.append(child)
    return found


def sections(node: StructureNode) -> list[StructureNode]:
    """Every structure node below the root, in document order."""
    found: list[StructureNode] = []
    for child in node.children:
        if isinstance(child, StructureNode):
            found.append(child)
            found.extend(sections(child))
    return found


def runtime_page(source: Page) -> VisionPage:
    """The runtime page a live run would have produced for the same blocks."""
    return VisionPage(
        page=source.request_page,
        running=source.running,
        continues_previous=source.continues_previous,
        blocks=[
            VisionBlock(
                kind=block.kind,
                marker=block.marker,
                depth=block.depth,
                group=block.group or 1,
                text=block.text,
            )
            for block in source.blocks
        ],
    )


class TestReplay:
    """Replay validates the artifact instead of guessing what it meant."""

    def test_group_is_required_to_replay_grouping(self) -> None:
        pages = [page(0, Block(kind="p", text="Nội dung"))]

        with pytest.raises(ReplayError, match="group is required"):
            build_vision_groups(pages, TITLE)

    def test_missing_group_is_numbered_for_the_atomic_builder(self) -> None:
        pages = [page(0, Block(kind="p", text="Một"), Block(kind="p", text="Hai"))]

        runtime = replay_page(pages[0], require_group=False)

        assert [block.group for block in runtime.blocks] == [1, 2]

    def test_request_page_becomes_the_page_number(self) -> None:
        pages = [page(2, Block(kind="p", text=long_text("Một"), group=1))]

        result = build_atomic(pages, TITLE)

        assert result.root.metadata.start_page == 7
        assert [node.metadata.page for node in details(result.root)] == [7]

    def test_unknown_kind_is_rejected(self) -> None:
        pages = [page(0, Block(kind="sidebar", text="Nội dung", group=1))]

        with pytest.raises(ReplayError, match="not a runtime vision kind"):
            build_atomic(pages, TITLE)

    def test_blank_text_is_rejected_outside_figures(self) -> None:
        pages = [page(0, Block(kind="p", text="   ", group=1))]

        with pytest.raises(ReplayError, match="text is blank"):
            build_atomic(pages, TITLE)

    def test_textless_figure_is_allowed(self) -> None:
        pages = [page(0, Block(kind="figure", text="", group=1))]

        result = build_atomic(pages, TITLE)

        assert result.counters["textless_figures_dropped"] == 1

    def test_heading_without_depth_is_rejected(self) -> None:
        pages = [page(0, Block(kind="heading", text="Mục", group=1))]

        with pytest.raises(ReplayError, match="heading requires depth"):
            build_atomic(pages, TITLE)

    def test_negative_page_index_is_rejected(self) -> None:
        broken = Page(page_index=-1, request_page=1, blocks=[Block(kind="p", text="X")])

        with pytest.raises(ReplayError, match="page_index"):
            build_atomic([broken], TITLE)

    def test_replay_reproduces_a_direct_parser_run(self) -> None:
        """The strongest replay claim: same tree, same counters, no live call."""
        pages = [
            page(
                0,
                Block(kind="heading", text="CHƯƠNG I", group=1, depth=1, marker="I"),
                Block(kind="p", text=long_text("Mở đầu"), group=1),
                Block(kind="table", text="| a | b |\n| 1 | 2 |", group=2),
            ),
            page(
                1,
                Block(kind="list", text=long_text("Tiếp"), group=3),
                continues_previous=True,
            ),
        ]

        parser = VisionDocumentParser(TITLE)
        for source in pages:
            parser.add_page(runtime_page(source), page_number=source.request_page)
        direct = parser.finish()

        replayed = build_vision_groups(pages, TITLE)

        assert semantic_signature(replayed.root) == semantic_signature(direct)
        assert page_signature(replayed.root) == page_signature(direct)
        assert replayed.counters == dict(parser.normalization)
        assert replayed.blocks_by_kind == dict(parser.blocks_by_kind)


class TestVisionGroups:
    """The grouped builder keeps the runtime parser's joining behaviour."""

    def test_group_of_blocks_becomes_one_detail(self) -> None:
        pages = [
            page(
                0,
                Block(kind="p", text=long_text("Một"), group=1),
                Block(kind="list", text=long_text("Hai"), group=1),
            )
        ]

        result = build_vision_groups(pages, TITLE)

        assert len(details(result.root)) == 1
        assert result.counters["groups_created"] == 1

    def test_cross_page_continuation_joins_the_open_group(self) -> None:
        pages = [
            page(0, Block(kind="p", text=long_text("Một"), group=1)),
            page(1, Block(kind="p", text=long_text("Hai"), group=1), continues_previous=True),
        ]

        result = build_vision_groups(pages, TITLE)

        assert len(details(result.root)) == 1
        assert result.counters["cross_page_joins"] == 1

    def test_short_group_is_absorbed_into_the_previous_detail(self) -> None:
        """Two consecutive short chunks become one; the atomic builder keeps them apart."""
        pages = [
            page(
                0,
                Block(kind="p", text="Một.", group=1),
                Block(kind="p", text="Ngắn.", group=2),
            )
        ]

        result = build_vision_groups(pages, TITLE)

        assert result.counters["short_groups_merged"] == 1
        assert [node.attributes.text for node in details(result.root)] == ["Một.\nNgắn."]

    def test_many_blocks_per_group_collapse_but_stay_apart_in_atomic(self) -> None:
        """Nine blocks in two groups: two details for the parser, nine for atomic."""
        first = [
            f"Đoạn {index} của nhóm một kể lại sự việc theo trình tự." for index in range(1, 6)
        ]
        second = [
            f"Đoạn {index} của nhóm hai nói về kết quả sau đó và cách xử lí."
            for index in range(1, 5)
        ]
        pages = [
            page(
                0,
                *[Block(kind="p", text=text, group=1) for text in first],
                *[Block(kind="p", text=text, group=2) for text in second],
            )
        ]

        grouped = build_vision_groups(pages, TITLE)
        atomic = build_atomic(pages, TITLE)

        assert [node.attributes.text for node in details(grouped.root)] == [
            "\n".join(first),
            "\n".join(second),
        ]
        assert grouped.counters["groups_created"] == 2
        assert grouped.counters["groups_split"] == 0
        assert "short_groups_merged" not in grouped.counters  # neither join reached the net
        assert [node.attributes.text for node in details(atomic.root)] == [*first, *second]
        assert atomic.counters["details_created"] == 9
        assert "short_groups_merged" not in atomic.counters


class TestAtomic:
    """The atomic builder never joins two blocks."""

    def test_every_block_becomes_its_own_detail(self) -> None:
        pages = [
            page(
                0,
                Block(kind="p", text=long_text("Một"), group=1),
                Block(kind="p", text="Ngắn.", group=2),
                Block(kind="list", text=long_text("Ba"), group=2),
            )
        ]

        result = build_atomic(pages, TITLE)

        assert [node.attributes.text for node in details(result.root)] == [
            long_text("Một"),
            "Ngắn.",
            long_text("Ba"),
        ]
        assert result.counters["details_created"] == 3
        assert "short_groups_merged" not in result.counters

    def test_blocks_sharing_a_group_stay_apart(self) -> None:
        pages = [
            page(
                0,
                Block(kind="p", text=long_text("Một"), group=1),
                Block(kind="p", text=long_text("Hai"), group=1),
            )
        ]

        assert len(details(build_atomic(pages, TITLE).root)) == 2

    def test_pages_never_join_across_a_continuation(self) -> None:
        pages = [
            page(0, Block(kind="p", text=long_text("Một"), group=1)),
            page(1, Block(kind="p", text=long_text("Hai"), group=1), continues_previous=True),
        ]

        result = build_atomic(pages, TITLE)

        assert len(details(result.root)) == 2
        assert "cross_page_joins" not in result.counters

    def test_oversized_block_is_split_within_the_chunk_budget(self) -> None:
        oversized = "Nội dung câu này đủ dài để kiểm tra giới hạn. " * 60
        assert len(oversized) > 2 * MAX_CHUNK_SIZE

        result = build_atomic([page(0, Block(kind="p", text=oversized, group=1))], TITLE)
        texts = [node.attributes.text for node in details(result.root)]

        assert len(texts) == 3
        assert all(len(text) <= MAX_CHUNK_SIZE for text in texts)
        assert result.counters["details_split"] == 2
        assert text_fidelity(
            result.root, [page(0, Block(kind="p", text=oversized, group=1))]
        ).exact

    def test_heading_opens_a_section_and_the_table_stays_a_table(self) -> None:
        table_text = "| a | b |\n| 1 | 2 |"
        pages = [
            page(
                0,
                Block(kind="heading", text="Bài 1. Dao động", group=1, depth=2, marker="Bài 1"),
                Block(kind="table", text=table_text, group=2),
            )
        ]

        result = build_atomic(pages, TITLE)
        root = result.root

        assert [node.attributes.title for node in sections(root)] == ["Bài 1. Dao động"]
        assert [type(child) for child in root.children] == [StructureNode]
        assert isinstance(root.children[0].children[0], TableNode)
        assert root.children[0].children[0].attributes.extracted_text == table_text
        assert result.counters["tables_created"] == 1

    def test_raw_engine_kind_is_rejected(self) -> None:
        """Atomic feeds the runtime vocabulary, so the artifact must carry the mapping."""
        pages = [page(0, Block(kind="Text", text=long_text("Một"), group=1))]

        with pytest.raises(ReplayError, match="not a runtime vision kind"):
            build_atomic(pages, TITLE)


def marker_pages() -> list[Page]:
    """A synthetic Marker artifact: raw kinds only, no depths, one running head."""
    return [
        page(
            0,
            Block(kind="SectionHeader", text="CHƯƠNG I. DAO ĐỘNG CƠ", group=1),
            Block(kind="Text", text=long_text("Nội dung chương"), group=1),
            Block(kind="PageHeader", text=RUNNING_HEAD, group=1),
            Block(kind="ListGroup", text="a) Ý thứ nhất\nb) Ý thứ hai", group=2),
            Block(kind="Table", text="| đại lượng | giá trị |\n| chu kì | 2 s |", group=3),
        ),
        page(
            1,
            Block(kind="SectionHeader", text="Bài 1. Dao động điều hoà", group=1),
            Block(kind="SectionHeader", text="Mục không có mẫu", group=2),
            Block(kind="Text", text=long_text("Thân bài"), group=2),
        ),
    ]


class TestMarkerStructure:
    """Marker's structure decisions, rebuilt from the text alone."""

    def test_known_patterns_set_the_headings_level(self) -> None:
        result = build_marker_structure(marker_pages(), TITLE)

        assert [(node.level, node.attributes.title) for node in sections(result.root)] == [
            (2, "CHƯƠNG I. DAO ĐỘNG CƠ"),
            (3, "Bài 1. Dao động điều hoà"),
            (2, "Mục không có mẫu"),
        ]

    def test_pattern_levels_are_read_from_the_text(self) -> None:
        assert text_heading_level("CHƯƠNG I. DAO ĐỘNG CƠ") == 2
        assert text_heading_level("Bài 1. Dao động điều hoà") == 3
        assert text_heading_level("a) Ý thứ nhất") == 6
        assert text_heading_level("Mục không có mẫu") is None

    def test_unmatched_heading_falls_back_to_the_v1_rule(self) -> None:
        result = build_marker_structure(marker_pages(), TITLE)

        assert result.counters["marker_depth_fallback"] == 1
        assert result.counters["marker_depth_from_text"] == 2

    def test_artifact_depth_wins_over_the_text_pattern(self) -> None:
        """Both heads read as chapters, so only the adapter's depth separates them."""
        pages = [
            page(
                0,
                Block(kind="SectionHeader", text="CHƯƠNG I. DAO ĐỘNG CƠ", group=1, depth=1),
                Block(kind="SectionHeader", text="CHƯƠNG II. SÓNG CƠ", group=2, depth=3),
            )
        ]

        result = build_marker_structure(pages, TITLE)

        assert [node.level for node in sections(result.root)] == [2, 3]
        assert result.counters["marker_depth_from_artifact"] == 2
        # The parser still limits a heading to one level deeper than the open one.
        assert result.counters["depth_clamped"] == 1

    def test_running_heads_are_dropped_and_counted(self) -> None:
        result = build_marker_structure(marker_pages(), TITLE)
        titles = "\n".join(node.attributes.title for node in sections(result.root))

        assert result.counters["marker_dropped_blocks"] == 1
        assert RUNNING_HEAD not in normalize_stream(titles)

    def test_list_items_become_one_detail_each(self) -> None:
        result = build_marker_structure(marker_pages(), TITLE)

        assert result.counters["marker_list_items"] == 2
        # Three details on the first page (text, both list items) and one on the second.
        assert result.counters["details_created"] == 4

    def test_table_becomes_a_table_node(self) -> None:
        result = build_marker_structure(marker_pages(), TITLE)
        tables = [
            child
            for node in sections(result.root)
            for child in node.children
            if isinstance(child, TableNode)
        ]

        assert len(tables) == 1
        assert tables[0].attributes.rows == [["đại lượng", "giá trị"], ["chu kì", "2 s"]]

    def test_unknown_kind_is_counted_and_kept(self) -> None:
        pages = [
            page(
                0,
                Block(kind="SectionHeader", text="CHƯƠNG I", group=1),
                Block(kind="Sidebar", text=long_text("Ghi chú bên lề"), group=2),
            )
        ]

        result = build_marker_structure(pages, TITLE)

        assert result.counters["marker_unknown_blocks"] == 1
        assert result.counters["marker_unknown_kind:Sidebar"] == 1
        assert [node.attributes.text for node in details(result.root)] == [
            long_text("Ghi chú bên lề")
        ]

    def test_non_marker_artifact_is_rejected(self) -> None:
        pages = [page(0, Block(kind="p", text=long_text("Một"), group=1))]

        assert marker_compatibility(pages).compatible is False
        with pytest.raises(MarkerStructureError, match="no raw Marker block kind"):
            build_marker_structure(pages, TITLE)

    def test_compatibility_reports_the_unrecognized_kinds(self) -> None:
        pages = [
            page(
                0,
                Block(kind="SectionHeader", text="CHƯƠNG I", group=1),
                Block(kind="Sidebar", text="Ghi chú", group=2),
            )
        ]

        compatibility = marker_compatibility(pages)

        assert compatibility.compatible is True
        assert compatibility.recognized == 1
        assert compatibility.unknown_kinds == ("Sidebar",)


# ---------------------------------------------------------------------------
# Historical text-first fixture (audit F12)
#
# `build_marker_structure` reconstructs the structure the pre-a5b5080
# `TextbookParser` decided from the heading text alone. Nothing recorded those
# decisions, so "historical-exact" was a claim no test could check. This fixture
# freezes them for one synthetic artifact: the generator below is a transcription
# of the old header classifier, the HISTORICAL_* values are what it decides, and
# the RECONSTRUCTED_* values are what the reconstruction decides on the same
# artifact. Every deliberate divergence between the two is recorded in
# HISTORICAL_DIVERGENCES and proven by a test; the case fixture below registers
# the one heading rule the reconstruction reads more broadly than the old parser.
#
# It is a text-first reconstruction, never historical-exact: the old parser read
# the markup (`<h1>`, `<h2>`, `<h4>`) and the reconstruction reads the text the
# adapter kept.
# ---------------------------------------------------------------------------

HISTORICAL_PROVENANCE = {
    "path": "workflows/ingest_file/parser.py",
    "git_ref": "a5b5080^",
    "generator": "git show a5b5080^:workflows/ingest_file/parser.py",
    "transcribed": (
        "TextbookParser._classify_header, _clean_html, _get_h_tag, "
        "_is_question_or_instruction and _get_next_section_header. The h-tag "
        "pre-scan (_build_h_tag_level_map) is left out: _classify_header never "
        "consulted it."
    ),
    "claim": "text-first reconstruction, not historical-exact",
}

HISTORICAL_OVERSIZED = ("Nội dung câu này đủ dài để kiểm tra giới hạn. " * 60).strip()

_HISTORICAL_TOPIC_TITLES = ("ASEAN: NHỮNG CHẶNG ĐƯỜNG LỊCH SỬ", "VIỆT NAM VÀ ASEAN")

_HISTORICAL_INSTRUCTION_PATTERNS = (
    r"^Nêu\s+",
    r"^Trình bày\s+",
    r"^Giải thích\s+",
    r"^Phân tích\s+",
    r"^So sánh\s+",
    r"^Liệt kê\s+",
    r"^Cho biết\s+",
    r"^Hãy\s+",
    r"^Em hãy\s+",
    r"^Tại sao\s+",
    r"^Vì sao\s+",
    r"^Làm thế nào\s+",
    r"^Như thế nào\s+",
)


def _historical_clean_html(html: str) -> str:
    """`TextbookParser._clean_html`, bullet marker included."""
    if not html:
        return ""
    text = re.sub(r"<li[^>]*>", "\n* ", html)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    return text.strip()


def _historical_h_tag(html: str) -> int | None:
    match = re.search(r"<h(\d)>", html)
    return int(match.group(1)) if match else None


def _historical_is_question_or_instruction(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if text.startswith("?"):
        return True
    for pattern in _HISTORICAL_INSTRUCTION_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return text.endswith("?")


def _historical_next_section_header(
    blocks: list[dict[str, Any]], start: int
) -> dict[str, Any] | None:
    for index in range(start, len(blocks)):
        if blocks[index].get("block_type") == "SectionHeader":
            return blocks[index]
    return None


def _historical_classify_header(
    block: dict[str, Any], next_blocks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """`TextbookParser._classify_header`, without its logging."""
    block_type = block.get("block_type")
    raw_html = block.get("html", "")
    clean_text = _historical_clean_html(raw_html)
    h_tag = _historical_h_tag(raw_html)

    if block_type == "SectionHeader" and _historical_is_question_or_instruction(clean_text):
        return None

    if block_type == "SectionHeader" and "<h1>" in raw_html and "CHƯƠNG" in clean_text.upper():
        match = re.match(r"^(CHƯƠNG\s+[IVXLCDM]+)\s*(.*)", clean_text.upper(), re.DOTALL)
        number = match.group(1).strip() if match else "Chương"
        title = match.group(2).strip() if match else clean_text
        consumed = 1
        if not title and next_blocks and "<h1>" in next_blocks[0].get("html", ""):
            title = _historical_clean_html(next_blocks[0]["html"])
            consumed = 2
        return {
            "type": "Chapter",
            "title": f"{number}: {title}".upper(),
            "code": number,
            "level": 2,
            "blocks_consumed": consumed,
        }

    if block_type == "SectionHeader" and "<h1>" in raw_html:
        match = re.match(
            r"^(CHỦ\s*ĐỀ\s+\d+|CHỦ\s*ĐỂ\s+\d+)\s*(.*)", clean_text, re.IGNORECASE | re.DOTALL
        )
        if match:
            number = match.group(1).strip().upper().replace("ĐỂ", "ĐỀ")
            title = match.group(2).strip()
            return {
                "type": "Topic",
                "title": f"{number}: {title}".upper() if title else number.upper(),
                "code": number,
                "level": 2,
                "blocks_consumed": 1,
            }

    if block_type == "SectionHeader" and "<h1>" in raw_html:
        for topic in _HISTORICAL_TOPIC_TITLES:
            if topic.upper() in clean_text.upper():
                return {
                    "type": "Topic",
                    "title": clean_text.upper(),
                    "code": None,
                    "level": 2,
                    "blocks_consumed": 1,
                }

    if block_type == "SectionHeader" and "<h1>" in raw_html:
        next_header = _historical_next_section_header(next_blocks, 0)
        next_text = _historical_clean_html(next_header.get("html", "")) if next_header else ""
        if re.match(r"^Bài\s+\d+", next_text, re.IGNORECASE):
            return {
                "type": "Topic",
                "title": clean_text.upper(),
                "code": None,
                "level": 2,
                "blocks_consumed": 1,
            }

    if block_type == "SectionHeader" and "<h1>" in raw_html:
        match_bai = re.match(r"^(Bài\s+\d+)\s+(.*)", clean_text, re.IGNORECASE | re.DOTALL)
        if match_bai:
            code = match_bai.group(1).strip()
            title = match_bai.group(2).strip()
            return {
                "type": "Lesson",
                "title": f"{code}: {title}".upper(),
                "code": code,
                "level": 3,
                "blocks_consumed": 1,
            }

    if block_type == "PageHeader" and clean_text.upper() == "BÀI" and len(next_blocks) >= 2:
        number_block = next_blocks[0]
        title_block = next_blocks[1]
        number = _historical_clean_html(number_block.get("html", ""))
        if number_block.get("block_type") == "PageHeader" and number.isdigit():
            title = re.sub(r"\s+", " ", _historical_clean_html(title_block.get("html", "")))
            return {
                "type": "Lesson",
                "title": f"Bài {number}: {title.upper()}",
                "code": f"Bài {number}",
                "level": 3,
                "blocks_consumed": 3,
            }

    if block_type == "SectionHeader":
        match_roman = re.match(r"^\s*([IVXLCDM]+)\.\s+(.*)", clean_text, re.DOTALL)
        if match_roman and "ĐỊNH LUẬT" not in match_roman.group(1).upper():
            return {
                "type": "Section",
                "title": match_roman.group(2).strip(),
                "code": match_roman.group(1).upper(),
                "level": 4,
                "blocks_consumed": 1,
            }

        match_number = re.match(r"^\s*(\d+)\.\s+(.*)", clean_text, re.DOTALL)
        if match_number:
            return {
                "type": "Section",
                "title": match_number.group(2).strip(),
                "code": match_number.group(1),
                "level": 5,
                "blocks_consumed": 1,
            }

        match_letter = re.match(r"^\s*([a-z])\)\s+(.*)", clean_text, re.DOTALL)
        if match_letter:
            return {
                "type": "Section",
                "title": match_letter.group(2).strip(),
                "code": match_letter.group(1).lower(),
                "level": 6,
                "blocks_consumed": 1,
            }

        if h_tag and h_tag >= 3:
            return None

    return None


def historical_artifact() -> list[dict[str, Any]]:
    """A synthetic Marker artifact covering the decisions the fixture pins.

    It carries the markup the old parser classified (an h1 chapter, an h1 topic,
    an h1 lesson with a `.` after its number, an h2 roman section, an h2 and an h4
    arabic one, an h4 lettered one, an unmatched h1), a running head, a table, a
    list group with its items, a text over the chunk ceiling, a short line and a
    page whose first block continues the previous page.
    """
    return [
        {
            "page": 0,
            "children": [
                {
                    "id": "/page/0/SectionHeader/1",
                    "block_type": "SectionHeader",
                    "html": "<h1>CHƯƠNG I. DAO ĐỘNG CƠ</h1>",
                },
                {
                    "id": "/page/0/SectionHeader/2",
                    "block_type": "SectionHeader",
                    "html": "<h1>CHỦ ĐỀ 1. TRẬT TỰ THẾ GIỚI</h1>",
                },
                {
                    "id": "/page/0/Text/3",
                    "block_type": "Text",
                    "html": "<p>Dao động cơ là chuyển động qua lại quanh vị trí cân bằng.</p>",
                },
                {
                    "id": "/page/0/PageHeader/4",
                    "block_type": "PageHeader",
                    "html": "<p>Vật lí 12 – Trang 5</p>",
                },
                {
                    "id": "/page/0/Table/5",
                    "block_type": "Table",
                    "html": "<table><tr><td>Mốc</td><td>Sự kiện</td></tr>"
                    "<tr><td>1945</td><td>Liên hợp quốc</td></tr></table>",
                },
                {
                    "id": "/page/0/ListGroup/6",
                    "block_type": "ListGroup",
                    "html": "<ul><li>Dao động tuần hoàn.</li><li>Dao động điều hoà.</li></ul>",
                    "children": [
                        {
                            "id": "/page/0/ListItem/7",
                            "block_type": "ListItem",
                            "html": "<li>Dao động tuần hoàn.</li>",
                        },
                        {
                            "id": "/page/0/ListItem/8",
                            "block_type": "ListItem",
                            "html": "<li>Dao động điều hoà.</li>",
                        },
                    ],
                },
            ],
        },
        {
            "page": 1,
            "children": [
                {
                    "id": "/page/1/Text/1",
                    "block_type": "Text",
                    "html": "<p>Nội dung tiếp theo ở trang sau, không nối với trang trước.</p>",
                },
                {
                    "id": "/page/1/SectionHeader/2",
                    "block_type": "SectionHeader",
                    "html": "<h1>Bài 1. DAO ĐỘNG ĐIỀU HOÀ</h1>",
                },
                {
                    "id": "/page/1/Text/3",
                    "block_type": "Text",
                    "html": "<p>Nội dung bài học mở đầu.</p>",
                },
                {
                    "id": "/page/1/SectionHeader/4",
                    "block_type": "SectionHeader",
                    "html": "<h2>I. SỰ HÌNH THÀNH TRẬT TỰ</h2>",
                },
                {
                    "id": "/page/1/Text/5",
                    "block_type": "Text",
                    "html": "<p>Một đoạn ngắn.</p>",
                },
                {
                    "id": "/page/1/SectionHeader/6",
                    "block_type": "SectionHeader",
                    "html": "<h2>2. Bối cảnh thế giới</h2>",
                },
                {
                    "id": "/page/1/Text/7",
                    "block_type": "Text",
                    "html": "<p>Bối cảnh sau chiến tranh.</p>",
                },
                {
                    "id": "/page/1/SectionHeader/8",
                    "block_type": "SectionHeader",
                    "html": "<h4>3. Tình hình Việt Nam</h4>",
                },
                {
                    "id": "/page/1/Text/9",
                    "block_type": "Text",
                    "html": "<p>Tình hình trong nước.</p>",
                },
                {
                    "id": "/page/1/SectionHeader/10",
                    "block_type": "SectionHeader",
                    "html": "<h4>a) Mục nhỏ</h4>",
                },
                {
                    "id": "/page/1/Text/11",
                    "block_type": "Text",
                    "html": f"<p>{HISTORICAL_OVERSIZED}</p>",
                },
            ],
        },
        {
            "page": 2,
            "children": [
                {
                    "id": "/page/2/SectionHeader/1",
                    "block_type": "SectionHeader",
                    "html": "<h1>Mục không có mẫu</h1>",
                },
                {
                    "id": "/page/2/Text/2",
                    "block_type": "Text",
                    "html": "<p>Phần thân của mục không có mẫu.</p>",
                },
            ],
        },
    ]


def historical_pages(artifact: list[dict[str, Any]] | None = None) -> list[Page]:
    """The neutral artifact the reconstruction reads, derived from the same html.

    The text comes from the engine's own converter, so the two sides are compared
    on what the pipeline would really have carried, not on a second transcription.
    ``artifact`` defaults to the historical fixture; the case fixture below passes
    its own html through the same conversion.
    """
    pages: list[Page] = []
    for index, raw in enumerate(historical_artifact() if artifact is None else artifact):
        blocks: list[Block] = []
        for block in raw["children"]:
            text = html_text(block.get("html"))
            if text:
                blocks.append(Block(kind=block["block_type"], text=text))
        pages.append(
            Page(
                page_index=index,
                request_page=index + 5,
                blocks=blocks,
                continues_previous=index > 0,
            )
        )
    return pages


def _historical_structure(
    artifact: list[dict[str, Any]] | None = None,
) -> tuple[tuple[tuple[str, int, str], ...], tuple[str, ...]]:
    """What the old parser decided for the fixture artifact.

    Sections are ``(source heading text, level, title the parser wrote)`` and
    details are the text of every node it appended, in document order. The walk
    mirrors ``TextbookParser.parse``: a classified header opens a section and
    consumes the blocks it claimed, and a block that is neither a classified
    header nor a ListGroup/Text/Caption contributes nothing at all.
    """
    found_sections: list[tuple[str, int, str]] = []
    found_details: list[str] = []
    for page in historical_artifact() if artifact is None else artifact:
        blocks = page["children"]
        index = 0
        while index < len(blocks):
            block = blocks[index]
            info = _historical_classify_header(block, blocks[index + 1 :])
            if info:
                found_sections.append(
                    (_historical_clean_html(block.get("html", "")), info["level"], info["title"])
                )
                index += info["blocks_consumed"]
                continue
            kind = block.get("block_type")
            if kind == "ListGroup":
                children = block.get("children", [])
                if children:
                    for child in children:
                        if child.get("block_type") == "ListItem":
                            text = _historical_clean_html(child.get("html", ""))
                            if text:
                                found_details.append(text)
                else:
                    text = _historical_clean_html(block.get("html", ""))
                    if text:
                        found_details.append(text)
            elif kind in ("Text", "Caption"):
                text = _historical_clean_html(block.get("html", ""))
                if text:
                    found_details.append(text)
            index += 1
    return tuple(found_sections), tuple(found_details)


HISTORICAL_SECTIONS: tuple[tuple[str, int, str], ...] = (
    ("CHƯƠNG I. DAO ĐỘNG CƠ", 2, "CHƯƠNG I: . DAO ĐỘNG CƠ"),
    ("CHỦ ĐỀ 1. TRẬT TỰ THẾ GIỚI", 2, "CHỦ ĐỀ 1: . TRẬT TỰ THẾ GIỚI"),
    ("I. SỰ HÌNH THÀNH TRẬT TỰ", 4, "SỰ HÌNH THÀNH TRẬT TỰ"),
    ("2. Bối cảnh thế giới", 5, "Bối cảnh thế giới"),
    ("3. Tình hình Việt Nam", 5, "Tình hình Việt Nam"),
    ("a) Mục nhỏ", 6, "Mục nhỏ"),
)
"""The old parser's sections: source heading text, level opened, title written."""

HISTORICAL_DETAILS: tuple[str, ...] = (
    "Dao động cơ là chuyển động qua lại quanh vị trí cân bằng.",
    "* Dao động tuần hoàn.",
    "* Dao động điều hoà.",
    "Nội dung tiếp theo ở trang sau, không nối với trang trước.",
    "Nội dung bài học mở đầu.",
    "Một đoạn ngắn.",
    "Bối cảnh sau chiến tranh.",
    "Tình hình trong nước.",
    HISTORICAL_OVERSIZED,
    "Phần thân của mục không có mẫu.",
)
"""The old parser's detail texts, in document order."""

RECONSTRUCTED_SECTIONS: tuple[tuple[str, int], ...] = (
    ("CHƯƠNG I. DAO ĐỘNG CƠ", 2),
    ("CHỦ ĐỀ 1. TRẬT TỰ THẾ GIỚI", 2),
    ("Bài 1. DAO ĐỘNG ĐIỀU HOÀ", 3),
    ("I. SỰ HÌNH THÀNH TRẬT TỰ", 4),
    ("2. Bối cảnh thế giới", 4),
    ("3. Tình hình Việt Nam", 4),
    ("a) Mục nhỏ", 5),
    ("Mục không có mẫu", 2),
)
"""The reconstruction's sections on the same artifact: title, level opened."""

HISTORICAL_LEVEL_DIVERGENCES: tuple[tuple[str, int | None, int | None], ...] = (
    ("Bài 1. DAO ĐỘNG ĐIỀU HOÀ", None, 3),
    ("2. Bối cảnh thế giới", 5, 4),
    ("3. Tình hình Việt Nam", 5, 4),
    ("a) Mục nhỏ", 6, 5),
    ("Mục không có mẫu", None, 2),
)
"""`(source heading, level the old parser opened, level the reconstruction opens)`.

``None`` means the old parser did not open a section for that heading at all.
"""

HISTORICAL_DIVERGENCES: tuple[tuple[str, str, str], ...] = (
    ("a table block", "dropped", "kept as a TableNode"),
    (
        "a list item",
        "the `<li>` rewritten to a `* ` bullet in the text",
        "the item text alone",
    ),
    ("a text over the chunk ceiling", "one detail of any length", "split at MAX_CHUNK_SIZE"),
    (
        "a CHƯƠNG / CHỦ ĐỀ title",
        "rewritten as `CODE: TITLE`, uppercase",
        "the source text as it stands",
    ),
    (
        "a lowercase roman or uppercase lettered heading",
        "the pattern required its own case, so no section was opened",
        "matched case-insensitively, so a section is opened",
    ),
)
"""`(aspect, what the old parser did, what the reconstruction does)`."""


def _divergence(aspect: str) -> tuple[str, str, str]:
    """The recorded divergence for one aspect, so a test proves the record is real."""
    for entry in HISTORICAL_DIVERGENCES:
        if entry[0] == aspect:
            return entry
    raise AssertionError(f"no recorded divergence for {aspect!r}")


# ---------------------------------------------------------------------------
# Case fixture (audit F6)
#
# `_HEADING_PATTERNS` compiles the Roman and lettered patterns with
# `re.IGNORECASE`, while the old `_classify_header` matched them in the case it
# wrote them in (`[IVXLCDM]+\.` uppercase, `[a-z]\)` lowercase). This artifact
# holds one heading of each kind, so the divergence is measured rather than
# described.
# ---------------------------------------------------------------------------


def case_divergence_artifact() -> list[dict[str, Any]]:
    """One lowercase Roman heading and one uppercase lettered heading."""
    return [
        {
            "page": 0,
            "children": [
                {
                    "id": "/page/0/SectionHeader/1",
                    "block_type": "SectionHeader",
                    "html": "<h2>v. Mục La Mã viết thường</h2>",
                },
                {
                    "id": "/page/0/SectionHeader/2",
                    "block_type": "SectionHeader",
                    "html": "<h4>A) Mục chữ cái viết hoa</h4>",
                },
            ],
        }
    ]


CASE_DIVERGENCE_HEADINGS: tuple[tuple[str, int], ...] = (
    ("v. Mục La Mã viết thường", 4),
    ("A) Mục chữ cái viết hoa", 6),
)
"""`(heading text, level the reconstruction's pattern reads from it)`."""


def detail_nodes(node: StructureNode) -> list[tuple[str, str]]:
    """Every detail below a node, in document order, as ``(kind, text)``."""
    found: list[tuple[str, str]] = []
    for child in node.children:
        if isinstance(child, StructureNode):
            found.extend(detail_nodes(child))
        elif isinstance(child, TableNode):
            found.append(("table", child.attributes.extracted_text or ""))
        elif isinstance(child, ParagraphNode):
            found.append(("paragraph", child.attributes.text or ""))
    return found


class TestHistoricalFixture:
    """The reconstruction against what the old parser decided, not against a claim."""

    def test_the_generator_reproduces_the_recorded_goldens(self) -> None:
        found_sections, found_details = _historical_structure()

        assert found_sections == HISTORICAL_SECTIONS
        assert found_details == HISTORICAL_DETAILS

    def test_the_reconstruction_reproduces_the_recorded_goldens(self) -> None:
        result = build_marker_structure(historical_pages(), TITLE)

        assert [(node.attributes.title, node.level) for node in sections(result.root)] == list(
            RECONSTRUCTED_SECTIONS
        )

    def test_the_reconstruction_keeps_the_historical_details_in_order(self) -> None:
        _, historical = _historical_structure()
        found = detail_nodes(build_marker_structure(historical_pages(), TITLE).root)
        texts = [text for _, text in found]

        assert len(found) == len(historical) + 3  # the table, and two split pieces
        assert texts[0] == historical[0]
        assert found[1] == ("table", "Mốc | Sự kiện\n1945 | Liên hợp quốc")
        assert texts[2:9] == [text.lstrip("* ") for text in historical[1:8]]
        assert texts[12] == historical[9]
        # The split re-flows whitespace at its boundaries (units are joined with a
        # newline, and the space a sentence split dropped is gone), so the pieces
        # are compared as words, not as the byte string the old parser kept.
        assert " ".join(" ".join(texts[9:12]).split()) == " ".join(historical[8].split())

    def test_the_patterned_headings_keep_the_historical_levels(self) -> None:
        """Where both patterns read the same spelling, the levels agree.

        Roman and lettered headings are the exception: the reconstruction reads
        them case-insensitively and the old parser did not, so neither is claimed
        for the other (``TestCaseDivergence``).
        """
        historical = {title: level for title, level, _ in HISTORICAL_SECTIONS}
        reconstructed = dict(RECONSTRUCTED_SECTIONS)

        for title in (
            "CHƯƠNG I. DAO ĐỘNG CƠ",
            "CHỦ ĐỀ 1. TRẬT TỰ THẾ GIỚI",
            "I. SỰ HÌNH THÀNH TRẬT TỰ",
        ):
            assert reconstructed[title] == historical[title]

    def test_the_level_divergences_are_exactly_the_recorded_ones(self) -> None:
        historical = {title: level for title, level, _ in HISTORICAL_SECTIONS}
        reconstructed = dict(RECONSTRUCTED_SECTIONS)
        found = {
            title: (historical.get(title), reconstructed.get(title))
            for title in set(historical) | set(reconstructed)
            if historical.get(title) != reconstructed.get(title)
        }
        recorded = {title: (old, new) for title, old, new in HISTORICAL_LEVEL_DIVERGENCES}

        assert found == recorded

    def test_a_table_is_dropped_historically_and_kept_by_the_reconstruction(self) -> None:
        assert _divergence("a table block") == (
            "a table block",
            "dropped",
            "kept as a TableNode",
        )
        _, historical = _historical_structure()
        found = detail_nodes(build_marker_structure(historical_pages(), TITLE).root)

        assert not any("Liên hợp quốc" in text for text in historical)
        assert ("table", "Mốc | Sự kiện\n1945 | Liên hợp quốc") in found

    def test_the_historical_bullet_is_not_reconstructed(self) -> None:
        assert _divergence("a list item") == (
            "a list item",
            "the `<li>` rewritten to a `* ` bullet in the text",
            "the item text alone",
        )
        _, historical = _historical_structure()
        texts = [
            text
            for _, text in detail_nodes(build_marker_structure(historical_pages(), TITLE).root)
        ]

        assert "* Dao động tuần hoàn." in historical
        assert "Dao động tuần hoàn." in texts
        assert "* Dao động tuần hoàn." not in texts

    def test_the_ceiling_is_the_reconstruction_only(self) -> None:
        assert _divergence("a text over the chunk ceiling") == (
            "a text over the chunk ceiling",
            "one detail of any length",
            "split at MAX_CHUNK_SIZE",
        )
        _, historical = _historical_structure()
        result = build_marker_structure(historical_pages(), TITLE)
        texts = [text for _, text in detail_nodes(result.root)]

        assert len(historical[8]) > MAX_CHUNK_SIZE  # the old parser had no ceiling
        assert all(len(text) <= MAX_CHUNK_SIZE for text in texts)
        assert result.counters["details_split"] == 2

    def test_a_heading_title_is_never_rewritten(self) -> None:
        assert _divergence("a CHƯƠNG / CHỦ ĐỀ title") == (
            "a CHƯƠNG / CHỦ ĐỀ title",
            "rewritten as `CODE: TITLE`, uppercase",
            "the source text as it stands",
        )
        historical_titles = {title: written for title, _, written in HISTORICAL_SECTIONS}
        reconstructed = {
            node.attributes.title
            for node in sections(build_marker_structure(historical_pages(), TITLE).root)
        }

        assert historical_titles["CHƯƠNG I. DAO ĐỘNG CƠ"] == "CHƯƠNG I: . DAO ĐỘNG CƠ"
        assert "CHƯƠNG I. DAO ĐỘNG CƠ" in reconstructed
        assert "CHƯƠNG I: . DAO ĐỘNG CƠ" not in reconstructed

    def test_neither_parser_joins_a_detail_across_a_page_break(self) -> None:
        pages = historical_pages()
        assert pages[1].continues_previous is True
        _, historical = _historical_structure()
        texts = [text for _, text in detail_nodes(build_marker_structure(pages, TITLE).root)]

        assert texts[3] == historical[2].lstrip("* ") == "Dao động điều hoà."
        assert (
            texts[4]
            == historical[3]
            == "Nội dung tiếp theo ở trang sau, không nối với trang trước."
        )

    def test_a_short_line_stays_its_own_detail_in_both(self) -> None:
        """Neither side absorbs a fragment into its neighbour."""
        _, historical = _historical_structure()
        texts = [
            text
            for _, text in detail_nodes(build_marker_structure(historical_pages(), TITLE).root)
        ]

        assert texts[5] == historical[4] == "Nội dung bài học mở đầu."
        assert texts[6] == historical[5] == "Một đoạn ngắn."

    def test_the_fixture_does_not_claim_historical_exactness(self) -> None:
        assert HISTORICAL_PROVENANCE["git_ref"] == "a5b5080^"
        assert "not historical-exact" in HISTORICAL_PROVENANCE["claim"]


class TestCaseDivergence:
    """The heading rule the reconstruction reads more broadly than the old parser.

    Both sides read the same artifact, so the difference is the pattern alone:
    the old one matched `v.` and `A)` in no case at all.
    """

    ASPECT = "a lowercase roman or uppercase lettered heading"

    def test_the_historical_oracle_opens_no_section_for_either_heading(self) -> None:
        found_sections, found_details = _historical_structure(case_divergence_artifact())

        assert found_sections == ()
        assert found_details == ()

    def test_the_reconstruction_opens_a_section_for_both_headings(self) -> None:
        pages = historical_pages(case_divergence_artifact())

        result = build_marker_structure(pages, TITLE)

        assert [node.attributes.title for node in sections(result.root)] == [
            heading for heading, _ in CASE_DIVERGENCE_HEADINGS
        ]
        # Both levels come from the text: neither heading fell back to v1.
        assert result.counters["marker_depth_from_text"] == 2
        assert "marker_depth_fallback" not in result.counters

    def test_the_patterns_read_the_heading_levels_case_insensitively(self) -> None:
        found = [(heading, text_heading_level(heading)) for heading, _ in CASE_DIVERGENCE_HEADINGS]

        assert found == list(CASE_DIVERGENCE_HEADINGS)

    def test_the_divergence_is_recorded(self) -> None:
        assert _divergence(self.ASPECT) == (
            self.ASPECT,
            "the pattern required its own case, so no section was opened",
            "matched case-insensitively, so a section is opened",
        )


class TestBoundariesAndFidelity:
    """The comparison the Marker reconstruction exists to feed."""

    def test_boundaries_list_every_section_with_its_page(self) -> None:
        boundaries = heading_boundaries(build_marker_structure(marker_pages(), TITLE).root)

        assert [(b.level, b.title, b.start_page) for b in boundaries] == [
            (2, "CHƯƠNG I. DAO ĐỘNG CƠ", 5),
            (3, "Bài 1. Dao động điều hoà", 6),
            (2, "Mục không có mẫu", 6),
        ]

    def test_compare_headings_reports_a_level_mismatch(self) -> None:
        """Same page, two artifacts: the model stated a depth, Marker's text did not."""
        marker = build_marker_structure(marker_pages(), TITLE).root
        vision_pages = [
            page(
                0,
                Block(kind="heading", text="CHƯƠNG I. DAO ĐỘNG CƠ", group=1, depth=1),
                Block(kind="heading", text="Mục không có mẫu", group=2, depth=2),
            )
        ]
        atomic = build_atomic(vision_pages, TITLE).root

        comparison = compare_headings(marker, atomic)

        assert comparison.shared == ("CHƯƠNG I. DAO ĐỘNG CƠ", "Mục không có mẫu")
        assert comparison.level_mismatches == (("Mục không có mẫu", 2, 3),)
        assert comparison.only_in_first == ("Bài 1. Dao động điều hoà",)
        assert comparison.only_in_second == ()
        assert comparison.jaccard == pytest.approx(2 / 3)

    def test_compare_headings_of_two_empty_trees_is_not_measured(self) -> None:
        empty = [page(0, Block(kind="p", text=long_text("Một"), group=1))]

        comparison = compare_headings(
            build_atomic(empty, TITLE).root, build_atomic(empty, TITLE).root
        )

        assert comparison.jaccard is None
        assert comparison.shared == ()

    def test_fidelity_is_exact_when_nothing_is_dropped(self) -> None:
        pages = [
            page(
                0,
                Block(kind="heading", text="Bài 1. Dao động", group=1, depth=2),
                Block(kind="p", text=long_text("Một"), group=1),
                Block(kind="table", text="| a | b |\n| 1 | 2 |", group=2),
            )
        ]

        fidelity = text_fidelity(build_atomic(pages, TITLE).root, pages)

        assert fidelity.exact is True
        assert fidelity.artifact_chars == fidelity.tree_chars

    def test_fidelity_counts_a_dropped_running_head(self) -> None:
        pages = marker_pages()

        marker = text_fidelity(build_marker_structure(pages, TITLE).root, pages)

        # The separator that followed the head collapses into the one before it, so
        # the drop costs one character more than the head's own text.
        assert marker.omitted == len(RUNNING_HEAD) + 1
        assert marker.added == 0
        assert marker.omitted_ratio == pytest.approx(
            (len(RUNNING_HEAD) + 1) / marker.artifact_chars
        )

    def test_fidelity_separates_a_move_from_a_loss(self) -> None:
        pages = [
            page(
                0,
                Block(kind="p", text="Alpha beta gamma", group=1),
                Block(kind="p", text="Delta epsilon", group=2),
            )
        ]
        result = build_atomic(pages, TITLE)
        first, second = details(result.root)
        first.attributes.text = "Delta epsilon"
        second.attributes.text = "Alpha beta gamma"

        fidelity = text_fidelity(result.root, pages)

        assert fidelity.omitted == 0
        assert fidelity.added == 0
        assert fidelity.reordered > 0
        assert fidelity.exact is False

    def test_fidelity_of_an_untouched_artifact_is_measured(self) -> None:
        fidelity = text_fidelity(build_atomic([], TITLE).root, [])

        assert fidelity.artifact_chars == 0
        assert fidelity.omitted_ratio is None


class TestSignatures:
    """The signatures a run is scored on never see an id or a timestamp."""

    def test_signature_is_stable_across_two_builds(self) -> None:
        pages = [
            page(
                0,
                Block(kind="heading", text="Bài 1", group=1, depth=2),
                Block(kind="p", text=long_text("Một"), group=1),
            )
        ]

        first = build_atomic(pages, TITLE).root
        second = build_atomic(pages, TITLE).root

        assert first.id != second.id
        assert semantic_signature(first) == semantic_signature(second)
        assert page_signature(first) == page_signature(second)

    def test_semantic_signature_ignores_page_numbers(self) -> None:
        blocks = [Block(kind="p", text=long_text("Một"), group=1)]

        first = build_atomic([page(0, *blocks)], TITLE).root
        second = build_atomic([Page(page_index=3, request_page=42, blocks=blocks)], TITLE).root

        assert semantic_signature(first) == semantic_signature(second)
        assert page_signature(first) != page_signature(second)

    def test_semantic_signature_ignores_runtime_extras(self) -> None:
        """Ids, embeddings, summaries and positions are runtime state, not content."""
        pages = [page(0, Block(kind="p", text=long_text("Một"), group=1))]
        result = build_atomic(pages, TITLE)
        detail = result.root.children[0]

        before = semantic_signature(result.root)
        result.root.summary = "Tóm tắt do LLM viết"
        result.root.embedding = [0.5] * 64
        detail.parent_id = result.root.id
        detail.metadata.position = 7

        assert semantic_signature(result.root) == before

    def test_validation_does_not_move_the_semantic_signature(self) -> None:
        pages = [
            page(
                0,
                Block(kind="heading", text="Bài 1", group=1, depth=2),
                Block(kind="p", text=long_text("Một"), group=1),
            )
        ]
        root = build_atomic(pages, TITLE).root
        before = semantic_signature(root)

        statistics = validate_tree(root)

        assert semantic_signature(root) == before
        assert statistics["total_nodes"] == 3
        assert statistics["nodes_by_level"] == {1: 1, 2: 1, 0: 1}
        assert root.parent_id is None
        assert root.children[0].parent_id == root.id


class TestProfiles:
    """The table is the only place that maps a profile name to a builder."""

    def test_every_plan_profile_is_present(self) -> None:
        assert sorted(profiles.PROFILES) == [
            "docling_atomic",
            "marker_atomic",
            "marker_llm",
            "native_atomic",
            "vision_atomic",
            "vision_atomic_merge",
            "vision_direct",
        ]

    @pytest.mark.parametrize(
        ("name", "builder"),
        [
            ("vision_direct", "build_vision_groups"),
            ("vision_atomic", "build_atomic"),
            ("marker_atomic", "build_atomic"),
            ("marker_llm", "build_marker_structure"),
        ],
    )
    def test_builder_resolution(self, name: str, builder: str) -> None:
        assert profiles.builder_for(name).__name__ == builder

    def test_build_runs_the_profiles_builder(self) -> None:
        pages = [page(0, Block(kind="p", text=long_text("Một"), group=1))]

        result = profiles.build("vision_atomic", pages, TITLE)

        assert isinstance(result, BuildResult)
        assert result.builder == "atomic"
        assert len(details(result.root)) == 1

    def test_unknown_profile_lists_the_table(self) -> None:
        with pytest.raises(ValueError, match="unknown profile 'vision_direct_live'"):
            profiles.get_profile("vision_direct_live")

    def test_no_profile_runs_the_structure_summary(self) -> None:
        assert all(profile.structure_summary is False for profile in profiles.PROFILES.values())

    def test_merge_policy_follows_the_plan_table(self) -> None:
        assert profiles.get_profile("vision_direct").merge == "off"
        assert profiles.get_profile("marker_llm").merge == "real"
        # A merge is run against a live endpoint or not at all: no profile replays one.
        assert {profile.merge for profile in profiles.PROFILES.values()} == {"off", "real"}
