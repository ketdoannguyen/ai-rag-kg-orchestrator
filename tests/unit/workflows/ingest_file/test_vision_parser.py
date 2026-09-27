"""Offline tests for semantic normalization and hierarchy assembly.

The parser is fed ``VisionPage`` objects directly, so these tests describe what
the tree looks like for a given model output without any transport involved.
"""

from __future__ import annotations

import pytest
from arango_client import ParagraphNode, StructureNode, TableNode

from config.chunking import MAX_CHUNK_SIZE
from workflows.ingest_file.vision_ocr import VisionBlock, VisionPage
from workflows.ingest_file.vision_parser import VisionDocumentParser


def _heading(text: str, marker: str | None = None, depth: int = 1, group: int = 1) -> VisionBlock:
    return VisionBlock(kind="heading", marker=marker, depth=depth, group=group, text=text)


def _block(kind: str, text: str, group: int = 1) -> VisionBlock:
    return VisionBlock(kind=kind, group=group, text=text)


def _page(page: int, *blocks: VisionBlock, continues_previous: bool = False) -> VisionPage:
    return VisionPage(page=page, blocks=list(blocks), continues_previous=continues_previous)


def _long(prefix: str) -> str:
    """A realistic paragraph long enough to survive the fragment safety net."""
    return (f"{prefix} " + "nội dung diễn giải đầy đủ cho đoạn văn này. " * 5).strip()


def _titles(node: StructureNode) -> list[tuple[int, str]]:
    """Depth-first ``(level, title)`` listing of every structure node."""
    found = [(node.level, node.attributes.title)]
    for child in node.children:
        if isinstance(child, StructureNode):
            found.extend(_titles(child))
    return found


def _parent_of(root: StructureNode, title: str) -> str | None:
    for child in root.children:
        if isinstance(child, StructureNode):
            if child.attributes.title == title:
                return root.attributes.title
            found = _parent_of(child, title)
            if found is not None:
                return found
    return None


class TestNormalization:
    """Blocks the model mislabels are reclassified before they touch the tree."""

    def test_callout_label_never_becomes_a_section(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _heading("EM ĐÃ HỌC"), _block("p", "Nội dung")), page_number=1)
        root = parser.finish()

        assert parser.state["open_headings"] == []
        assert parser.normalization["callouts_reclassified"] == 1
        assert parser.blocks_by_kind["callout"] == 1
        assert _titles(root) == [(1, "Vật lí 12")]

    def test_unmarked_instruction_is_a_question(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, _heading("Giải thích hiện tượng nóng chảy"), _block("p", "Trả lời")),
            page_number=1,
        )
        root = parser.finish()

        assert parser.state["open_headings"] == []
        assert parser.normalization["questions_reclassified"] == 1
        assert _titles(root) == [(1, "Vật lí 12")]

    def test_marked_instruction_verb_stays_a_section(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, _heading("b) Giải thích sự nóng chảy của chất rắn kết tinh", marker="b")),
            page_number=1,
        )
        root = parser.finish()

        assert _titles(root) == [
            (1, "Vật lí 12"),
            (2, "b) Giải thích sự nóng chảy của chất rắn kết tinh"),
        ]
        assert parser.state["open_headings"] == [
            {"marker": "b", "depth": 1, "text": "b) Giải thích sự nóng chảy của chất rắn kết tinh"}
        ]

    def test_trailing_question_mark_is_a_question(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, _heading("Thế nào là dao động điều hoà?", marker="1.2")), page_number=1
        )
        root = parser.finish()

        assert parser.state["open_headings"] == []
        assert root.children[0].attributes.text == "Thế nào là dao động điều hoà?"

    def test_lettered_marker_is_not_tied_to_a_depth(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, _heading("1 Nội dung", marker="1", depth=1)),
            page_number=1,
        )
        parser.add_page(
            _page(2, _heading("a) Trường hợp riêng", marker="a", depth=2)),
            page_number=2,
        )
        root = parser.finish()

        assert _titles(root) == [(1, "Vật lí 12"), (2, "1 Nội dung"), (3, "a) Trường hợp riêng")]


class TestHierarchy:
    """Depth, clamping, duplicates and page spans follow the documented rules."""

    def test_depth_is_relative_to_the_root(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(
                1,
                _heading("Chương 1", depth=1),
                _heading("Bài 1", depth=2),
                _heading("1.1 Mục nhỏ", depth=3),
            ),
            page_number=1,
        )
        root = parser.finish()

        assert _titles(root) == [
            (1, "Vật lí 12"),
            (2, "Chương 1"),
            (3, "Bài 1"),
            (4, "1.1 Mục nhỏ"),
        ]

    def test_depth_gap_is_clamped(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, _heading("Chương 1", depth=1), _heading("Mục sâu", depth=6)),
            page_number=1,
        )
        root = parser.finish()

        assert parser.normalization["depth_clamped"] == 1
        assert _titles(root) == [(1, "Vật lí 12"), (2, "Chương 1"), (3, "Mục sâu")]

    def test_sibling_pops_back_to_the_same_parent(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(
                1,
                _heading("Chương 1", depth=1),
                _heading("Bài 1", depth=2),
                _heading("Bài 2", depth=2),
            ),
            page_number=1,
        )
        root = parser.finish()

        assert _titles(root) == [(1, "Vật lí 12"), (2, "Chương 1"), (3, "Bài 1"), (3, "Bài 2")]
        assert _parent_of(root, "Bài 2") == "Chương 1"

    def test_repeated_heading_is_not_duplicated(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _heading("Chương 1", depth=1)), page_number=1)
        parser.add_page(_page(2, _heading("Chương 1", depth=1)), page_number=2)
        root = parser.finish()

        assert parser.normalization["duplicate_headings"] == 1
        assert _titles(root) == [(1, "Vật lí 12"), (2, "Chương 1")]

    def test_state_survives_page_boundaries(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, _heading("Chương 1", depth=1), _heading("Bài 1", depth=2)),
            page_number=1,
        )
        # ``code`` is derived from the title by StructureAttributes when the model
        # sends no marker, so the state still carries something the model can match.
        assert parser.state["open_headings"] == [
            {"marker": "Chương 1", "depth": 1, "text": "Chương 1"},
            {"marker": "Bài 1", "depth": 2, "text": "Bài 1"},
        ]

        parser.add_page(_page(2, _heading("1.1 Mục nhỏ", marker="1.1", depth=3)), page_number=2)
        root = parser.finish()

        assert _parent_of(root, "1.1 Mục nhỏ") == "Bài 1"

    def test_page_spans_are_recorded(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(3, _heading("Chương 1", depth=1)), page_number=3)
        parser.add_page(_page(4, _heading("Chương 2", depth=1)), page_number=4)
        root = parser.finish()

        chapter1, chapter2 = root.children
        assert root.metadata.start_page == 3
        assert root.metadata.end_page == 4
        assert (chapter1.metadata.start_page, chapter1.metadata.end_page) == (3, 4)
        assert (chapter2.metadata.start_page, chapter2.metadata.end_page) == (4, 4)


class TestContent:
    """Lists, tables and text-first blocks become valid detail nodes."""

    def test_blocks_sharing_a_group_become_one_paragraph(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(
                1,
                _heading("Chương 1", depth=1),
                _block("list", "- Ý một", group=2),
                _block("list", "- Ý hai", group=2),
                _block("list", "- Ý ba", group=2),
            ),
            page_number=1,
        )
        root = parser.finish()

        items = root.children[0].children
        assert [item.attributes.text for item in items] == ["- Ý một\n- Ý hai\n- Ý ba"]
        assert parser.normalization["groups_created"] == 1
        assert all(isinstance(item, ParagraphNode) for item in items)

    def test_changing_group_starts_a_new_paragraph(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(
                1,
                _heading("Chương 1", depth=1),
                _block("p", _long("Đoạn một"), group=2),
                _block("p", _long("Đoạn hai"), group=3),
            ),
            page_number=1,
        )
        root = parser.finish()

        assert [child.attributes.text for child in root.children[0].children] == [
            _long("Đoạn một"),
            _long("Đoạn hai"),
        ]
        assert parser.normalization["groups_created"] == 2

    def test_heading_ends_the_open_group(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(
                1,
                _heading("Chương 1", depth=1),
                _block("p", _long("Trước"), group=2),
                _heading("Bài 1", depth=2),
                _block("p", _long("Sau"), group=2),
            ),
            page_number=1,
        )
        root = parser.finish()

        chapter = root.children[0]
        assert chapter.children[0].attributes.text == _long("Trước")
        assert [child.attributes.text for child in chapter.children[1].children] == [_long("Sau")]

    def test_table_is_never_absorbed_into_a_group(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(
                1,
                _heading("Chương 1", depth=1),
                _block("p", _long("Trước"), group=2),
                _block("table", "| a | b |\n| --- | --- |\n| 1 | 2 |", group=2),
                _block("p", _long("Sau"), group=2),
            ),
            page_number=1,
        )
        root = parser.finish()

        children = root.children[0].children
        assert [type(child).__name__ for child in children] == [
            "ParagraphNode",
            "TableNode",
            "ParagraphNode",
        ]
        assert parser.normalization["groups_created"] == 2

    def test_oversized_group_is_split_at_the_chunk_budget(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _block("p", "câu diễn giải. " * 100)), page_number=1)
        root = parser.finish()

        children = root.children
        assert len(children) == 2
        assert all(len(child.attributes.text) <= MAX_CHUNK_SIZE for child in children)
        assert parser.normalization["groups_split"] == 1

    def test_two_tiny_groups_are_absorbed_into_one(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(
                1,
                _heading("Chương 1", depth=1),
                _block("p", "Đèn cồn (3).", group=2),
                _block("p", "Đèn cồn (4).", group=3),
            ),
            page_number=1,
        )
        root = parser.finish()

        assert [child.attributes.text for child in root.children[0].children] == [
            "Đèn cồn (3).\nĐèn cồn (4)."
        ]
        assert parser.normalization["short_groups_merged"] == 1

    def test_marker_on_a_list_item_is_ignored(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, VisionBlock(kind="list", marker="a)", group=1, text="a) Chất khí")),
            page_number=1,
        )
        root = parser.finish()

        assert [child.attributes.text for child in root.children] == ["a) Chất khí"]
        assert _titles(root) == [(1, "Vật lí 12")]

    def test_callout_label_and_body_share_one_chunk(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        body = _long("Em đã học được")
        parser.add_page(_page(1, _heading("EM ĐÃ HỌC"), _block("p", body)), page_number=1)
        root = parser.finish()

        children = root.children
        assert [child.attributes.text for child in children] == [f"EM ĐÃ HỌC\n{body}"]
        assert parser.normalization["callouts_reclassified"] == 1

    def test_table_keeps_rows_and_raw_text(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, _block("table", "| Đại lượng | Đơn vị |\n| --- | --- |\n| v | m/s |")),
            page_number=1,
        )
        root = parser.finish()

        table = root.children[0]
        assert isinstance(table, TableNode)
        assert table.attributes.rows == [["Đại lượng", "Đơn vị"], ["v", "m/s"]]
        assert table.attributes.extracted_text.startswith("| Đại lượng |")

    def test_table_without_pipes_falls_back_to_one_column(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _block("table", "Không có dấu phân cách")), page_number=1)
        root = parser.finish()

        table = root.children[0]
        assert isinstance(table, TableNode)
        assert table.attributes.rows == [["Không có dấu phân cách"]]

    def test_textless_figure_is_dropped_and_counted(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(
                1,
                _heading("Chương 1", depth=1),
                _block("figure", ""),
                _block("p", "Đoạn văn"),
            ),
            page_number=1,
        )
        root = parser.finish()

        assert parser.normalization["textless_figures_dropped"] == 1
        assert parser.blocks_by_kind["figure"] == 0
        assert [child.attributes.text for child in root.children[0].children] == ["Đoạn văn"]

    @pytest.mark.parametrize("kind", ["p", "question", "callout", "figure", "equation"])
    def test_text_first_kinds_become_paragraphs(self, kind: str) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _block(kind, "Chữ nhìn thấy trên trang")), page_number=1)
        root = parser.finish()

        paragraph = root.children[0]
        assert isinstance(paragraph, ParagraphNode)
        assert paragraph.attributes.text == "Chữ nhìn thấy trên trang"
        assert paragraph.metadata.page == 1

    def test_content_lands_under_the_open_section(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(
            _page(1, _heading("Chương 1", depth=1), _block("p", "Đoạn trong chương")),
            page_number=1,
        )
        root = parser.finish()

        chapter = root.children[0]
        assert [child.attributes.text for child in chapter.children] == ["Đoạn trong chương"]


class TestCrossPageGroups:
    """A group can continue onto the next page only when the model says so."""

    def test_state_exposes_the_open_group_label(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _block("p", _long("Đang mở"), group=7)), page_number=1)

        assert parser.state["open_group"] == 7
        # The tail is the end of the open buffer, so the model can see what it
        # is continuing rather than just being told a number.
        assert parser.state["open_group_tail"] == _long("Đang mở")[-200:]

    def test_state_has_no_open_group_before_any_content(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")

        assert parser.state["open_group"] is None

    def test_continued_group_joins_across_pages(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _block("p", _long("Trang một"), group=2)), page_number=1)
        parser.add_page(
            _page(2, _block("p", _long("Trang hai"), group=2), continues_previous=True),
            page_number=2,
        )
        root = parser.finish()

        children = root.children
        assert [child.attributes.text for child in children] == [
            f"{_long('Trang một')}\n{_long('Trang hai')}"
        ]
        assert parser.normalization["groups_created"] == 1
        assert parser.normalization["cross_page_continues"] == 1
        assert parser.normalization["cross_page_joins"] == 1

    def test_claimed_continuation_with_a_fresh_label_does_not_join(self) -> None:
        # Restarting the numbering is the model's default; the flag alone must
        # not splice two unrelated ideas together.
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _block("p", _long("Trang một"), group=5)), page_number=1)
        parser.add_page(
            _page(2, _block("p", _long("Trang hai"), group=1), continues_previous=True),
            page_number=2,
        )
        root = parser.finish()

        children = root.children
        assert [child.attributes.text for child in children] == [
            _long("Trang một"),
            _long("Trang hai"),
        ]
        assert parser.normalization["cross_page_continues"] == 1
        assert parser.normalization["cross_page_joins"] == 0

    def test_heading_at_start_of_continued_page_cuts_the_group(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _block("p", _long("Trang một"), group=2)), page_number=1)
        parser.add_page(
            _page(
                2,
                _heading("Mục mới", marker="2", group=2),
                _block("p", _long("Trang hai"), group=2),
                continues_previous=True,
            ),
            page_number=2,
        )
        root = parser.finish()

        assert root.children[0].attributes.text == _long("Trang một")
        section = root.children[1]
        assert section.attributes.title == "Mục mới"
        assert [child.attributes.text for child in section.children] == [_long("Trang hai")]
        assert parser.normalization["cross_page_continues"] == 1
        assert parser.normalization["cross_page_joins"] == 0

    def test_group_is_cut_at_the_page_without_continuation(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(1, _block("p", _long("Trang một"), group=2)), page_number=1)
        parser.add_page(_page(2, _block("p", _long("Trang hai"), group=2)), page_number=2)
        root = parser.finish()

        children = root.children
        assert [child.attributes.text for child in children] == [
            _long("Trang một"),
            _long("Trang hai"),
        ]
        assert parser.normalization["cross_page_continues"] == 0
        assert parser.normalization["cross_page_joins"] == 0


class TestEmptyInput:
    """A document with no usable block still produces a valid root."""

    def test_root_without_pages(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        root = parser.finish()

        assert root.attributes.title == "Vật lí 12"
        assert root.children == []
        assert root.metadata.end_page == 0

    def test_root_metadata_uses_the_first_page(self) -> None:
        parser = VisionDocumentParser("Vật lí 12")
        parser.add_page(_page(4), page_number=4)
        root = parser.finish()

        assert root.metadata.start_page == 4
        assert root.metadata.end_page == 4
