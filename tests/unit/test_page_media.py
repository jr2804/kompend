"""Tests for the page-media helpers (figures and tables in a page body)."""

from __future__ import annotations

from kompend.core.text.page_media import find_page_figures, find_page_tables


class TestFindPageFigures:
    """The figures a page shows, with their captions."""

    def test_reads_caption_and_path(self) -> None:
        body = "Some prose.\n\n![Fig. 1. Overview](attachments/fig1.png)\n"
        assert find_page_figures(body) == [("Fig. 1. Overview", "attachments/fig1.png")]

    def test_a_figure_without_a_caption_has_an_empty_caption(self) -> None:
        body = "![](/abs/path/img.png)"
        assert find_page_figures(body) == [("", "/abs/path/img.png")]

    def test_a_link_title_is_not_part_of_the_path(self) -> None:
        body = '![caption](attachments/fig1.png "Figure 1")'
        assert find_page_figures(body) == [("caption", "attachments/fig1.png")]

    def test_document_order_is_kept(self) -> None:
        body = "![a](a.png)\ntext\n![b](b.png)\n![c](c.png)"
        assert [path for _, path in find_page_figures(body)] == ["a.png", "b.png", "c.png"]

    def test_the_same_image_twice_is_one_figure(self) -> None:
        body = "![a](a.png)\n![a again](a.png)"
        assert find_page_figures(body) == [("a", "a.png")]

    def test_an_image_in_a_code_block_is_not_a_figure(self) -> None:
        body = "```markdown\n![not a figure](code.png)\n```\n\n![real](real.png)"
        assert find_page_figures(body) == [("real", "real.png")]

    def test_a_plain_link_is_not_a_figure(self) -> None:
        assert find_page_figures("[link](page.md)") == []

    def test_an_empty_body_has_no_figures(self) -> None:
        assert find_page_figures("") == []


class TestFindPageTables:
    """The tables a page shows, as raw markdown."""

    def test_reads_the_whole_table(self) -> None:
        body = "intro\n\n| Name | Value |\n| --- | --- |\n| a | 1 |\n| b | 2 |\n\noutro"
        tables = find_page_tables(body)
        assert len(tables) == 1
        assert tables[0].startswith("| Name | Value |")
        assert "| b | 2 |" in tables[0]
        assert "outro" not in tables[0]

    def test_two_tables_are_two_blocks(self) -> None:
        body = "| a | 1 |\n| --- | --- |\n| b | 2 |\n\ntext\n\n| c | 3 |\n| --- | --- |\n| d | 4 |"
        assert len(find_page_tables(body)) == 2

    def test_a_table_without_a_delimiter_row_is_not_a_table(self) -> None:
        assert find_page_tables("| a | 1 |\n| b | 2 |") == []

    def test_pipe_lines_in_a_code_block_are_not_a_table(self) -> None:
        body = "```\n| a | 1 |\n| --- | --- |\n```\n"
        assert find_page_tables(body) == []

    def test_an_empty_body_has_no_tables(self) -> None:
        assert find_page_tables("") == []
