"""Factory functions for creating configurable test ParsedDocuments."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import DocumentElement, ElementType, ParsedDocument


def make_test_doc(
    num_paragraphs: int = 5,
    include_tables: bool = True,
    include_code: bool = True,
    include_images: bool = True,
    source_file: Path | None = None,
) -> ParsedDocument:
    """Create a realistic ParsedDocument with configurable elements.

    Args:
        num_paragraphs: Number of text sections (each gets a heading + text).
        include_tables: Whether to include a TABLE element.
        include_code: Whether to include a CODE element.
        include_images: Whether to include an IMAGE element.
        source_file: Optional custom source path (default: fixtures/test.pdf).

    Returns:
        A ParsedDocument suitable for chunking benchmarks and integration tests.
    """
    if source_file is None:
        source_file = Path("fixtures/test.pdf")

    elements: list[DocumentElement] = []

    # Document title heading
    elements.append(
        DocumentElement(
            type=ElementType.HEADING,
            content="Test Document Title",
            page=1,
            reading_order=0,
            heading_level=1,
        )
    )

    # Opening paragraph
    elements.append(
        DocumentElement(
            type=ElementType.TEXT,
            content="This is an introductory paragraph for the test document. It provides "
            "background context and sets up the sections that follow. The document is "
            "designed to mimic a real research paper or technical report.",
            page=1,
            reading_order=1,
        )
    )

    # Multiple sections with headings + text
    for i in range(1, num_paragraphs + 1):
        elements.append(
            DocumentElement(
                type=ElementType.HEADING,
                content=f"Section {i}: Research Findings",
                page=1 + (i // 3),
                reading_order=len(elements) + 1,
                heading_level=2,
            )
        )
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                content=(
                    f"This is the body text for Section {i}. It contains several sentences "
                    "describing the methodology and results. The proposed approach demonstrates "
                    "significant improvements over baseline methods in terms of accuracy and "
                    "computational efficiency. Further analysis reveals interesting patterns that "
                    "warrant additional investigation in future work. " * (1 + (i % 3))
                ),
                page=1 + (i // 3),
                reading_order=len(elements) + 1,
            )
        )

    # Optional table
    if include_tables:
        elements.append(
            DocumentElement(
                type=ElementType.HEADING,
                content="Experimental Results",
                page=2,
                reading_order=len(elements) + 1,
                heading_level=2,
            )
        )
        elements.append(
            DocumentElement(
                type=ElementType.TABLE,
                content=[
                    ["Method", "Accuracy (%)", "Latency (ms)", "Memory (MB)"],
                    ["Baseline", "85.2", "12.3", "256"],
                    ["Proposed", "93.7", "8.1", "192"],
                    ["SOTA-2025", "91.4", "10.5", "210"],
                ],
                page=2,
                reading_order=len(elements) + 1,
                metadata={"caption": "Comparison of methods on the benchmark dataset."},
            )
        )

    # Optional code block
    if include_code:
        elements.append(
            DocumentElement(
                type=ElementType.HEADING,
                content="Implementation Details",
                page=3,
                reading_order=len(elements) + 1,
                heading_level=2,
            )
        )
        elements.append(
            DocumentElement(
                type=ElementType.CODE,
                content=(
                    "def compute_metric(predictions, targets):\n"
                    '    """Calculate the primary evaluation metric."""\n'
                    "    assert len(predictions) == len(targets)\n"
                    "    correct = sum(1 for p, t in zip(predictions, targets) if p == t)\n"
                    "    return correct / len(predictions)\n"
                ),
                page=3,
                reading_order=len(elements) + 1,
            )
        )

    # Optional image element
    if include_images:
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                content="The figure below illustrates the architecture of our proposed system.",
                page=3,
                reading_order=len(elements) + 1,
            )
        )
        elements.append(
            DocumentElement(
                type=ElementType.IMAGE,
                content=b"\x89PNG\r\n\x1a\n" + b"mock_png_data_payload" * 50,
                page=3,
                reading_order=len(elements) + 1,
                metadata={"caption": "System architecture diagram.", "format": "png"},
            )
        )

    # Concluding section
    elements.append(
        DocumentElement(
            type=ElementType.HEADING,
            content="Conclusion",
            page=4,
            reading_order=len(elements) + 1,
            heading_level=1,
        )
    )
    elements.append(
        DocumentElement(
            type=ElementType.TEXT,
            content=(
                "This test document demonstrates the structure-preserving capabilities of "
                "the KNOX pipeline. It includes headings, body text, a table, a code block, "
                "and an image element to verify that all element types are handled correctly "
                "during chunking and downstream processing."
            ),
            page=4,
            reading_order=len(elements) + 1,
        )
    )

    return ParsedDocument(
        source_file=source_file,
        elements=elements,
        title="Test Document Title",
        page_count=4,
        metadata={"generated_by": "tests.fixtures.documents.make_test_doc"},
    )
