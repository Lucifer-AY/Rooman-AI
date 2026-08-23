"""
Tests for Multi-Format Document Loaders in agent.py.
"""

import pytest
from pathlib import Path
from agent import (
    extract_text_from_pdf,
    extract_text_from_docx,
    UniversalDocumentRetriever
)


def test_markdown_document_loader(tmp_path):
    """Verifies that markdown files are split correctly by headers."""
    md_file = tmp_path / "sample.md"
    md_file.write_text(
        "# Test Document\n\n## Section One\nContent for section one.\n\n## Section Two\nContent for section two.",
        encoding="utf-8"
    )

    retriever = UniversalDocumentRetriever(data_path=tmp_path)
    assert len(retriever.passages) == 2
    assert retriever.passages[0]["section"] == "Section One"
    assert "Content for section one." in retriever.passages[0]["content"]
    assert retriever.passages[1]["section"] == "Section Two"
    assert "Content for section two." in retriever.passages[1]["content"]


def test_empty_folder_handling(tmp_path):
    """Ensures empty data folders do not cause crashes."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    retriever = UniversalDocumentRetriever(data_path=empty_dir)
    assert len(retriever.passages) == 0
    assert retriever.search("anything") == []
