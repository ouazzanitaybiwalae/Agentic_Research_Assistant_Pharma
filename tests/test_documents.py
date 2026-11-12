from pathlib import Path

import pytest

from app.documents import load_corpus_documents


def test_load_corpus_documents_preserves_source_and_section_metadata(tmp_path: Path) -> None:
    (tmp_path / "dailymed-example.md").write_text(
        "# Example\n\nSource: https://example.test/label\n\n"
        "## 7 Drug Interactions\n\nThe interaction evidence goes here.\n",
        encoding="utf-8",
    )

    documents = load_corpus_documents(tmp_path, chunk_size=800, chunk_overlap=120)

    section = next(document for document in documents if document.metadata["section_number"])

    assert "The interaction evidence goes here." in section.page_content
    assert section.metadata["source_file"] == "dailymed-example.md"
    assert section.metadata["source_type"] == "dailymed"
    assert section.metadata["drug"] == "example"
    assert section.metadata["source_url"] == "https://example.test/label"
    assert section.metadata["section_number"] == "7"
    assert section.metadata["heading_path"] == "Example > 7 Drug Interactions"


def test_empty_corpus_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No Markdown files found"):
        load_corpus_documents(tmp_path, chunk_size=800, chunk_overlap=120)
