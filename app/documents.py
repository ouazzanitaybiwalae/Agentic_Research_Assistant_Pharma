import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
    strip_headers=False,
)


def _document_metadata(path: Path, text: str) -> dict[str, str]:
    stem = path.stem
    if stem == "GLOSSARY":
        source_type, drug = "glossary", "all"
    else:
        source_type, _, drug = stem.partition("-")

    source_match = re.search(r"^Source:\s*(\S+)", text, re.MULTILINE)
    return {
        "source_file": path.name,
        "source_type": source_type,
        "drug": drug,
        "source_url": source_match.group(1) if source_match else "",
    }


def load_corpus_documents(
    corpus_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown files found in corpus directory: {corpus_dir}")

    documents: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        base_metadata = _document_metadata(path, text)
        sections = _HEADER_SPLITTER.split_text(text)

        for section in sections:
            headings = [
                section.metadata[key]
                for key in ("h1", "h2", "h3", "h4")
                if section.metadata.get(key)
            ]
            section_number = ""
            for heading in reversed(headings):
                match = re.match(r"^(\d+(?:\.\d+)*)\b", heading)
                if match:
                    section_number = match.group(1)
                    break

            documents.append(
                Document(
                    page_content=section.page_content,
                    metadata={
                        **base_metadata,
                        **section.metadata,
                        "heading_path": " > ".join(headings),
                        "section_number": section_number,
                    },
                )
            )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
