from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore


def _format_result(document: Document) -> str:
    metadata = document.metadata
    citation = metadata.get("source_file", "unknown source")
    heading = metadata.get("heading_path")
    section = metadata.get("section_number")
    location = " | ".join(value for value in (heading, section) if value)
    if location:
        citation = f"{citation}, {location}"
    return f"[{citation}]\n{document.page_content}"


def create_search_corpus_tool(
    vector_store: InMemoryVectorStore,
    retrieval_k: int,
) -> BaseTool:
    @tool
    def search_corpus(
        query: str,
        drug: str | None = None,
        source_type: str | None = None,
    ) -> str:
        """Find relevant passages in the local drug corpus. Optionally filter by drug or source type."""
        filters = {
            key: value.casefold()
            for key, value in (("drug", drug), ("source_type", source_type))
            if value
        }

        def matches(document: Document) -> bool:
            metadata: dict[str, Any] = document.metadata
            return all(metadata.get(key, "").casefold() == value for key, value in filters.items())

        documents = vector_store.similarity_search(
            query,
            k=retrieval_k,
            filter=matches if filters else None,
        )
        if not documents:
            return "No matching passages were found in the local corpus."
        return "\n\n---\n\n".join(_format_result(document) for document in documents)

    return search_corpus
