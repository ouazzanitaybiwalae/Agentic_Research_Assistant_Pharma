from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app.tools.lookup_glossary import create_lookup_glossary_tool
from app.tools.search_corpus import create_search_corpus_tool


class StaticEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_search_tool_returns_file_and_section_citation() -> None:
    store = InMemoryVectorStore.from_documents(
        [
            Document(
                page_content="Warfarin has an interaction warning.",
                metadata={
                    "source_file": "dailymed-warfarin.md",
                    "heading_path": "7 Drug Interactions",
                    "section_number": "7",
                    "drug": "warfarin",
                    "source_type": "dailymed",
                },
            ),
            Document(
                page_content="Aspirin evidence.",
                metadata={
                    "source_file": "dailymed-aspirin.md",
                    "heading_path": "7 Drug Interactions",
                    "section_number": "7",
                    "drug": "aspirin",
                    "source_type": "dailymed",
                },
            ),
        ],
        StaticEmbeddings(),
    )
    search = create_search_corpus_tool(store, retrieval_k=3)

    result = search.invoke({"query": "interactions", "drug": "warfarin"})

    assert "[dailymed-warfarin.md, 7 Drug Interactions | 7]" in result
    assert "Warfarin has an interaction warning." in result
    assert "Aspirin evidence." not in result


def test_glossary_tool_finds_wrapped_entry(tmp_path: Path) -> None:
    path = tmp_path / "GLOSSARY.md"
    path.write_text("- **INR**: A blood test\n  monitoring warfarin's effect.\n", encoding="utf-8")
    lookup = create_lookup_glossary_tool(path)

    result = lookup.invoke({"term": "INR"})

    assert "[GLOSSARY.md]" in result
    assert "monitoring warfarin's effect." in result
