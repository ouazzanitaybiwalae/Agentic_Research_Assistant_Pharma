"""Local, read-only tools exposed to the research agent."""

from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore

from app.tools.lookup_glossary import create_lookup_glossary_tool
from app.tools.search_corpus import create_search_corpus_tool


def create_research_tools(
    vector_store: InMemoryVectorStore,
    corpus_dir: Path,
    retrieval_k: int,
) -> list[BaseTool]:
    return [
        create_search_corpus_tool(vector_store, retrieval_k),
        create_lookup_glossary_tool(corpus_dir / "GLOSSARY.md"),
    ]
