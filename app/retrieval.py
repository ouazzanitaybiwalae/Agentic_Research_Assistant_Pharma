from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app.documents import load_corpus_documents


def build_vector_store(
    corpus_dir: Path,
    embedding_model: Embeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> InMemoryVectorStore:
    documents = load_corpus_documents(corpus_dir, chunk_size, chunk_overlap)
    return InMemoryVectorStore.from_documents(documents, embedding_model)
