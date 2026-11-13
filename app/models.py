from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from app.config import get_settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        timeout=30, 
        max_retries=2
    )


class CachedQueryEmbeddings(Embeddings):
    """Caches embed_query results; corpus embedding (embed_documents) is untouched."""
    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner
        self._embed_query = lru_cache(maxsize=512)(inner.embed_query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_query(text.strip().casefold())


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    settings = get_settings()
    return CachedQueryEmbeddings(
        OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=30,
            max_retries=3,
            check_embedding_ctx_length=False,
        )
    )