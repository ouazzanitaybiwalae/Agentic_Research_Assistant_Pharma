import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str | None
    chat_model: str
    embedding_model: str
    corpus_dir: Path
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_k: int = 8
    max_research_rounds: int = 2

def _clean(value: str | None) -> str | None:
    """Strip whitespace and surrounding quotes (docker --env-file keeps them)."""
    if value is None:
        return None
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value or None


def _clean_url(value: str | None) -> str | None:
    value = _clean(value)
    if value and not value.startswith(("http://", "https://")):
        value = "https://" + value
    return value

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    required = {
        "OPENAI_API_KEY": _clean(os.getenv("OPENAI_API_KEY")),
        "OPENAI_MODEL": _clean(os.getenv("OPENAI_MODEL")),
        "OPENAI_EMBEDDING_MODEL": _clean(os.getenv("OPENAI_EMBEDDING_MODEL")),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        api_key=required["OPENAI_API_KEY"] or "",
        base_url=_clean_url(os.getenv("OPENAI_BASE_UR")) or _clean_url(os.getenv("OPENAI_BASE_URL")) or None,
        chat_model=required["OPENAI_MODEL"] or "",
        embedding_model=required["OPENAI_EMBEDDING_MODEL"] or "",
        corpus_dir=PROJECT_ROOT / "corpus",
    )
