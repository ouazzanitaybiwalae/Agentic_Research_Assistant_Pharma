from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from app.config import get_settings
from app.models import get_chat_model, get_embedding_model
from app.retrieval import build_vector_store
from app.tools import create_research_tools
from app.workflow import create_research_workflow


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return question


class AskResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    embedding_model = get_embedding_model()
    vector_store = build_vector_store(
        settings.corpus_dir,
        embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    tools = create_research_tools(vector_store, settings.corpus_dir, settings.retrieval_k)
    chat_model = get_chat_model()
    app.state.workflow = create_research_workflow(
        chat_model,
        tools,
        settings.max_research_rounds,
    )
    yield


app = FastAPI(lifespan=lifespan)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR / "vendor"),
    name="static",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}

@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, http_request: Request) -> AskResponse:
    workflow = http_request.app.state.workflow
    result = workflow.invoke({"question": request.question})
    return AskResponse(answer=result["answer"])
