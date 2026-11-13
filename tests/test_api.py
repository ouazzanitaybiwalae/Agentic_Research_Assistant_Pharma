from collections import Counter

import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch):
    calls: Counter[str] = Counter()
    received_questions: list[str] = []

    class FakeWorkflow:
        def invoke(self, input_data: dict[str, str]) -> dict[str, str]:
            calls["invoke"] += 1
            received_questions.append(input_data["question"])
            return {"answer": "A test answer."}

    class FakeSettings:
        corpus_dir = "test-corpus"
        chunk_size = 800
        chunk_overlap = 120
        retrieval_k = 5
        max_research_rounds = 2

    monkeypatch.setattr(main, "get_settings", lambda: calls.update(["settings"]) or FakeSettings())
    monkeypatch.setattr(
        main,
        "get_embedding_model",
        lambda: calls.update(["embedding"]) or object(),
    )
    monkeypatch.setattr(
        main,
        "build_vector_store",
        lambda *args: calls.update(["vector_store"]) or object(),
    )
    monkeypatch.setattr(
        main,
        "create_research_tools",
        lambda *args: calls.update(["tools"]) or [],
    )
    monkeypatch.setattr(
        main,
        "get_chat_model",
        lambda: calls.update(["chat_model"]) or object(),
    )
    monkeypatch.setattr(
        main,
        "create_research_workflow",
        lambda *args: calls.update(["workflow"]) or FakeWorkflow(),
    )

    with TestClient(main.app) as client:
        yield client, calls, received_questions


def test_health_returns_healthy_status(api_client) -> None:
    client, _, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_serves_frontend_html(api_client) -> None:
    client, _, _ = api_client

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<textarea" in response.text
    assert 'id="question"' in response.text
    assert 'src="/static/marked.umd.js"' in response.text
    assert 'src="/static/purify.min.js"' in response.text


@pytest.mark.parametrize("asset_name", ["marked.umd.js", "purify.min.js"])
def test_markdown_assets_are_served_locally(api_client, asset_name: str) -> None:
    client, _, _ = api_client

    response = client.get(f"/static/{asset_name}")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.content


def test_ask_returns_workflow_answer(api_client) -> None:
    client, calls, received_questions = api_client

    response = client.post("/ask", json={"question": "What are warfarin contraindications?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "A test answer."}
    assert calls["invoke"] == 1
    assert received_questions == ["What are warfarin contraindications?"]


@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_ask_rejects_empty_or_whitespace_question(api_client, question: str) -> None:
    client, calls, _ = api_client

    response = client.post("/ask", json={"question": question})

    assert response.status_code == 422
    assert calls["invoke"] == 0


def test_application_dependencies_are_initialized_once_for_multiple_questions(api_client) -> None:
    client, calls, _ = api_client

    first = client.post("/ask", json={"question": "Question one"})
    second = client.post("/ask", json={"question": "Question two"})

    assert first.status_code == second.status_code == 200
    assert calls["settings"] == 1
    assert calls["embedding"] == 1
    assert calls["vector_store"] == 1
    assert calls["tools"] == 1
    assert calls["chat_model"] == 1
    assert calls["workflow"] == 1
    assert calls["invoke"] == 2
