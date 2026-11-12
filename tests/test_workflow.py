from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, BaseMessage

import app.workflow as workflow_module
from app.schemas import GroundingReview, ResearchPlan
from app.workflow import INSUFFICIENT_EVIDENCE_RESPONSE, create_research_workflow


class FakeModel:
    def __init__(self) -> None:
        self.answer_calls = 0

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.answer_calls += 1
        return AIMessage(content="A grounded answer.")


class FakeResearchAgent:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, state: dict) -> dict:
        self.calls += 1
        return {"messages": state["messages"]}


def _review(supported: bool, needs_more_evidence: bool) -> GroundingReview:
    return GroundingReview(
        supported=supported,
        needs_more_evidence=needs_more_evidence,
        concerns=["Evidence is insufficient"] if not supported else [],
    )


def _run_workflow(
    monkeypatch: pytest.MonkeyPatch,
    reviews: list[GroundingReview],
    max_research_rounds: int,
) -> tuple[dict, FakeModel, FakeResearchAgent]:
    model = FakeModel()
    agent = FakeResearchAgent()
    review_results = iter(reviews)

    monkeypatch.setattr(
        workflow_module,
        "create_react_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        workflow_module,
        "plan_question",
        lambda question, model: ResearchPlan(steps=["Search the corpus"]),
    )
    monkeypatch.setattr(
        workflow_module,
        "review_research",
        lambda **kwargs: next(review_results),
    )

    workflow = create_research_workflow(model, [], max_research_rounds)
    result = workflow.invoke({"question": "What does the corpus support?"})
    return result, model, agent


def test_supported_review_routes_to_answer_even_if_more_evidence_is_suggested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, model, agent = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=True)],
        max_research_rounds=2,
    )

    assert result["answer"] == "A grounded answer.\n\n" + (
        "This information is for research purposes only and is not medical advice. "
        "Consult a qualified healthcare professional for personal medical decisions."
    )
    assert agent.calls == 1
    assert model.answer_calls == 1


def test_insufficient_review_retries_then_answers_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, model, agent = _run_workflow(
        monkeypatch,
        [
            _review(supported=False, needs_more_evidence=True),
            _review(supported=True, needs_more_evidence=False),
        ],
        max_research_rounds=2,
    )

    assert result["answer"].startswith("A grounded answer.")
    assert agent.calls == 2
    assert model.answer_calls == 1


def test_insufficient_review_at_total_attempt_limit_skips_answer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, model, agent = _run_workflow(
        monkeypatch,
        [
            _review(supported=False, needs_more_evidence=True),
            _review(supported=False, needs_more_evidence=True),
        ],
        max_research_rounds=2,
    )

    assert result["answer"] == INSUFFICIENT_EVIDENCE_RESPONSE
    assert agent.calls == 2
    assert model.answer_calls == 0


def test_insufficient_review_without_retrievable_evidence_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, model, agent = _run_workflow(
        monkeypatch,
        [_review(supported=False, needs_more_evidence=False)],
        max_research_rounds=2,
    )

    assert result["answer"] == INSUFFICIENT_EVIDENCE_RESPONSE
    assert agent.calls == 1
    assert model.answer_calls == 0


@pytest.mark.parametrize("needs_more_evidence", [False, True])
def test_insufficient_review_at_one_attempt_limit_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    needs_more_evidence: bool,
) -> None:
    result, model, agent = _run_workflow(
        monkeypatch,
        [_review(supported=False, needs_more_evidence=needs_more_evidence)],
        max_research_rounds=1,
    )

    assert result["answer"] == INSUFFICIENT_EVIDENCE_RESPONSE
    assert agent.calls == 1
    assert model.answer_calls == 0
