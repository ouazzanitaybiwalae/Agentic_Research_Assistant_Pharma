from collections.abc import Sequence

from fastapi import logger
import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

import app.workflow as workflow_module
from app.schemas import GroundingReview, ResearchPlan, ResearchState
from app.workflow import (
    CITATION_VALIDATION_RESPONSE,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    create_research_workflow,
)


class FakeModel:
    def __init__(self, answer: str = "A grounded answer [dailymed-warfarin.md, 4].") -> None:
        self.answer_calls = 0
        self.answer = answer

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.answer_calls += 1
        return AIMessage(content=self.answer)

    def bind_tools(self, *args, **kwargs):
        return self


class FakeResearchAgent:
    def __init__(self, tool_messages_by_round: list[list[BaseMessage]] | None = None) -> None:
        self.calls = 0
        self.tool_messages_by_round = tool_messages_by_round or [
            [
                ToolMessage(
                    content="[dailymed-warfarin.md, 4 CONTRAINDICATIONS]\nWarfarin evidence.",
                    tool_call_id="call-1",
                    name="search_corpus",
                )
            ]
        ]

    def invoke(self, state: dict) -> dict:
        messages = list(state["messages"])
        if self.calls < len(self.tool_messages_by_round):
            messages.extend(self.tool_messages_by_round[self.calls])
        self.calls += 1
        return {"messages": messages}


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
    answer: str = "A grounded answer [dailymed-warfarin.md, 4].",
    tool_messages_by_round: list[list[BaseMessage]] | None = None,
) -> tuple[dict, FakeModel, FakeResearchAgent]:
    model = FakeModel(answer)
    agent = FakeResearchAgent(tool_messages_by_round)
    review_results = iter(reviews)

    monkeypatch.setattr(
        workflow_module,
        "create_agent",
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

    assert result["answer"] == "A grounded answer [dailymed-warfarin.md, 4].\n\n" + (
        "This information is for research purposes only and is not medical advice. "
        "Consult a qualified healthcare professional for personal medical decisions."
    )
    assert agent.calls == 1
    assert model.answer_calls == 1


def test_existing_multiline_disclaimer_is_not_appended_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = (
        "The label lists contraindications [dailymed-warfarin.md, 4].\n\n"
        "This information is for research purposes only and is not medical advice.\n"
        "Consult a qualified healthcare professional for personal medical decisions."
    )
    result, _, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer=answer,
    )

    assert result["answer"] == answer
    disclaimer_start = "This information is for research purposes only and is not medical advice."
    assert result["answer"].count(disclaimer_start) == 1


def test_insufficient_review_at_total_attempt_limit_answers_with_caveats(monkeypatch):
    result, model, agent = _run_workflow(
        monkeypatch,
        [_review(False, True), _review(False, True)],
        max_research_rounds=2,
    )
    assert result["answer"].startswith("A grounded answer [dailymed-warfarin.md, 4].")
    assert agent.calls == 2
    assert model.answer_calls == 1


def test_insufficient_review_without_retrievable_evidence_fails_closed(monkeypatch):
    result, model, agent = _run_workflow(
        monkeypatch,
        [_review(supported=False, needs_more_evidence=False)],
        max_research_rounds=2,
        tool_messages_by_round=[[]],
    )
    assert result["answer"] == INSUFFICIENT_EVIDENCE_RESPONSE
    assert agent.calls == 1
    assert model.answer_calls == 0


@pytest.mark.parametrize("needs_more_evidence", [False, True])
def test_insufficient_review_at_one_attempt_limit_answers_with_caveats(
    monkeypatch: pytest.MonkeyPatch,
    needs_more_evidence: bool,
) -> None:
    result, model, agent = _run_workflow(
        monkeypatch,
        [_review(supported=False, needs_more_evidence=needs_more_evidence)],
        max_research_rounds=1,
    )

    assert result["answer"].startswith("A grounded answer [dailymed-warfarin.md, 4].")
    assert agent.calls == 1
    assert model.answer_calls == 1



def test_valid_retrieved_citations_are_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, model, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer="The label lists contraindications [dailymed-warfarin.md, section 4].",
        tool_messages_by_round=[
            [
                ToolMessage(
                    content="[dailymed-warfarin.md, 4 CONTRAINDICATIONS]\nEvidence.",
                    tool_call_id="call-search",
                    name="search_corpus",
                ),
                ToolMessage(
                    content="[GLOSSARY.md]\nMeaning.",
                    tool_call_id="call-glossary",
                    name="lookup_glossary",
                ),
            ]
        ],
    )

    assert result["answer"].startswith(
        "The label lists contraindications [dailymed-warfarin.md, section 4]."
    )
    assert model.answer_calls == 1


@pytest.mark.parametrize(
    "answer",
    [
        "Claim [dailymed-warfarin.md, section 4].",
        "Claim [`dailymed-warfarin.md` — section 4].",
        (
            "Claim *(dailymed-apixaban.md, section 4; "
            "dailymed-warfarin.md, section 4)*."
        ),
    ],
)
def test_supported_citation_group_formats_are_returned(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    result, _, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer=answer,
        tool_messages_by_round=[
            [
                ToolMessage(
                    content=(
                        "[dailymed-warfarin.md, section 4]\nWarfarin evidence.\n"
                        "[dailymed-apixaban.md, section 4]\nApixaban evidence."
                    ),
                    tool_call_id="call-search",
                    name="search_corpus",
                )
            ]
        ],
    )

    assert result["answer"].startswith(answer)


def test_citation_filename_comparison_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer="The label lists contraindications [DAILYMED-WARFARIN.MD, section 4].",
    )

    assert result["answer"].startswith(
        "The label lists contraindications [DAILYMED-WARFARIN.MD, section 4]."
    )


def test_unretrieved_citation_rejects_entire_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, model, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer="Claim [dailymed-apixaban.md, section 4].",
    )

    assert result["answer"] == CITATION_VALIDATION_RESPONSE
    assert "dailymed-apixaban.md" not in result["answer"]
    assert model.answer_calls == 1


def test_unretrieved_file_in_parenthetical_group_rejects_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer=(
            "Claim *(dailymed-warfarin.md, section 4; "
            "dailymed-apixaban.md, section 4)*."
        ),
    )

    assert result["answer"] == CITATION_VALIDATION_RESPONSE


def test_mixed_retrieved_and_unretrieved_citations_reject_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer="Claim [dailymed-warfarin.md, section 4] and [dailymed-apixaban.md, section 4].",
    )

    assert result["answer"] == CITATION_VALIDATION_RESPONSE


def test_answer_without_citations_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer="An uncited answer.",
    )

    assert result["answer"] == CITATION_VALIDATION_RESPONSE


def test_sources_are_accumulated_across_research_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, agent = _run_workflow(
        monkeypatch,
        [
            _review(supported=False, needs_more_evidence=True),
            _review(supported=True, needs_more_evidence=False),
        ],
        max_research_rounds=2,
        answer="Label [dailymed-warfarin.md, section 4]; glossary [GLOSSARY.md].",
        tool_messages_by_round=[
            [
                ToolMessage(
                    content="[dailymed-warfarin.md, 4 CONTRAINDICATIONS]\nEvidence.",
                    tool_call_id="call-search",
                    name="search_corpus",
                )
            ],
            [
                ToolMessage(
                    content="[GLOSSARY.md]\nMeaning.",
                    tool_call_id="call-glossary",
                    name="lookup_glossary",
                )
            ],
        ],
    )

    assert result["answer"].startswith("Label [dailymed-warfarin.md, section 4]")
    assert agent.calls == 2


def test_ai_and_unrelated_tool_messages_cannot_add_retrieved_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = _run_workflow(
        monkeypatch,
        [_review(supported=True, needs_more_evidence=False)],
        max_research_rounds=1,
        answer="Claim [dailymed-apixaban.md, section 4].",
        tool_messages_by_round=[
            [
                AIMessage(content="I found [dailymed-apixaban.md]."),
                ToolMessage(
                    content="[dailymed-apixaban.md, section 4]",
                    tool_call_id="call-other",
                    name="unrelated_tool",
                ),
            ]
        ],
    )

    assert result["answer"] == CITATION_VALIDATION_RESPONSE
