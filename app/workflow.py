import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langchain.agents import create_agent

from app.critic import review_research
from app.planning import plan_question
from app.prompts import ANSWER_SYSTEM_PROMPT, RESEARCH_AGENT_PROMPT
from app.schemas import ResearchState

logger = logging.getLogger(__name__)

_RETRIEVAL_TOOL_NAMES = {"search_corpus", "lookup_glossary"}
_SOURCE_MARKER_PATTERN = re.compile(
    r"\[`?([A-Za-z0-9_.-]+\.md)`?(?=[,\]\s])",
    re.IGNORECASE,
)
_CITATION_GROUP_PATTERN = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_CITATION_FILENAME_PATTERN = re.compile(
    r"`?([A-Za-z0-9_.-]+\.md)`?(?=$|[\s,;.!?)}\]])",
    re.IGNORECASE,
)
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "The available corpus does not provide sufficient evidence to answer this question reliably. "
    "No definitive conclusion can be provided from the available sources.\n\n"
    "This information is for research purposes only and is not medical advice. "
    "Consult a qualified healthcare professional for personal medical decisions."
)
CITATION_VALIDATION_RESPONSE = (
    "The generated answer could not be verified because it was missing citations or cited source "
    "files that were not retrieved during this request. Please retry the question.\n\n"
    "This information is for research purposes only and is not medical advice. "
    "Consult a qualified healthcare professional for personal medical decisions."
)

_MAX_EVIDENCE_CHUNKS = 30

def _evidence_transcript(messages: list[BaseMessage]) -> str:
    chunks: dict[str, None] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            for chunk in m.content.split("\n\n---\n\n"):
                chunks.setdefault(chunk.strip(), None)
    return "\n\n---\n\n".join(list(chunks)[-_MAX_EVIDENCE_CHUNKS:])

def _retrieved_sources(messages: list[BaseMessage]) -> list[str]:
    sources: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name not in _RETRIEVAL_TOOL_NAMES:
            continue
        if not isinstance(message.content, str):
            continue
        for source in _SOURCE_MARKER_PATTERN.findall(message.content):
            sources.setdefault(source.casefold(), source)
    return list(sources.values())


def _has_valid_citations(answer: str, retrieved_sources: list[str]) -> bool:
    cited_sources = [
        source
        for group in _CITATION_GROUP_PATTERN.findall(answer)
        for source in _CITATION_FILENAME_PATTERN.findall(group)
    ]
    if not cited_sources:
        return False
    retrieved_source_names = {source.casefold() for source in retrieved_sources}
    return all(source.casefold() in retrieved_source_names for source in cited_sources)


def create_research_workflow(
    model: BaseChatModel,
    tools: list[BaseTool],
    max_research_rounds: int,
):
    agent = create_agent(model, tools, system_prompt=RESEARCH_AGENT_PROMPT)

    def plan_node(state: ResearchState) -> dict:
        logger.info("Research request started question=%r", state["question"])
        plan = plan_question(state["question"], model)
        return {"plan": plan.steps, "research_rounds": 0}

    def research_node(state: ResearchState) -> dict:
        messages = list(state.get("messages", []))
        if messages:
            concerns = state["review"].concerns if "review" in state else []
            instruction = "Search again to address these evidence gaps: " + "; ".join(concerns)
        else:
            instruction = (
                f"Question: {state['question']}\n"
                f"Research plan:\n" + "\n".join(f"- {step}" for step in state["plan"])
            )
        messages.append(HumanMessage(content=instruction))
        research_message_start = len(messages)
        result = agent.invoke({"messages": messages})
        research_round = state.get("research_rounds", 0) + 1
        round_sources = _retrieved_sources(result["messages"][research_message_start:])
        logger.info(
            "Research diagnostic question=%r round=%d retrieved_sources=%s",
            state["question"],
            research_round,
            round_sources,
        )
        retrieved_sources = {
            source.casefold(): source
            for source in state.get("retrieved_sources", [])
        }
        for source in _retrieved_sources(result["messages"]):
            retrieved_sources.setdefault(source.casefold(), source)
        return {
            "messages": result["messages"],
            "retrieved_sources": list(retrieved_sources.values()),
            "research_rounds": research_round,
        }

    def critic_node(state: ResearchState) -> dict:
        review = review_research(
            question=state["question"],
            plan=state["plan"],
            evidence_transcript=_evidence_transcript(state["messages"]),
            model=model,
        )
        logger.info(
            "Critic verdict question=%r research_rounds=%d verdict=%s",
            state["question"],
            state.get("research_rounds", 0),
            review.model_dump(),
        )
        return {"review": review}

    def answer_node(state: ResearchState) -> dict:
        transcript = _evidence_transcript(state["messages"])
        review = state["review"]
        response = model.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Question:\n{state['question']}\n\n"
                        f"Research plan:\n{state['plan']}\n\n"
                        f"Critic review:\n{review.model_dump()}\n\n"
                        f"Retrieved evidence and tool outputs:\n{transcript}"
                    )
                ),
            ]
        )
        answer = str(response.content).strip()
        if not answer:
            raise ValueError("The answer model returned an empty response.")
        disclaimer = (
            "This information is for research purposes only and is not medical advice. "
            "Consult a qualified healthcare professional for personal medical decisions."
        )

        normalized_answer = " ".join(answer.split())
        normalized_disclaimer = " ".join(disclaimer.split())

        if normalized_disclaimer not in normalized_answer:
            answer = f"{answer}\n\n{disclaimer}"
        if not _has_valid_citations(answer, state.get("retrieved_sources", [])):
            return {"answer": CITATION_VALIDATION_RESPONSE}
        return {"answer": answer}

    def insufficient_evidence_node(state: ResearchState) -> dict:
        return {"answer": INSUFFICIENT_EVIDENCE_RESPONSE}


    def route_after_review(state: ResearchState) -> str:
        review = state["review"]
        can_retry = state.get("research_rounds", 0) < max_research_rounds
        if review.supported:
            decision = "answer"
        elif review.needs_more_evidence and can_retry:
            decision = "research"
        elif state.get("retrieved_sources"):
            decision = "answer"
        else:
            decision = "insufficient_evidence"
        logger.info(
            "Research routing question=%r research_rounds=%d final_routing_decision=%s",
            state["question"], state.get("research_rounds", 0), decision,
        )
        return decision

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("review", critic_node)
    graph.add_node("answer", answer_node)
    graph.add_node("insufficient_evidence", insufficient_evidence_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "research": "research",
            "answer": "answer",
            "insufficient_evidence": "insufficient_evidence",
        },
    )
    graph.add_edge("answer", END)
    graph.add_edge("insufficient_evidence", END)
    return graph.compile()
