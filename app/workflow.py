from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from app.critic import review_research
from app.planning import plan_question
from app.prompts import ANSWER_SYSTEM_PROMPT, RESEARCH_AGENT_PROMPT
from app.schemas import ResearchState

_MAX_TRANSCRIPT_CHARS = 30_000
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "The available corpus does not provide sufficient evidence to answer this question reliably. "
    "No definitive conclusion can be provided from the available sources.\n\n"
    "This information is for research purposes only and is not medical advice. "
    "Consult a qualified healthcare professional for personal medical decisions."
)


def _message_transcript(messages: list[BaseMessage]) -> str:
    entries = []
    for message in messages:
        content = message.content
        if not isinstance(content, str):
            content = str(content)
        entries.append(f"{message.type}: {content}")
    return "\n\n".join(entries)[-_MAX_TRANSCRIPT_CHARS:]


def create_research_workflow(
    model: BaseChatModel,
    tools: list[BaseTool],
    max_research_rounds: int,
):
    agent = create_react_agent(model, tools, prompt=RESEARCH_AGENT_PROMPT)

    def plan_node(state: ResearchState) -> dict:
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
        result = agent.invoke({"messages": messages})
        return {
            "messages": result["messages"],
            "research_rounds": state.get("research_rounds", 0) + 1,
        }

    def critic_node(state: ResearchState) -> dict:
        review = review_research(
            question=state["question"],
            plan=state["plan"],
            evidence_transcript=_message_transcript(state["messages"]),
            model=model,
        )
        return {"review": review}

    def answer_node(state: ResearchState) -> dict:
        transcript = _message_transcript(state["messages"])
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
        if disclaimer not in answer:
            answer = f"{answer}\n\n{disclaimer}"
        return {"answer": answer}

    def insufficient_evidence_node(state: ResearchState) -> dict:
        return {"answer": INSUFFICIENT_EVIDENCE_RESPONSE}

    def route_after_review(state: ResearchState) -> str:
        review = state["review"]
        can_retry = state.get("research_rounds", 0) < max_research_rounds
        if review.supported:
            return "answer"
        if review.needs_more_evidence and can_retry:
            return "research"
        return "insufficient_evidence"

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
