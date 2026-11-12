from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts import CRITIC_SYSTEM_PROMPT
from app.schemas import GroundingReview


def review_research(
    question: str,
    plan: list[str],
    evidence_transcript: str,
    model: BaseChatModel,
) -> GroundingReview:
    critic = model.with_structured_output(GroundingReview)
    return critic.invoke(
        [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\nResearch plan:\n{plan}\n\n"
                    f"Agent and local-tool results:\n{evidence_transcript}"
                )
            ),
        ]
    )
