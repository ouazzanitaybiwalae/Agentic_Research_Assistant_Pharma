from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts import PLANNER_SYSTEM_PROMPT
from app.schemas import ResearchPlan


def plan_question(question: str, model: BaseChatModel) -> ResearchPlan:
    planner = model.with_structured_output(ResearchPlan)
    return planner.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
    )
