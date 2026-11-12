from typing import TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    steps: list[str] = Field(description="Ordered, concise research steps for answering the question")


class GroundingReview(BaseModel):
    supported: bool = Field(
        description=(
            "Whether the available evidence supports a useful answer to the question, "
            "including its material claims without clinically significant ambiguity"
        )
    )
    needs_more_evidence: bool = Field(
        description="Whether a targeted search of the available corpus could plausibly address an important evidence gap"
    )
    concerns: list[str] = Field(
        description="Material unsupported claims, missing evidence, conflicting wording, or clinical ambiguity"
    )


class ResearchState(TypedDict, total=False):
    question: str
    plan: list[str]
    messages: list[BaseMessage]
    research_rounds: int
    review: GroundingReview
    answer: str
