# AI Assistance Disclosure

This document records the AI tools used during the development of this project, the tasks they assisted with, and how their outputs were verified. I remain responsible for the final implementation and can explain the project's architecture, design decisions, and code.

## Tools used

| Tool | Used for |
|---|---|
| ChatGPT | Discussing architecture, debugging, code review, understanding errors, improving prompts, test design, and reviewing technical decisions. |
| GitHub Copilot (via VS Code) | Code suggestions, implementing targeted changes, generating and updating tests, debugging, and frontend improvements. |

**AI coding configuration files:** No additional AI coding configuration files were used, apart from this disclosure document. 

## What AI helped with

- **Debugging and code review:** Identifying issues in the research workflow, citation validation, evidence handling, and frontend behaviour.
- **Research workflow:** Reviewing the planner, research agent, critic, and answer-generation stages, and suggesting improvements to evidence handling and workflow routing.
- **Retrieval and citations:** Investigating retrieval failures, citation-format inconsistencies, and validation of cited source files against documents retrieved during a request.
- **Performance:** Investigating slow requests, analysing server logs, and suggesting improvements to model calls, retries, and retrieval efficiency.
- **Testing:** Writing and updating unit tests, reviewing regression cases, and checking API behaviour.
- **Frontend:** Improving the minimal HTML/CSS/JavaScript interface, Markdown rendering, error handling, and medical disclaimer presentation.
- **Docker and documentation:** Reviewing container configuration, dependency management, and drafting project documentation.

## How AI output was verified

- The four public evaluation questions were tested against the local pharmacology corpus.
- Generated answers and their citations were compared with the corresponding source documents.
- Unit tests were run to verify workflow behaviour, citation validation, and API functionality.
- Frontend behaviour was checked through the browser and API responses.
- Docker configuration and dependency changes were reviewed and tested where the local environment permitted.

## What was my own work

I was responsible for defining the project requirements and guiding its development according to the assignment specifications.

My contributions included:
- Understanding the problem and deciding how the research assistant should behave.
- Working through the overall architecture: question planning, retrieval, tool use, evidence review, and answer generation.
- Defining the purpose of the local corpus and the responsibilities of the research tools.
- Reviewing and adjusting prompts and workflow decisions based on the behaviour observed during testing.
- Running the application, testing the evaluation questions, inspecting server logs, and identifying discrepancies between generated answers and source documents.
- Evaluating AI-generated suggestions, deciding which changes were appropriate, and checking their effects through testing.
- Reviewing the final implementation to understand how its components interact and how evidence is used to produce answers.

## Limits of AI assistance

- AI-generated suggestions were not applied blindly; they were reviewed, modified, or rejected when they did not match the project's requirements or observed behaviour.
- AI tools assisted with implementation and debugging, but verification against the actual corpus and application behaviour was performed during project testing.
- I remain accountable for the final code, its limitations, and the explanations provided during the project presentation.