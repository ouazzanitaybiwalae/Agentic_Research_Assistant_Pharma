# Agentic Research Assistant

A Python assistant for research questions over the local medication-information corpus. It uses
LangGraph for orchestration, LangChain's OpenAI integration for chat and embeddings, and an
in-memory vector index for retrieval.

## Project layout

```text
app/
  config.py             # Environment-backed settings and shared paths
  models.py             # Cached chat and embedding model factories
  documents.py          # Markdown loading, heading-aware splitting, metadata
  retrieval.py          # In-memory vector-store construction
  tools/                # Agent-selected, local read-only tools
  planning.py           # Question decomposition
  critic.py             # Evidence and clinical-ambiguity review
  prompts.py            # Shared system prompts
  schemas.py            # Structured model outputs and graph state
  workflow.py           # Plan → tool-using research → review → answer graph
  main.py               # CLI entry point
corpus/                 # Local Markdown sources
tests/                  # Unit tests for ingestion and tools
```

Each module owns one responsibility. Model setup and configuration are shared instead of recreated
inside tools or graph nodes; tool factories receive their dependencies, which keeps them testable.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_EMBEDDING_MODEL` in `.env`. For a LiteLLM
OpenAI-compatible endpoint, set `OPENAI_BASE_UR` to its base URL. Ensure the endpoint supports both
chat completions and embeddings.

## Run

```powershell
python -m app.main "Compare the corpus evidence about warfarin and ibuprofen interactions."
```

The workflow plans sub-questions, gives the model local search and glossary tools to choose from,
checks retrieved evidence with a critic, and then answers with source-file citations. Retrieval chunks
are split at Markdown headings before a recursive 800-token split with 120-token overlap. Metadata
retains drug, source type, source URL, heading path, section number, and source filename. The vector
index is rebuilt in memory each run; document embeddings are sent to the configured model endpoint.

## Tests

```powershell
python -m pytest
```