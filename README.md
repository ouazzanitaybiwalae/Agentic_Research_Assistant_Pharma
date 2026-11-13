# Agentic Pharma Research Assistant

A small agentic research assistant that answers compound questions about a local pharmacology
corpus (FDA DailyMed labels and MedlinePlus-style monographs). It plans, retrieves, uses tools,
reviews its own evidence, and returns a cited answer with a medical disclaimer.

> **This is a technical exercise, not medical advice.**

## Quick start (Docker)

```bash
cp .env.example .env        # then fill in your values
docker build -t agent .
docker run -p 8000:8000 --env-file .env agent
```

Open <http://localhost:8000> for the web UI, or call the API directly:

```bash
curl http://localhost:8000/health
# {"status":"healthy"}

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main contraindications for warfarin?"}'
```

Startup embeds the whole corpus once, so the first `/health` can take a few seconds.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes | LiteLLM / OpenAI-compatible key |
| `OPENAI_MODEL` | yes | Chat model name |
| `OPENAI_EMBEDDING_MODEL` | yes | Embedding model name |
| `OPENAI_BASE_UR` | for LiteLLM | Proxy base URL |

## Run locally (without Docker)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --port 8000
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check, returns 200 |
| `POST` | `/ask` | Body `{"question": "..."}`, returns `{"answer": "<markdown>"}` |
| `GET` | `/` | Minimal web UI that calls `/ask` and renders the answer |

An empty or whitespace-only question is rejected with a 422.

## How it works

```text
question
   |
   v
 PLAN ---> RESEARCH (tool-using agent) ---> REVIEW (critic) ---> ANSWER
                ^                               |
                |_____ needs more evidence _____|   (capped rounds)
                                                |
                              no evidence ---> "insufficient evidence"
```

Orchestration is a **LangGraph** `StateGraph`; each stage is a node.

1. **Plan** (`planning.py`): the model decomposes the question into short, searchable
   sub-questions (structured output).
2. **Research** (`workflow.py`): a LangChain `create_agent` tool-calling agent follows the plan.
   *The model decides which tool to call, with which arguments, and how often* - nothing is hardcoded.
   - `search_corpus(query, drug?, source_type?)` - similarity search over the in-memory vector
     store, with optional metadata filters (drug name matching is tolerant, e.g. "Warfarin sodium").
   - `lookup_glossary(term)` - looks up terminology in `GLOSSARY.md`.
3. **Review** (`critic.py`): a critic model checks the retrieved evidence against the question and
   returns structured output: `supported`, `needs_more_evidence`, and a list of `concerns`
   (missing sections, conflicting wording, clinical ambiguity).
4. **Route** (`workflow.py`): if more evidence is needed and rounds remain, the agent searches
   again with the critic's concerns as instructions. Otherwise the answer is written, with the
   concerns passed along so ambiguity is stated rather than hidden.
5. **Answer** (`workflow.py`): the answer model writes a cited response using only the retrieved
   evidence. The code then **validates citations** (every cited `.md` file must actually have been
   retrieved in this request) and guarantees the disclaimer is present.

### Retrieval

- Markdown is split at headings first (keeping section numbers such as `4 CONTRAINDICATIONS`), then
  by a token-aware splitter (800 tokens, 120 overlap).
- Each chunk carries metadata: source file, drug, source type (`dailymed` / `medlineplus` /
  `glossary`), heading path, section number, source URL.
- Vectors live in LangChain's `InMemoryVectorStore`; the index is rebuilt at startup.
- Query embeddings are cached (`CachedQueryEmbeddings` in `models.py`) to reduce calls to the proxy.

### Design decisions

- **The critic flags, it does not veto.** An early version rejected any answer where the labels
  were merely worded differently (e.g. apixaban's "active pathological bleeding" vs warfarin's
  itemised bleeding conditions), which produced "insufficient evidence" for questions the corpus
  can answer. Now ambiguity becomes a caveat in the answer; only truly missing evidence triggers
  another search round or a refusal.
- **The critic and answer see retrieved evidence, not the chat transcript.** Evidence chunks are
  de-duplicated and the newest are kept (`_evidence_transcript`), so earlier results are no longer
  silently truncated away on retry rounds.
- **Bounded work.** Retry rounds are capped (`max_research_rounds`), and model calls have timeouts
  and limited retries, so a slow or rate-limited proxy cannot hang a request indefinitely.
- **Grounding is enforced in code, not just in the prompt.** Citations to files that were not
  retrieved cause the answer to be rejected.
- **Frontend disclaimer.** The backend's disclaimer text is rendered as a distinct note after
  Markdown rendering; it is restyled, never duplicated or rewritten.

## Project layout

```text
app/
  main.py          # FastAPI app: /health, /ask, static frontend
  config.py        # Environment-backed settings
  models.py        # Chat + embedding model factories (timeouts, query-embedding cache)
  documents.py     # Markdown loading, heading-aware splitting, metadata
  retrieval.py     # In-memory vector store construction
  tools/           # Agent-selected local tools (search_corpus, lookup_glossary)
  planning.py      # Question decomposition
  critic.py        # Grounding / clinical-ambiguity review
  prompts.py       # System prompts
  schemas.py       # Structured outputs and graph state
  workflow.py      # LangGraph: plan -> research -> review -> answer
corpus/            # Provided DailyMed / MedlinePlus markdown + GLOSSARY.md
frontend/          # Single-page UI (vendored marked + DOMPurify, no CDN needed)
tests/             # API, ingestion, tool and workflow tests
Dockerfile
requirements.txt   # Pinned dependencies
AI_USAGE.md        # AI assistance disclosure
```

## Tests

```bash
python -m pytest
```

The tests use fake models and a fake agent, so they need no API key. The ingestion test loads the
tokenizer, which is downloaded once and cached (the Docker image does this at build time).

## Evaluation questions

The four public questions from the assignment were run end to end and checked against the corpus:

1. Main contraindications for warfarin
2. Warfarin vs apixaban contraindications (shared and unique)
3. Warfarin + sertraline bleeding risk and monitoring
4. Metformin at eGFR 25 mL/min

## Known limitations

- Answers are only as good as the corpus; the assistant says so when evidence is missing rather
  than falling back on general knowledge.
- The critic and answer use the same model, so the review is not fully independent.
- The index is rebuilt in memory on every start (fine for ~20 files, not for a large corpus).
- LLM output varies between runs; the citation check and disclaimer guarantee are deterministic,
  the wording is not.
- `create_agent` (LangChain 1.x) is used for the tool loop; versions are pinned in `requirements.txt`.