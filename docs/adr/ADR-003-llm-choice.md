# ADR-003: LLM and Embedding Model Selection

**Status:** Accepted (amended)
**Date:** July 2026 · **Amended:** July 2026 (Gemini → OpenAI migration)
**Author:** Tejas T. P.
**Context:** ClauseInsight — Foundations of Applied Machine Learning Internship

---

## 0. Amendment Note

This ADR originally chose Google Gemini (`text-embedding-004` +
`gemini-2.0-flash`) for both embeddings and LLM calls. The project has
since migrated to OpenAI (`text-embedding-3-small` + `gpt-4o-mini`).
Sections 1–4 below describe the **current** decision; the original
Gemini rationale is kept in Section 6 for historical context, since
the trade-offs that motivated switching are useful to remember.

**Practical consequence of the migration:** embedding dimensionality
changed from 768 → 1536. Old Gemini-based ChromaDB collections are
not compatible with the new embeddings — any existing local vector
store must be deleted and contracts re-ingested.

---

## 1. Context

ClauseInsight requires two distinct model types:

1. **Embedding model** — converts clause text and user queries into
   vectors for storage in ChromaDB and similarity search
2. **LLM** — generates natural language answers to user questions
   (Q&A engine) and classifies clauses by risk level (risk scanner)

Both models are called at runtime — the embedding model at ingestion and
query time, the LLM at Q&A and scan time.

---

## 2. Embedding Model Decision

### Current choice: OpenAI `text-embedding-3-small`

- 1536-dimensional output, strong semantic quality for legal text
- Single OpenAI account covers both embeddings and LLM calls — one
  API key, one billing relationship
- No `task_type` distinction like Gemini's asymmetric
  `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` — OpenAI's embeddings API
  treats indexing and query embedding identically. This is a real
  trade-off: Gemini's asymmetric training could meaningfully improve
  retrieval quality, and that nuance is lost in the OpenAI setup.

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)
```

**Critical constraint (unchanged):** Never mix embeddings from
different models in the same ChromaDB collection. Since the model
changed as part of this migration, the entire collection must be
deleted and rebuilt from scratch — this is a breaking change for
anyone with a locally-ingested Gemini-era database.

---

## 3. LLM Decision

### Current choice: OpenAI `gpt-4o-mini`

- Strong structured JSON output for risk scanner
  (`{risk_level, category, reason}` format)
- 128K token context window — comfortably handles clause-level context
  for this project's prompt sizes
- Same API key as the embedding model — single credential for the
  entire project
- Sufficient reasoning quality for clause-level risk classification
  and grounded Q&A

LLM temperature is set to `0.2` for both use cases:
- Low enough to be deterministic and factual for legal Q&A
- High enough to produce varied phrasing across different clauses
- `0.0` would make risk classifications overly rigid; `0.7+` would
  risk hallucinated legal conclusions

---

## 4. SDK

All LLM and embedding calls go through the official `openai` Python
SDK (`from openai import OpenAI`), via `client.chat.completions.create()`
for generation and `client.embeddings.create()` for embeddings.

Environment variable: `OPENAI_API_KEY`.

---

## 5. Consequences

**Positive:**
- Single API key covers all model calls — simpler `.env`, simpler auth
- `gpt-4o-mini` structured JSON output is reliable for the scanner
- Large context window removes any practical concern about prompt size
  for this project's contract lengths

**Negative / Accepted Trade-offs:**
- OpenAI usage is billed per request — unlike the free-tier Gemini setup
  this project started with, there is no free quota. Batch sizes
  (`EMBED_BATCH_SIZE`, scanner batch of 5 clauses) should stay in mind
  when scanning many large contracts back-to-back.
- Losing Gemini's asymmetric `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`
  task types is a minor retrieval-quality regression that hasn't been
  benchmarked yet — worth revisiting if retrieval quality feels off.
- Network dependency — no offline mode. Contracts cannot be processed
  without internet access.
- Migration breaks any existing local ChromaDB collection built on
  768-dim Gemini embeddings (see Section 0).

---

## 6. Historical Context: Original Gemini Decision (superseded)

The original decision (July 2026) chose Google Gemini for both
embeddings and LLM, motivated primarily by cost (free tier) and the
asymmetric embedding task types:

- **Embedding:** `text-embedding-004`, 768-dim, free via Gemini API,
  1,500 requests/minute free tier — sufficient for batch ingestion.
- **LLM:** `gemini-2.0-flash`, free tier of 15 requests/minute and
  1,500 requests/day, 1,048,576 token context window.
- **OpenAI was evaluated and rejected at the time** specifically for
  cost (no free tier, ~$0.02/1M tokens) and the overhead of managing
  two API keys — reasoning that no longer applies now that the whole
  project runs on a single OpenAI key.
- **`sentence-transformers` (local models)** and **local LLMs via
  Ollama** were also rejected — no GPU on the development machine made
  CPU inference too slow for interactive use, and smaller local models
  struggled with structured legal JSON output.

These trade-offs are worth keeping in mind if cost ever becomes a
constraint again — a return to a free-tier or local-inference setup
remains a valid option.

---

## 7. Future Considerations

- Benchmark retrieval quality with vs. without asymmetric query/document
  embeddings, now that OpenAI doesn't offer that distinction natively —
  confirm the Gemini-era assumption that it "meaningfully improves"
  retrieval actually mattered in practice.
- If cost becomes a constraint, evaluate `text-embedding-3-small`
  usage volume against a return to a free-tier provider or local
  embeddings for the ingestion side.
- Monitor OpenAI's model releases for cheaper or higher-quality
  small models that could replace `gpt-4o-mini`.
