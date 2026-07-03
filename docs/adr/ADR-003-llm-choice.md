# ADR-003: LLM and Embedding Model Selection

**Status:** Accepted  
**Date:** July 2026  
**Author:** Tejas T. P.  
**Context:** ClauseInsight — Foundations of Applied Machine Learning Internship

---

## 1. Context

ClauseInsight requires two distinct model types:

1. **Embedding model** — converts clause text and user queries into
   768-dimensional vectors for storage in ChromaDB and similarity search
2. **LLM** — generates natural language answers to user questions
   (Q&A engine) and classifies clauses by risk level (risk scanner)

Both models are called at runtime — the embedding model at ingestion and
query time, the LLM at Q&A and scan time. Cost, latency, and free tier
limits are primary constraints for an internship project.

---

## 2. Embedding Model Decision

### Options Considered

**Option A: Google `text-embedding-004` (chosen)**
- 768-dimensional output — well-suited for legal semantic similarity
- Free via Gemini API (same key as the LLM — no second account needed)
- `task_type` parameter allows separate optimisation for indexing
  (`RETRIEVAL_DOCUMENT`) vs. querying (`RETRIEVAL_QUERY`) — Google trains
  the model on (query, document) pairs, so asymmetric task types
  meaningfully improve retrieval quality
- Free tier: 1,500 embedding requests/minute — sufficient for batch
  ingestion of large contracts with inter-batch sleep

**Option B: OpenAI `text-embedding-3-small`**
- Strong performance, 1536-dim (can be truncated to 768)
- Requires separate OpenAI API key and billing setup
- $0.02 per 1M tokens — not free at any volume
- Two API keys (OpenAI + Gemini for LLM) adds credential management overhead
- **Rejected** — cost and two-key complexity not justified

**Option C: `sentence-transformers` (local models)**
- Runs fully offline — no API key, no rate limits
- `all-MiniLM-L6-v2`: 384-dim, fast but lower quality for legal text
- `legal-bert-base-uncased`: domain-specific but 768-dim, large download
- No GPU on development machine — CPU inference is slow for ingestion batches
- **Rejected** — latency unacceptable, no clear quality advantage over
  `text-embedding-004` for this use case

### Decision

**`text-embedding-004`** via `google-genai` SDK.

```python
response = client.models.embed_content(
    model="text-embedding-004",
    contents=texts,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",  # at index time
        output_dimensionality=768,
    ),
)
```

Query time uses `task_type="RETRIEVAL_QUERY"` — both task types are
centralised in `src/pipeline/embedder.py` so all Gemini API calls
live in one module.

**Critical constraint:** Never mix embeddings from different models in
the same ChromaDB collection. If the embedding model changes, the
entire collection must be deleted and rebuilt from scratch.

---

## 3. LLM Decision

### Options Considered

**Option A: `gemini-2.0-flash` (chosen)**
- Fast inference — sub-2s response for clause Q&A
- Free tier: 15 requests/minute, 1,500 requests/day
- Strong structured JSON output for risk scanner
  (`{risk_level, category, reason}` format)
- 1,048,576 token context window — can handle large clause context
- Same API key as `text-embedding-004` — single credential for the
  entire project
- Sufficient reasoning quality for clause-level risk classification
  and grounded Q&A

**Option B: `gemini-1.5-pro`**
- Better reasoning on complex multi-clause questions
- 2,097,152 token context window
- Slower and more expensive — lower free tier limits
- The quality difference is marginal for single-clause Q&A
- **Rejected** — `gemini-2.0-flash` is sufficient, lower cost/latency

**Option C: OpenAI `gpt-4o-mini`**
- Strong structured output, JSON mode
- Requires separate API key and billing
- Higher cost than free Gemini tier
- **Rejected** — cost and two-key complexity

**Option D: Local LLM via Ollama (`llama3`, `mistral`)**
- Fully offline, no API costs or rate limits
- No GPU on development machine — 7B parameter models run at
  ~2 tokens/second on CPU — unacceptably slow for interactive Q&A
- Smaller models struggle with legal reasoning and structured JSON output
- **Rejected** — hardware limitation

### Decision

**`gemini-2.0-flash`** for both Q&A generation and risk classification.

Using one model for both tasks simplifies the codebase — one client
instantiation, one set of rate limit considerations, one API key.

LLM temperature is set to `0.2` for both use cases:
- Low enough to be deterministic and factual for legal Q&A
- High enough to produce varied phrasing across different clauses
- `0.0` would make risk classifications overly rigid; `0.7+` would
  hallucinate legal conclusions

---

## 4. SDK Decision: `google-genai` over `google-generativeai`

The older `google-generativeai` package is deprecated as of mid-2025.
The new `google-genai` package (`from google import genai`) is the
actively maintained replacement with:
- Unified client for both embeddings and LLM calls
- `types.EmbedContentConfig` for task_type and dimensionality control
- Consistent error types for retry logic

All Gemini API calls use `from google import genai` — the deprecated
package is not installed.

Environment variable: `GOOGLE_API_KEY` (new SDK convention).
The older SDK used `GEMINI_API_KEY` — these are the same key, different
variable names. ClauseInsight uses `GOOGLE_API_KEY` throughout.

---

## 5. Consequences

**Positive:**
- Single API key covers all model calls — simpler `.env`, simpler auth
- Free tier is sufficient for an internship-scale project
- `task_type` asymmetry measurably improves retrieval quality
- `gemini-2.0-flash` structured JSON output is reliable for scanner

**Negative / Accepted Trade-offs:**
- Free tier rate limits require batched embedding with sleep between
  batches (`INTER_BATCH_SLEEP=0.5s`, `EMBED_BATCH_SIZE=20`)
- 1,500 requests/day LLM limit means the risk scanner must batch
  clause classification carefully — scanning a 161-chunk contract
  exhausts the daily limit in one run if not batched
- Network dependency — no offline mode. Scanned contracts with no
  internet access cannot be processed.
- `gemini-2.0-flash` occasionally produces slightly informal phrasing
  in risk reasons — acceptable for an internship tool, not for production

---

## 6. Future Considerations

- If the project moves to production, swap `gemini-2.0-flash` for
  `gemini-1.5-pro` or `gemini-2.0-pro` for better legal reasoning
- If offline use is required, evaluate `legal-bert` for embeddings
  and a quantised local LLM for risk classification
- Monitor Google's embedding model releases — `text-embedding-005`
  may offer higher dimensions or better legal domain performance
