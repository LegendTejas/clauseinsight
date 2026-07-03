# ADR-001: Vector Store Selection — ChromaDB

**Status:** Accepted  
**Date:** July 2026  
**Author:** Tejas T. P.  
**Context:** ClauseInsight — Foundations of Applied Machine Learning Internship

---

## 1. Context

ClauseInsight needs a vector store to persist 768-dimensional embeddings
produced by `text-embedding-004` and support similarity search at query time.
The store must handle multiple contracts simultaneously (one collection, many
source_name namespaces) and run locally without a server process — the project
is a single-developer internship tool, not a production SaaS.

The retriever requires:
- Cosine similarity search over 768-dim vectors
- Metadata filtering (optional `source_name` filter for per-contract scoping)
- Return of stored embeddings alongside search results (required for MMR reranking)
- Persistent storage that survives process restarts
- Python SDK with no server dependency

---

## 2. Options Considered

### Option A: ChromaDB (chosen)
- Embedded Python library — no server, no Docker, no config
- Persistent via `PersistentClient` with a local directory
- Native cosine similarity via `hnsw:space=cosine`
- Returns stored embeddings in query results (`include=["embeddings"]`)
- `get_or_create_collection` is idempotent — safe to call on every startup
- Free, open source, actively maintained

### Option B: Pinecone
- Managed cloud vector database — strong production choice
- Requires API key, account signup, and network connectivity
- Free tier has index limits and dimension restrictions
- Adds external dependency and network latency to every query
- Overkill for a single-developer local tool

### Option C: FAISS (Facebook AI Similarity Search)
- Extremely fast, battle-tested at scale
- No built-in persistence — requires manual serialisation/deserialisation
- No metadata storage — would require a separate store for clause_id, page numbers
- No Python-native filtering — per-contract scoping would require index-level separation
- Significantly more implementation work for no benefit at this scale

### Option D: pgvector (PostgreSQL extension)
- Solid choice for production systems already using PostgreSQL
- Requires PostgreSQL installation and server management
- No justification for a full RDBMS when SQLite already handles our metadata needs
- Mixing two database systems (PostgreSQL + SQLite) for a small project is overengineering

---

## 3. Decision

**ChromaDB** via `PersistentClient`.

Single collection (`clauseinsight_chunks`) with cosine similarity metric.
All contracts share one collection, disambiguated by the `source_name`
metadata field — no need for per-contract collections.

```python
client = chromadb.PersistentClient(
    path=str(chroma_dir),
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection(
    name="clauseinsight_chunks",
    metadata={"hnsw:space": "cosine"},
)
```

---

## 4. Consequences

**Positive:**
- Zero infrastructure — `uv add chromadb` and it works
- Idempotent collection creation — safe to call on every startup
- Embeddings returned in query results — enables MMR without extra API calls
- Cosine similarity is the right metric for text embeddings (length-normalised)
- `anonymized_telemetry=False` — no data leaves the machine

**Negative / Accepted Trade-offs:**
- Not suitable for production scale (millions of vectors) — acceptable for
  an internship tool processing tens of contracts
- ChromaDB's metadata field is flat (strings + ints only, no nested dicts,
  no full-text search) — this is why we added SQLite as a companion store
  for structured queries and full clause text storage
- Do not mix embeddings from different models in the same collection —
  dimension mismatches will cause silent errors. If the embedding model
  changes, the collection must be deleted and rebuilt.

---

## 5. Implementation Notes

- `data/chroma/` is gitignored — never commit the vector store
- Connection management lives in `src/utils/store.py` — single source of truth
- `delete_contract()` in `store.py` removes from both ChromaDB and SQLite atomically
- ChromaDB bulk deletes are batched in groups of 100 (implicit API limit)
- Re-ingesting the same contract is safe — idempotency check in `embedder.py`
  compares all chunk IDs against existing ChromaDB IDs before embedding
