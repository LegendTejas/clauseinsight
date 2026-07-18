# ADR-001: Vector Store Selection — ChromaDB

**Status:** Accepted (updated)
**Date:** July 2026 · **Updated:** July 2026 (embedding model migration + collection-naming fix)
**Author:** Tejas T. P.
**Context:** ClauseInsight — Foundations of Applied Machine Learning Internship

---

## 0. Update Note

This ADR originally described 768-dimensional embeddings from Gemini's
`text-embedding-004`, and a single hardcoded collection name
(`clauseinsight_chunks`). Both are stale: the project migrated to
OpenAI (`text-embedding-3-small`, 1536-dim — see ADR-003), and the
collection-naming scheme was changed to derive the collection name
from the active embedding model, specifically to make embedding-model
changes safe by construction rather than something requiring a manual
folder wipe. Sections 1–4 below describe the **current** decision;
this update note exists so the reasoning behind the fix isn't lost.

---

## 1. Context

ClauseInsight needs a vector store to persist embeddings produced by
OpenAI's `text-embedding-3-small` (1536-dim) and support similarity
search at query time. The store must handle multiple contracts
simultaneously (one collection, many `source_name` namespaces) and run
locally without a server process — the project is a single-developer
internship tool, not a production SaaS.

The retriever requires:
- Cosine similarity search over high-dimensional vectors
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

Collection name is **derived from the active embedding model**, not
hardcoded — this is the piece that changed since the original version
of this ADR:

```python
def get_embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

def get_chroma_collection_name() -> str:
    model_name = get_embedding_model().replace("-", "_")
    return f"contracts_{model_name}"
```

```python
client = chromadb.PersistentClient(
    path=str(chroma_dir),
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection(
    name=get_chroma_collection_name(),   # e.g. "contracts_text_embedding_3_small"
    metadata={"hnsw:space": "cosine"},
)
```

All contracts embedded with the same model share one collection,
disambiguated by the `source_name` metadata field — no need for
per-contract collections. Contracts embedded with a *different* model
land in a *different*, automatically-named collection instead.

---

## 4. Consequences

**Positive:**
- Zero infrastructure — `pip install chromadb` and it works
- Idempotent collection creation — safe to call on every startup
- Embeddings returned in query results — enables MMR without extra API calls
- Cosine similarity is the right metric for text embeddings (length-normalised)
- `anonymized_telemetry=False` — no data leaves the machine
- **Embedding-model changes are safe by construction.** Because the
  collection name is derived from `EMBEDDING_MODEL`, switching models
  (e.g. testing `text-embedding-3-large` instead of `-small`) creates
  a new, separate collection rather than risking a dimension mismatch
  inside an existing one. This was a real problem during the Gemini →
  OpenAI migration (see ADR-003) — the collection had to be deleted
  and rebuilt by hand — and this naming scheme was added specifically
  so that never has to happen again.

**Negative / Accepted Trade-offs:**
- Not suitable for production scale (millions of vectors) — acceptable for
  an internship tool processing tens of contracts
- ChromaDB's metadata field is flat (strings + ints only, no nested dicts,
  no full-text search) — this is why we added SQLite as a companion store
  for structured queries and full clause text storage
- Switching embedding models does mean old collections are simply
  *orphaned*, not migrated — a contract embedded under a previous
  model won't be searchable until it's re-ingested under the current
  one. This is a deliberate trade-off (safety over automatic migration):
  silently re-embedding everything on every model change would be
  surprising and could rack up API costs without the developer noticing.

---

## 5. Implementation Notes

- `data/chroma/` is gitignored — never commit the vector store
- Connection management lives in `src/utils/store.py` — single source of truth
- `delete_contract()` in `store.py` removes from both ChromaDB and SQLite atomically
- ChromaDB bulk deletes are batched in groups of 100 (implicit API limit)
- Re-ingesting the same contract is safe — idempotency check in `embedder.py`
  compares all chunk IDs against existing ChromaDB IDs before embedding
- `sync_sqlite_to_chroma()` (in `embedder.py`) re-embeds SQLite's stored
  chunk text into the *current* model's collection — this is how a
  contract ingested under an old model becomes searchable again after
  a model switch, without needing to re-upload the original PDF
