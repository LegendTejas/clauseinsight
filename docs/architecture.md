## ClauseInsight — System Architecture

```mermaid
%%{init: {"theme": "base", "themeVariables": { "background": "#ffffff", "primaryColor": "#ffffff", "primaryTextColor": "#000000", "lineColor": "#000000", "fontFamily": "Inter, sans-serif"}}}%%
flowchart TD
    USER(["👤 User"])

    subgraph UI["🖥️ Streamlit Multipage App"]
        APP["app.py<br/>Entry point · page config<br/>logging setup · .env load"]
        PAGE_UP["pages/1_upload.py<br/>Upload and Ingest"]
        PAGE_QA["pages/2_qa.py<br/>Q and A Chat"]
        PAGE_SC["pages/3_scanner.py<br/>Risk Scanner Dashboard"]
    end

    subgraph INGEST["📥 Ingestion Pipeline"]
        PARSER["parser.py<br/>PyMuPDF<br/>Text + page numbers"]
        CHUNKER["chunker.py<br/>Format detector → strategy dispatch<br/>section_n · bare_n · onenda_table · fallback_prose"]
        EMBEDDER["embedder.py<br/>Google text-embedding-004<br/>768-dim vectors · idempotent upsert"]
    end

    subgraph STORE["🗄️ Data Layer — utils/store.py"]
        CHROMA[("ChromaDB<br/>clauseinsight_chunks<br/>cosine ANN index")]
        SQLITE[("SQLite<br/>data/metadata.db<br/>chunks + ingestions tables")]
        STORE_FN["store.py helpers<br/>get_chroma_collection · get_sqlite_connection<br/>list_ingested_contracts · delete_contract"]
    end

    subgraph RETRIEVAL_MOD["🔍 Retrieval Pipeline"]
        RETRIEVER["retriever.py<br/>Stage 1 — embed query<br/>Stage 2 — fetch 15 candidates from ChromaDB<br/>Stage 3 — MMR rerank λ=0.7<br/>Output: list[RetrievedChunk]"]
        CTX_BUILD["context_builder.py<br/>Deduplicate · token budget<br/>Format citations<br/>Output: BuiltContext"]
    end

    subgraph RISK_MOD["⚠️ Risk Module"]
        RISK_LABELS["risk_labels.py<br/>RiskLevel enum · ClauseCategory enum<br/>20 clause categories · UI display constants"]
        SCANNER["scanner.py<br/>Batched LLM calls · batch=5<br/>JSON output parsing · retry logic<br/>Output: ScanResult · list[RiskLabel]"]
    end

    subgraph EXT_API["🌐 External APIs — Google AI"]
        OAI_E["Google Generative AI<br/>text-embedding-004<br/>768-dim vectors"]
        OAI_L["Google Generative AI<br/>gemini-2.0-flash<br/>Q and A generation · Risk classification"]
    end

    subgraph UTILS["🛠️ Utilities"]
        LOGGER["utils/logger.py<br/>setup_logging · get_logger"]
    end

    subgraph OUTPUTS["📊 Streamlit Output Components"]
        RISK_DASH["Risk Dashboard<br/>🔴 HIGH  🟡 MEDIUM  🟢 LOW<br/>clause · category · reason · recommended action · JSON export"]
        QA_OUT["Answer Panel<br/>Chat history · grounded answer<br/>Section + page citation · multi-contract toggle"]
        UP_OUT["Upload Panel<br/>Ingestion progress · contract list<br/>Select / delete contract"]
    end

    subgraph DEPLOY["☁️ Deployment"]
        GITHUB["GitHub<br/>Public repo — main branch"]
        SCC["Streamlit Community Cloud<br/>streamlit run src/ui/app.py"]
    end

    USER --> APP
    APP --> PAGE_UP
    APP --> PAGE_QA
    APP --> PAGE_SC

    PAGE_UP -->|"PDF upload"| PARSER
    PARSER --> CHUNKER
    CHUNKER --> EMBEDDER
    EMBEDDER <-->|"embed req / 768-dim vectors"| OAI_E
    EMBEDDER -->|"vectors + metadata"| CHROMA
    EMBEDDER -->|"chunk rows"| SQLITE
    PAGE_UP -->|"list / delete contracts"| STORE_FN
    STORE_FN --- CHROMA
    STORE_FN --- SQLITE

    PAGE_QA -->|"user question + source filter"| RETRIEVER
    RETRIEVER <-->|"embed req / query vector"| OAI_E
    RETRIEVER <-->|"ANN query / top-15 chunks"| CHROMA
    RETRIEVER -->|"full text lookup"| SQLITE
    RETRIEVER -->|"list[RetrievedChunk]"| CTX_BUILD
    CTX_BUILD -->|"BuiltContext"| PAGE_QA
    PAGE_QA <-->|"generate req / cited answer"| OAI_L
    PAGE_QA --> QA_OUT

    PAGE_SC -->|"scan_contract"| SCANNER
    SCANNER -->|"get_all_chunks_for_contract"| SQLITE
    SCANNER <-->|"classify req / JSON result"| OAI_L
    SCANNER -->|"ScanResult + list[RiskLabel]"| PAGE_SC
    RISK_LABELS -->|"enums + UI constants"| PAGE_SC
    RISK_LABELS -->|"category and risk definitions"| SCANNER
    PAGE_SC --> RISK_DASH

    PAGE_UP --> UP_OUT

    PAGE_UP -->|"st.session_state[active_contract]"| PAGE_QA
    PAGE_UP -->|"st.session_state[active_contract]"| PAGE_SC

    LOGGER -.->|"setup_logging"| APP
    LOGGER -.->|"get_logger"| PAGE_UP
    LOGGER -.->|"get_logger"| PAGE_QA
    LOGGER -.->|"get_logger"| PAGE_SC

    RISK_DASH --> USER
    QA_OUT --> USER
    UP_OUT --> USER

    GITHUB -->|"push triggers deploy"| SCC

    classDef ui      fill:#e0e7ff,stroke:#3730a3,color:#000
    classDef ingest  fill:#ccfbf1,stroke:#0d9488,color:#000
    classDef store   fill:#dcfce7,stroke:#15803d,color:#000
    classDef risk    fill:#fee2e2,stroke:#b91c1c,color:#000
    classDef qa      fill:#ede9fe,stroke:#6d28d9,color:#000
    classDef comp    fill:#fef3c7,stroke:#b45309,color:#000
    classDef ext     fill:#e2e8f0,stroke:#475569,color:#000
    classDef out     fill:#d1fae5,stroke:#059669,color:#000
    classDef deploy  fill:#e5e7eb,stroke:#1e293b,color:#000

    class APP,PAGE_UP,PAGE_QA,PAGE_SC ui
    class PARSER,CHUNKER,EMBEDDER ingest
    class CHROMA,SQLITE,STORE_FN store
    class RISK_LABELS,SCANNER risk
    class RETRIEVER,CTX_BUILD qa
    class LOGGER comp
    class OAI_E,OAI_L ext
    class RISK_DASH,QA_OUT,UP_OUT out
    class GITHUB,SCC deploy
```

---

### Key Architecture Changes from Earlier Version

| Area | Old Design | Current Design |
|---|---|---|
| **UI structure** | Single `app.py` with 3 tabs | Multipage app: `app.py` + `pages/1_upload.py`, `2_qa.py`, `3_scanner.py` |
| **Compare tab** | Separate `comparator.py` with V1/V2 collections | **Removed** — single shared `clauseinsight_chunks` collection |
| **Embedding model** | OpenAI `text-embedding-3-small` (1536-dim) | Google `text-embedding-004` (768-dim) |
| **LLM** | OpenAI `gpt-4o-mini` | Google `gemini-2.0-flash` |
| **Vector store** | Two collections (`contract_v1`, `contract_v2`) | One collection with `source_name` metadata filter |
| **Data layer** | ChromaDB only via `vectorstore.py` | Dual store: ChromaDB (vectors) + SQLite (full text + metadata) via `utils/store.py` |
| **Retrieval** | Plain similarity search, top-3 | Two-stage MMR pipeline: 15 candidates → rerank → top-5 |
| **Context assembly** | Inline in Q&A engine | Dedicated `retrieval/context_builder.py` → `BuiltContext` |
| **Risk labels** | Inline in `risk_scanner.py` | Separate `risk/risk_labels.py` (enums, definitions, UI constants) |
| **Chunker** | Regex + 500-char fallback | Format detector → 4 strategies (section_n, bare_n, onenda_table, fallback_prose) |
| **Session state** | Per-tab state | Cross-page `active_contract` via `st.session_state` |
| **Logging** | Ad-hoc | Centralised `utils/logger.py` with `setup_logging` / `get_logger` |
