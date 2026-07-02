# ClauseInsight
> Upload a contract. Understand every clause. Know every risk.

**Author:** Tejas T. P.  
**Track:** Foundations of Applied Machine Learning  

---

## 1. Demo
* **Live Deployment:** [Link to Streamlit Community Cloud - *Coming Soon*]
* **Video Walkthrough:** [Link to 3-min Loom Video - *Coming Soon*]

---

## 2. Problem Statement
People sign contracts they don't understand. A fresh hire signs an employment agreement without realising that Section 9.3 assigns all their side-project IP to the company. A freelancer agrees to an NDA with a 5-year non-compete buried on page 14.

The problem isn't carelessness — it's that legal language is deliberately dense, contracts are long, and most people don't know what questions to ask. ClauseInsight fixes this by doing two things: automatically scanning every clause the moment a contract is uploaded and flagging risky ones without the user asking, and letting the user ask plain-English questions and get answers cited to the exact clause and page number.

---

## 3. Folder Structure

```
ClauseInsight/ 
│
├── src/                             🔲 — Main source code for the application
│   ├── pipeline/                    🔲 — Data ingestion and processing pipeline
│   │   ├── __init__.py              🔲 — Marks pipeline as a Python package
│   │   ├── parser.py                🔲 — Extracts text and structure from raw contract PDFs
│   │   ├── chunker.py               🔲 — Splits extracted text into semantic clauses or chunks
│   │   └── embedder.py              🔲 — Generates vector embeddings for text chunks using LLMs
│   │
│   ├── retrieval/                   🔲 — Search and context generation logic
│   │   ├── __init__.py              🔲 — Marks retrieval as a Python package
│   │   ├── retriever.py             🔲 — Vector search + reranking
│   │   └── context_builder.py       🔲 — Assembles retrieved chunks into LLM prompt context
│   │
│   ├── risk/                        🔲 — Risk analysis and classification modules
│   │   ├── __init__.py              🔲 — Marks risk as a Python package
│   │   ├── scanner.py               🔲 — Clause-level risk classification
│   │   └── risk_labels.py           🔲 — Low/Medium/High definitions + clause type taxonomy
│   │
│   ├── ui/                          🔲 — User interface components
│   │   ├── __init__.py              🔲 — Marks ui as a Python package
│   │   ├── app.py                   🔲 — Streamlit entry point (st.set_page_config only)
│   │   └── pages/                   🔲 — Streamlit multipage application screens
│   │       ├── 1_upload.py          🔲 — PDF upload + ingestion progress
│   │       ├── 2_qa.py              🔲 — Q&A interface over contract
│   │       └── 3_scanner.py         🔲 — Risk scanner results dashboard
│   │
│   └── utils/                       🔲 — Helper functions and shared utilities
│       ├── __init__.py              🔲 — Marks utils as a Python package
│       ├── store.py                 🔲 — Shared ChromaDB + SQLite connection helpers
│       └── logger.py                🔲 — Centralised logging config
│
├── data/                            🔲 — Local storage for databases and metadata
│   ├── chroma/                      🔲 — Vector database storage (auto-created at runtime, gitignored)
│   └── metadata.db                  🔲 — SQLite database for document metadata (auto-created at runtime, gitignored)
│
├── docs/                            🔲 — Project documentation and architecture decisions
│   └── adr/                         🔲 — Architecture Decision Records
│       ├── ADR-001-chromadb.md   
│       ├── ADR-002-clause-chunking.md
│       └── ADR-003-llm-choice.md 
│
├── tests/                           🔲 — Unit and integration tests for the project
│   ├── pipeline/                    🔲 — Tests for the data processing pipeline
│   │   ├── test_parser.py           🔲 — Unit tests for PDF parsing and extraction logic
│   │   ├── test_chunker.py          🔲 — Unit tests for clause segmentation rules
│   │   └── test_embedder.py         🔲 — Unit tests for vector embedding generation
│   ├── retrieval/                   🔲 — Tests for search and context generation
│   │   └── test_retriever.py        🔲 — Unit tests for vector search and reranking accuracy
│   └── risk/                        🔲 — Tests for risk analysis modules
│       └── test_scanner.py          🔲 — Unit tests for risk classification rules
│
├── legal_contracts/                 🔲 — documents for testing and development
│   ├── oneNDA_v2_1.pdf
│   ├── Stripe_Services_Agreement_India.pdf
│   ├── MASTER SERVICES AGREEMENT.pdf
│   └── CUAD_v1/
│       └── full_contract_pdf
│
├── .env                             🔲 — Environment variables (GOOGLE_API_KEY — gitignored)
├── .gitignore  
├── pyproject.toml                   🔲 — Project metadata and dependencies (uv manages dependencies)
└── README.md  
```

---

## 4. Architecture

```text
[PDF upload]                          [User question]
      |                                      |
[PDF parser — PyMuPDF]                       |
      |                                      |
[Clause chunker — section headers]           |
      |                                      |
      +---------> [Embedding model] <--------+
                  (text-embedding-004)
                         |
                    [ChromaDB]
                    (vector store)
                    /          \
          [Risk scanner]    [Q&A engine]
          (LLM classify)    (retrieve + generate)
                |                  |
        [Risk dashboard]   [Answer + citations]
                \                  /
                 [Streamlit UI]
```

*(An architecture diagram can also be found in `docs/architecture_clauseinsight.pdf`)*

---

## 5. Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| PDF parsing | PyMuPDF (`fitz`) | Fast, reliable, returns page numbers for citations |
| Chunking | Custom Python (regex + fallback) | Legal contracts have natural section headers — split by them, not by character count |
| Embedding model | Google `text-embedding-004` | Free via Gemini API, 768-dim embeddings, strong semantic quality for legal text |
| Vector store | ChromaDB | Runs locally, zero config, perfect for a single-contract scope |
| LLM — risk scan | Gemini 2.0 Flash | Structured JSON output (`{risk_level, category, reason}`), free tier (1,500 req/day), fast |
| LLM — Q&A | Same model | Grounded generation with citation enforcement in prompt |
| Frontend | Streamlit | File upload + two-panel UI in pure Python, fast iteration |
| Deployment | Streamlit Community Cloud | Free, GitHub-connected, live URL |
| Testing | pytest | Chunker output shape test + risk scanner JSON format test |

---

## 6. Quickstart

### Prerequisites
* Python 3.11+
* Git
* A Gemini API Key — get one free at [aistudio.google.com](https://aistudio.google.com)

### Installation & Setup

**a. Clone the repository**
```bash
git clone https://github.com/LegendTejas/ClauseInsight.git
cd ClauseInsight
```

**b. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**c. Install dependencies**
```bash
pip install -r requirements.txt
```

**d. Environment Variables**  
Create a `.env` file in the root directory and add your API key:
```
GEMINI_API_KEY=your_api_key_here
```

### Run the Application
```bash
streamlit run app/main.py
```

### Run Tests
```bash
pytest tests/
```

---

## 7. Data Sources

This system does not rely on a pre-loaded external dataset. The data source is entirely user-provided via PDF uploads of standard legal contracts (e.g., NDAs, Employment Agreements, SaaS Vendor Terms).

---

## 8. Architecture Decision Records (ADRs)

Technical trade-offs and decisions made during development are documented here:

- ADR-001: PDF Parsing Tool Selection  
- ADR-002: Clause-Aware Chunking Strategy  
- ADR-003: Choice of Vector Store  

---

## 9. Mini-Extension: Compare Two Contracts

Standard RAG systems are reactive. To go beyond the minimum requirements, I implemented a "Compare Two Contracts" feature. A user can upload an original NDA and a revised version, and ask "What changed in Section 4?". The system retrieves from both documents, diffs the relevant clauses, and surfaces the changes. This demonstrates multi-document reasoning.

---

## 10. Known Limitations

- **Scanned Images**: PDFs without a text layer (scanned images) will return empty strings via PyMuPDF. Optical Character Recognition (OCR) is not currently implemented.

- **Formatting Variances**: Contracts entirely lacking standard section headers may default to a fallback paragraph-splitting chunker, slightly reducing citation accuracy.

- **Cold Starts**: The Streamlit Community Cloud deployment may experience a 15-30 second cold start if the app has been inactive.

- **Embedding Dimensions**: `text-embedding-004` produces 768-dimensional vectors. Do not mix embeddings from different models in the same ChromaDB collection, as dimension mismatches will cause errors.

- **Gemini Free Tier Rate Limits**: The free tier allows 15 requests per minute and 1,500 requests per day. For large contracts with many clauses, the risk scanner may need to batch requests with short delays to stay within limits.

---

## 11. License & Acknowledgements
- License: MIT  
- **Acknowledgements**: Developed as part of the 2nd Year B.Tech CSE-AIDE Summer Internship (2026).
