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

## 3. Architecture

```text
[PDF upload]                          [User question]
      |                                      |
[PDF parser — PyMuPDF]                       |
      |                                      |
[Clause chunker — section headers]           |
      |                                      |
      +---------> [Embedding model] <--------+
                  (text-embedding-3-small)
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

*(An architecture diagram can also be found in `docs/architecture.png)`*

---

## 5. Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| PDF parsing | PyMuPDF (`fitz`) | Fast, reliable, returns page numbers for citations |
| Chunking | Custom Python (regex + fallback) | Legal contracts have natural section headers — split by them, not by character count |
| Embedding model | OpenAI `text-embedding-3-small` | Cheap, fast, good enough for legal text; fallback: `all-MiniLM-L6-v2` (free, local) |
| Vector store | ChromaDB | Runs locally, zero config, perfect for a single-contract scope |
| LLM — risk scan | GPT-4o-mini or Claude Haiku | Structured JSON output (`{risk_level, category, reason}`), low cost per clause |
| LLM — Q&A | Same model | Grounded generation with citation enforcement in prompt |
| Frontend | Streamlit | File upload + two-panel UI in pure Python, fast iteration |
| Deployment | Streamlit Community Cloud | Free, GitHub-connected, live URL |
| Testing | pytest | Chunker output shape test + risk scanner JSON format test |

---

## 5. Quickstart

### Prerequisites
* Python 3.11+
* Git
* An OpenAI API Key

### Installation & Setup

a. **Clone the repository**
   ```bash
   git clone [https://github.com/LegendTejas/ClauseInsight.git](https://github.com/LegendTejas/ClauseInsight.git)
   cd ClauseInsight
   ```

b. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

c. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

d. **Environment Variables**
   nv file in the root directory and add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
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

## 6. Data Sources

This system does not rely on a pre-loaded external dataset. The data source is entirely user-provided via PDF uploads of standard legal contracts (e.g., NDAs, Employment Agreements, SaaS Vendor Terms).  

---

## 7. Architecture Decision Records (ADRs)

Technical trade-offs and decisions made during development are documented here:

ADR-001: PDF Parsing Tool Selection
ADR-002: Clause-Aware Chunking Strategy
ADR-003: Choice of Vector Store

---

## 8. Mini-Extension: Compare Two Contracts

Standard RAG systems are reactive. To go beyond the minimum requirements, I implemented a "Compare Two Contracts" feature. A user can upload an original NDA and a revised version, and ask "What changed in Section 4?". The system retrieves from both documents, diffs the relevant clauses, and surfaces the changes. This demonstrates multi-document reasoning.  

---

## 9. Known Limitations

- **Scanned Images**: PDFs without a text layer (scanned images) will return empty strings via PyMuPDF. Optical Character Recognition (OCR) is not currently implemented.  

- **Formatting Variances**: Contracts entirely lacking standard section headers may default to a fallback paragraph-splitting chunker, slightly reducing citation accuracy.[cite: 4]
  
- **Cold Starts**: The Streamlit Community Cloud deployment may experience a 15-30 second cold start if the app has been inactive.[cite: 4]

---

## 10. License & Acknowledgements
- License: MIT
- **Acknowledgements**: Developed as part of the 2nd Year B.Tech CSE-AIDE Summer Internship (2026).
