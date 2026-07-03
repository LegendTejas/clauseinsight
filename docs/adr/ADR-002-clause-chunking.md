# ADR-002: Clause-Aware Chunking Strategy

**Status:** Accepted  
**Date:** July 2026  
**Author:** Tejas T. P.  
**Context:** ClauseInsight — Foundations of Applied Machine Learning Internship

---

## 1. Context

A RAG pipeline's retrieval quality is only as good as its chunks. For legal
contracts specifically, the chunking decision has outsized consequences:

- **Too coarse** (whole sections): embeddings average over too much content,
  retrieval returns entire sections when the user asked about one sub-clause
- **Too fine** (sentences): citations become meaningless ("page 8, sentence 3"),
  and sub-clause text loses the definitional context from its parent clause
- **Wrong boundaries** (fixed character count): splits mid-clause, destroying
  the semantic unit that legal text is structured around

The standard RAG approach — fixed-size sliding window with overlap — is
inappropriate for legal contracts because contracts have natural, meaningful
boundaries (section headers, sub-clause labels) that must be respected.

Additionally, real-world contracts are structurally heterogeneous. The same
chunking logic that works for a numbered agency agreement fails silently on
a table-based NDA template.

---

## 2. Options Considered

### Option A: Fixed-size character chunking with overlap (standard RAG)
- Common default in LangChain, LlamaIndex
- Simple to implement — no format awareness needed
- Ignores clause boundaries — splits "The Company shall (a) indemnify..." 
  at character 500 regardless of whether that's mid-clause
- Citations are meaningless: "characters 4500-5000, page 8"
- **Rejected** — clause boundary violations are unacceptable for a legal tool

### Option B: Sentence-level chunking
- Splits on `.`, `!`, `?` boundaries
- Legal sentences routinely run 200-400 words across multiple lines —
  a single sentence in Section 4 of the Agency Agreement runs ~600 words
- One-sentence chunks lose definitional context
- **Rejected** — legal sentences are not semantic units

### Option C: Recursive text splitter (LangChain default)
- Tries paragraph → sentence → word boundaries in order
- Better than pure character chunking but still format-agnostic
- Would still split `(a) The Receiver must...` from its parent clause number
- **Rejected** — still ignores contract structure

### Option D: Format-detecting clause-aware chunking (chosen)
- Detect document format first, dispatch to appropriate strategy
- Each strategy respects the actual semantic boundaries of that format
- Fallback to paragraph chunking when structure is absent
- More implementation work, but the only approach that produces
  citation-accurate, semantically coherent chunks for legal text

---

## 3. Decision

**Format-detecting, strategy-dispatched clause chunker** implemented in
`src/pipeline/chunker.py`.

### Format Detection

A sample of the first 10 pages is analysed using both regex pattern matching
on flat text and PyMuPDF block-level layout inspection (bold flags, x0
coordinates). The first strategy whose signal fires above a threshold of 3
matches wins.

### Strategies (in detection priority order)

**Strategy 1 — `section_n` (Agency Agreement style):**  
Signal: `^Section\s+\d+\.` regex match AND the line is bold in PyMuPDF layout.  
The bold check is critical — body text contains cross-references like
"as defined in Section 9(a)" which the regex alone would falsely match.

**Strategy 2 — `bare_n` (Affiliate Agreement style):**  
Signal: `^\d+[\.\xa0]\s+[A-Z]` regex, with sequential-number validation
and short-line length check.  
Font-based signals are unusable here — heading and body text are visually
identical (same font, same size, same weight). Sequence checking (reject
numbers that jump by more than 3) filters false positives like dollar amounts
and page numbers.

**Strategy 3 — `onenda_table` (oneNDA style):**  
Signal: bold isolated number blocks (`"1."`, `"2."`) followed by bold title
text at the same x0 coordinate.  
This format is a PyMuPDF layout artifact of table-cell PDFs — clause numbers
and their content land as physically separate text blocks even though they're
visually adjacent. The chunker stitches them back together by walking the
block stream and merging by coordinate proximity and bold continuity.  
Sub-clause labels like `(a)` also land alone on their own lines — the stitching
loop peeks forward to attach the following content lines.

**Strategy 4 — `fallback_prose`:**  
Signal: none of the above fired above threshold.  
Splits on double newlines (paragraph boundaries). Each paragraph becomes a
chunk with ID `Para N`. Used for prose-heavy contracts with no numbering
and as a safe fallback for formats not yet covered.

### Sub-clause splitting

Within each top-level chunk, sub-clauses `(a)`, `(b)`, `(i)`, `(ii)` etc.
are further split into their own chunks for finer retrieval granularity.
Sub-clause chunks carry the parent ID in their clause_id:
`Section 4` → `Section 4(a)`, `Section 4(b)`, etc.

The retriever's MMR step handles the resulting increase in chunk count by
filtering redundant sub-clauses that cover the same content.

---

## 4. Consequences

**Positive:**
- Citation accuracy: every chunk maps to exactly one named clause/sub-clause
  with a correct page range
- Semantic coherence: chunks respect the boundaries the contract authors
  intended — a chunk is a clause, not an arbitrary text window
- Format robustness: three real contract formats confirmed working
  (Agency Agreement, Chase Affiliate Agreement, oneNDA v2.1)
- Fallback safety: unknown formats degrade gracefully to paragraph chunks

**Negative / Accepted Trade-offs:**
- More complex than a fixed-size chunker — three strategies + fallback
  instead of one universal approach
- Format detection can misfire on edge cases (e.g. a contract that uses
  `Section` as a cross-reference word but doesn't actually have bold section
  headers). Threshold of 3 matches reduces false positives.
- The `onenda_table` strategy re-opens the PDF via fitz for block-level
  inspection — slight performance cost vs. flat-text-only strategies
- Duplicate clause_id values can occur when the same sub-clause label
  (e.g. `(i)`) exists under multiple parent clauses — IDs must be made
  unique at the embedder level using the `source_name::clause_id` composite

---

## 5. Empirical Results

Tested against four real contracts:

| Contract | Format Detected | Chunks Produced | Notes |
|---|---|---|---|
| Athens Bancshares Agency Agreement (33pp) | `section_n` | 161 | Section 1–20, sub-clauses (a)–(rr) |
| Chase Affiliate Agreement (12pp) | `bare_n` | 23 | Sections 1–23 matched exactly |
| oneNDA v2.1 (3pp) | `onenda_table` | 33 | Metadata chunk + 5 clauses + sub-clauses |
| NVIDIA 8-K EDGAR filing (190pp) | `fallback_prose` | — | SEC markup detected, warning logged |

The NVIDIA file confirmed that non-contract PDFs (SEC EDGAR filings with
inline XBRL) are detected early and handled gracefully rather than producing
meaningless chunks.
