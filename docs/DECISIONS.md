# Decision Log (ADR-lite)

This file captures the key engineering decisions made during the build of the Candidate Search System, along with the rationale and trade-offs.

## D001 — Data ingestion source format (Excel)
- **Status**: Accepted
- **Context**:
  - Input is provided as an `.xlsx` file in `data/`.
  - Candidate data may not be fully structured (may contain only `Resume Text`).
- **Decision**:
  - Use `pandas` + `openpyxl` to load the workbook and ingest all sheets containing the substring `candidate`.
- **Why**:
  - Works reliably on Windows, easy to inspect/debug.
  - Supports multi-sheet workbooks without manual concatenation.
- **Consequences**:
  - Requires `openpyxl` dependency.

## D002 — Handling unstructured resumes
- **Status**: Accepted
- **Context**:
  - Candidate sheets may contain only `Resume Text`.
  - Query constraints require structured signals like years of experience and stability.
- **Decision**:
  - Compute derived fields from text using heuristics when structured fields are missing.
  - Prefer a structured `work history`-type column if present; otherwise fall back to parsing date ranges from `resume text`.
- **Why**:
  - Enables search/ranking to work even with unstructured inputs.
  - Provides minimum viable signals for YOE and job-hopper detection.
- **Consequences**:
  - Heuristic parsing can miss or misread unusual date formats; should be improved if accuracy becomes critical.

## D003 — Experience metrics used for filtering/ranking
- **Status**: Accepted
- **Context**:
  - Queries can include `4+ years` and stability requirements like `not a job hopper`.
- **Decision**:
  - Parse date ranges and compute:
    - `total_years_exp`: sum of non-overlapping months across all ranges.
    - `avg_tenure`: average months per role.
    - `is_job_hopper`: `avg_tenure < 1.5` years.
- **Why**:
  - Non-overlapping sum reduces double-counting overlapping roles.
  - Avg tenure is a simple, explainable proxy for stability.
- **Consequences**:
  - Threshold is a heuristic; should be tuned with data/feedback.

## D004 — Text field used for semantic retrieval
- **Status**: Accepted
- **Context**:
  - We need semantic retrieval over resumes plus lightweight metadata checks.
- **Decision**:
  - Build a `combined_text` field (used as the embedding input) combining:
    - Title/company (if available)
    - Skills (if available)
    - Bio/summary/resume text
- **Why**:
  - Consolidates relevant signals into one searchable representation.
- **Consequences**:
  - If future structured columns are added, `combined_text` should be updated to incorporate them.

## D005 — Embedding model selection
- **Status**: Accepted
- **Context**:
  - Must embed ~2k documents locally with reasonable speed.
- **Decision**:
  - Use `sentence-transformers/all-MiniLM-L6-v2`.
- **Why**:
  - Strong speed/quality trade-off on CPU.
  - Common baseline for semantic search.
- **Consequences**:
  - Larger models may improve quality but increase latency/cost.

## D006 — Vector index selection
- **Status**: Accepted
- **Context**:
  - Dataset size is currently ~2k candidates.
- **Decision**:
  - Use FAISS `IndexFlatL2`.
- **Why**:
  - Simple, deterministic, no training required.
  - At this size, brute-force L2 is fast enough.
- **Consequences**:
  - For much larger datasets, migrate to IVF/HNSW.

## D007 — Avoid re-embedding every run
- **Status**: Accepted
- **Context**:
  - Re-encoding resumes every time is slow.
- **Decision**:
  - Persist artifacts:
    - FAISS index (`faiss.index`)
    - Embeddings (`embeddings.npy`)
    - Metadata (`meta.json`)
- **Why**:
  - Makes demos and iteration significantly faster.
- **Consequences**:
  - Must rebuild artifacts when the source dataset or embedding input changes.

## D008 — Intent extraction strategy
- **Status**: Accepted
- **Context**:
  - Need structured constraints from natural language queries.
- **Decision**:
  - Primary: Gemini 1.5 Flash returns strict JSON.
  - Fallback: regex-based extraction if API key is missing or the LLM fails.
- **Why**:
  - LLM improves coverage of phrasing variations.
  - Fallback keeps the system usable offline and reduces brittleness.
- **Consequences**:
  - LLM behavior can drift; prompt should be versioned if changes are made.

## D009 — Hybrid reranking multipliers
- **Status**: Accepted
- **Context**:
  - Semantic similarity alone may not respect hard constraints (YOE, location, stability).
- **Decision**:
  - Retrieve top 100 semantically, then apply multipliers:
    - Experience match: `* 1.4`
    - Location match: `* 1.3`
    - Stability required + stable: `* 1.5`
    - Stability required + job hopper: `* 0.5`
- **Why**:
  - Simple and explainable approach that prioritizes constraints without discarding semantic relevance.
- **Consequences**:
  - Weights are heuristic; should be tuned using evaluation feedback.

## D010 — Explanation generation scope
- **Status**: Accepted
- **Context**:
  - LLM calls cost tokens/time.
- **Decision**:
  - Call the explainer only for the final Top 20 results.
- **Why**:
  - Controls cost/latency while still providing user-facing reasoning.
- **Consequences**:
  - If Top 20 is too slow/costly, reduce count or cache explanations.

## D011 — Local `.env` loading
- **Status**: Accepted
- **Context**:
  - Developers often store API keys locally in `.env`.
- **Decision**:
  - `main.py`/`batch_eval.py` load `.env` in a minimal way to populate `GEMINI_API_KEY`.
- **Why**:
  - Improves developer experience in local runs.
- **Consequences**:
  - `.env` should never be committed; must be ignored in version control.
