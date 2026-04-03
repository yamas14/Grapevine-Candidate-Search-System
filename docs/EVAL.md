# Evaluation Notes

This document describes how the system was evaluated and how to interpret the results.

## What we evaluated

- **Task**: Rank candidate profiles against natural-language queries using:
  - Semantic retrieval (embeddings + FAISS)
  - Metadata-aware reranking (YOE, location, stability)
  - LLM-generated concise explanations for the final shortlist

## Query set

A fixed set of 16 representative queries is defined in `batch_eval.py` (`EVAL_QUERIES`).

The queries cover:
- Backend / frontend / full-stack roles
- ML/NLP/LLM requirements
- Cloud/DevOps requirements
- Domain hints (Fintech/SaaS)
- Stability requirement (`not a job hopper`)

## How scoring works

1. **Semantic retrieval**
   - The system performs a vector search to get the top 100 by embedding similarity.

2. **Hybrid reranking**
   - A final score is computed by applying multipliers to the semantic score:
     - Experience match: `* 1.4` if `candidate.total_years_exp >= intent.min_yoe`
     - Location match: `* 1.3` if `intent.location` substring is found in candidate text
     - Stability required + stable: `* 1.5` if `stability_required` and `is_job_hopper == False`
     - Stability required + job hopper: `* 0.5`

3. **Final shortlist**
   - The system returns the top 20 for interactive CLI and top 5 for batch evaluation output.

## Outputs

- `analysis_results.md`
  - Produced by `python batch_eval.py`
  - For each query, stores the top 5 results:
    - Rank and score
    - A short preview of the candidate text
    - A concise explanation (Gemini if configured; otherwise fallback)

## Known limitations

- Candidate sheets may be unstructured (`Resume Text` only). Some constraints like location and skills may be missing or implicit.
- Experience parsing uses heuristics; unusual formats may reduce accuracy.
- Location matching is currently substring-based and will miss cases like abbreviations or nearby regions.

## Future improvements

- Add better location normalization (Bengaluru/Bangalore, NCR/Delhi, etc.).
- Extract explicit skills from resume text (NER/keywording) and compare to required skills.
- Add structured candidate parsing when structured exports are available.
- Tune reranking multipliers using evaluation feedback.
