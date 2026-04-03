from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

from src.api.explainer import RelevanceExplainer
from src.core.indexer import CandidateIndexer
from src.core.loader import load_candidates_excel
from src.search.intent import IntentExtractor
from src.search.ranker import HybridRanker


EVAL_QUERIES = [
    "Senior Python developer in Bangalore, 4+ years",
    "Frontend React engineer, 3+ years, remote",
    "Data Scientist with NLP and LLM experience, 2+ years",
    "Backend engineer with FastAPI and PostgreSQL, 5+ years",
    "DevOps engineer with AWS, Docker, Kubernetes, 4+ years",
    "Machine Learning engineer with PyTorch, 3+ years",
    "Full-stack engineer (React + Node), 4+ years",
    "Java developer with Spring Boot, 5+ years",
    "Golang backend engineer, 3+ years",
    "Android developer with Kotlin, 3+ years",
    "iOS developer with Swift, 3+ years",
    "QA automation engineer with Selenium, 3+ years",
    "Product engineer for SaaS, 5+ years",
    "Fintech engineer with payments domain, 4+ years",
    "Startup engineer (early-stage), 4+ years",
    "Senior Python dev in Bangalore, 4+ years, not a job hopper",
    "ML engineer who's done production deployment, not just notebooks"
]


def _load_simple_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in dict(**__import__("os").environ):
            __import__("os").environ[k] = v


def _pick_name(row: pd.Series) -> str:
    for c in ["name", "candidate name", "full name", "candidate", "email"]:
        if c in row.index and str(row.get(c) or "").strip():
            return str(row.get(c)).strip()
    return "(Unnamed Candidate)"


def _ensure_index(df, artifacts_dir: Path, rebuild: bool) -> CandidateIndexer:
    if not rebuild and (artifacts_dir / "faiss.index").exists() and (artifacts_dir / "meta.json").exists():
        return CandidateIndexer.load_index(artifacts_dir)
    indexer = CandidateIndexer().build(df)
    indexer.save_index(artifacts_dir)
    return indexer


def main() -> int:
    console = Console()

    parser = argparse.ArgumentParser(prog="batch-eval")
    parser.add_argument(
        "--excel",
        type=str,
        default=str(Path("data") / "Candidates and Jobs.xlsx"),
        help="Path to candidate Excel file",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(Path("artifacts") / "candidate_index"),
        help="Directory to save/load FAISS index artifacts",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild embeddings + FAISS index")

    args = parser.parse_args()

    _load_simple_dotenv(Path(".env"))

    excel_path = Path(args.excel)
    artifacts_dir = Path(args.index_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_candidates_excel(excel_path)
    indexer = _ensure_index(df, artifacts_dir, rebuild=bool(args.rebuild))

    intent_extractor = IntentExtractor()
    ranker = HybridRanker(indexer=indexer, candidates_df=df, intent_extractor=intent_extractor)
    explainer = RelevanceExplainer()

    out_path = Path("docs") / "analysis_results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Candidate Search Evaluation Results", ""]

    for q in EVAL_QUERIES:
        intent = intent_extractor.extract(q)
        top = ranker.rank(q, top_k_semantic=100, top_k_final=5)

        lines.append(f"## Query: {q}")
        lines.append("")

        for _, row in top.iterrows():
            score = float(row.get("final_score", 0.0) or 0.0)
            why = explainer.explain(q, intent, row)
            name = _pick_name(row)
            combined = str(row.get("combined_text", "") or "")
            preview = combined[:200].replace("\n", " ")
            lines.append(
                f"### Rank {int(row.get('rank', 0) or 0)} | {name} | Score {score:.4f}"
            )
            lines.append("")
            lines.append(f"- Preview: {preview}")
            lines.append(f"- Why: {why}")
            lines.append("")

        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
