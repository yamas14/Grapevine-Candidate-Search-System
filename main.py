from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.api.explainer import RelevanceExplainer
from src.core.indexer import CandidateIndexer
from src.core.loader import load_candidates_excel
from src.search.intent import IntentExtractor
from src.search.ranker import HybridRanker


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


def _ensure_index(
    *,
    df: pd.DataFrame,
    artifacts_dir: Path,
    force_rebuild: bool,
) -> CandidateIndexer:
    if not force_rebuild and (artifacts_dir / "faiss.index").exists() and (artifacts_dir / "meta.json").exists():
        return CandidateIndexer.load_index(artifacts_dir)

    indexer = CandidateIndexer()
    indexer.build(df)
    indexer.save_index(artifacts_dir)
    return indexer


def main() -> int:
    parser = argparse.ArgumentParser(prog="candidate-search")
    parser.add_argument("query", type=str, nargs="?", help="Natural language search query")
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

    console = Console()

    _load_simple_dotenv(Path(".env"))

    query = (args.query or "").strip()
    if not query:
        query = console.input("Enter your query: ").strip()
        if not query:
            console.print("No query provided.")
            return 2

    excel_path = Path(args.excel)
    if not excel_path.exists():
        console.print(f"Excel file not found: {excel_path}")
        return 2

    df = load_candidates_excel(excel_path)

    artifacts_dir = Path(args.index_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    indexer = _ensure_index(df=df, artifacts_dir=artifacts_dir, force_rebuild=bool(args.rebuild))

    intent_extractor = IntentExtractor()
    intent = intent_extractor.extract(query)

    ranker = HybridRanker(indexer=indexer, candidates_df=df, intent_extractor=intent_extractor)
    results = ranker.rank(query, top_k_semantic=100, top_k_final=20)

    explainer = RelevanceExplainer()

    table = Table(title="Candidate Search Results")
    table.add_column("Rank", justify="right", no_wrap=True)
    table.add_column("Name", overflow="fold")
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Why it's a match", overflow="fold")

    for _, row in results.iterrows():
        why = explainer.explain(query, intent, row)
        table.add_row(
            str(int(row.get("rank", 0) or 0)),
            _pick_name(row),
            f"{float(row.get('final_score', 0.0) or 0.0):.4f}",
            why,
        )

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
