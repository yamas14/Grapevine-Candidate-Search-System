from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.indexer import CandidateIndexer
from src.search.intent import IntentExtractor, SearchIntent


@dataclass
class RankedCandidate:
    idx: int
    base_distance: float
    base_score: float
    final_score: float
    breakdown: Dict[str, Any]


class HybridRanker:
    def __init__(
        self,
        *,
        indexer: CandidateIndexer,
        candidates_df: pd.DataFrame,
        intent_extractor: Optional[IntentExtractor] = None,
        location_text_column: str = "combined_text",
        yoe_column: str = "total_years_exp",
        job_hopper_column: str = "is_job_hopper",
    ) -> None:
        self.indexer = indexer
        self.df = candidates_df
        self.intent_extractor = intent_extractor or IntentExtractor()

        self.location_text_column = location_text_column
        self.yoe_column = yoe_column
        self.job_hopper_column = job_hopper_column

        if self.location_text_column not in self.df.columns:
            raise ValueError(
                f"Expected column '{self.location_text_column}' in candidates_df. "
                f"Available columns: {list(self.df.columns)}"
            )

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        # For L2 distance: smaller is better. Convert to a bounded-ish score.
        return float(1.0 / (1.0 + max(distance, 0.0)))

    def rank(self, query: str, *, top_k_semantic: int = 100, top_k_final: int = 20) -> pd.DataFrame:
        intent = self.intent_extractor.extract(query)
        distances, indices = self.indexer.search(query, top_k=top_k_semantic)

        records: List[RankedCandidate] = []
        for dist, idx in zip(distances.tolist(), indices.tolist()):
            if idx < 0 or idx >= len(self.df):
                continue
            base_score = self._distance_to_score(float(dist))
            final_score, breakdown = self._apply_multipliers(base_score, idx, intent)
            records.append(
                RankedCandidate(
                    idx=int(idx),
                    base_distance=float(dist),
                    base_score=float(base_score),
                    final_score=float(final_score),
                    breakdown=breakdown,
                )
            )

        if not records:
            return self.df.head(0).copy()

        ranked = sorted(records, key=lambda r: r.final_score, reverse=True)[: int(top_k_final)]
        out = self.df.iloc[[r.idx for r in ranked]].copy()
        out["base_distance"] = [r.base_distance for r in ranked]
        out["base_score"] = [r.base_score for r in ranked]
        out["final_score"] = [r.final_score for r in ranked]
        out["score_multiplier"] = [float(r.breakdown.get("score_multiplier", 1.0) or 1.0) for r in ranked]
        out["boost_yoe"] = [bool(r.breakdown.get("boost_yoe", False)) for r in ranked]
        out["boost_location"] = [bool(r.breakdown.get("boost_location", False)) for r in ranked]
        out["boost_stability"] = [bool(r.breakdown.get("boost_stability", False)) for r in ranked]
        out["penalty_job_hopper"] = [bool(r.breakdown.get("penalty_job_hopper", False)) for r in ranked]
        out["rank"] = list(range(1, len(out) + 1))

        return out.sort_values("final_score", ascending=False).reset_index(drop=True)

    def _apply_multipliers(self, score: float, idx: int, intent: SearchIntent) -> tuple[float, Dict[str, Any]]:
        s = float(score)

        breakdown: Dict[str, Any] = {
            "intent": intent.to_dict(),
            "base_score": float(score),
            "score_multiplier": 1.0,
            "boost_yoe": False,
            "boost_location": False,
            "boost_stability": False,
            "penalty_job_hopper": False,
        }

        row = self.df.iloc[idx]

        # Experience match
        min_yoe = int(intent.min_yoe or 0)
        if min_yoe > 0:
            try:
                cand_yoe = float(row.get(self.yoe_column, 0.0) or 0.0)
            except Exception:
                cand_yoe = 0.0
            if cand_yoe >= float(min_yoe):
                s *= 1.4
                breakdown["boost_yoe"] = True
                breakdown["score_multiplier"] = float(breakdown["score_multiplier"]) * 1.4

        # Location match (substring in combined text)
        loc = (intent.location or "").strip()
        if loc:
            text = str(row.get(self.location_text_column, "") or "")
            if loc.lower() in text.lower():
                s *= 1.3
                breakdown["boost_location"] = True
                breakdown["score_multiplier"] = float(breakdown["score_multiplier"]) * 1.3

        # Stability match / penalty
        if intent.stability_required:
            is_job_hopper = bool(row.get(self.job_hopper_column, False))
            if not is_job_hopper:
                s *= 1.5
                breakdown["boost_stability"] = True
                breakdown["score_multiplier"] = float(breakdown["score_multiplier"]) * 1.5
            else:
                s *= 0.5
                breakdown["penalty_job_hopper"] = True
                breakdown["score_multiplier"] = float(breakdown["score_multiplier"]) * 0.5

        return s, breakdown
