from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from src.search.intent import SearchIntent


class RelevanceExplainer:
    def __init__(
        self,
        *,
        model: str = "gemini-1.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env

    def explain(self, query: str, intent: SearchIntent, candidate_row: pd.Series) -> str:
        breakdown = self._breakdown_text(intent, candidate_row)

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            narrative = self._fallback_explain(query, intent, candidate_row)
            return f"{breakdown}\n{narrative}".strip()

        try:
            narrative = self._explain_with_gemini(query, intent, candidate_row)
        except Exception:
            narrative = self._fallback_explain(query, intent, candidate_row)

        return f"{breakdown}\n{narrative}".strip()

    def _explain_with_gemini(self, query: str, intent: SearchIntent, candidate_row: pd.Series) -> str:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv(self.api_key_env))
        model = genai.GenerativeModel(self.model)

        profile = str(candidate_row.get("combined_text", "") or "")
        yoe = candidate_row.get("total_years_exp", "")
        avg_tenure = candidate_row.get("avg_tenure", "")
        is_job_hopper = candidate_row.get("is_job_hopper", "")

        prompt = (
            "You are an expert technical recruiter. "
            f"Given the user query \"{query}\" and the candidate profile \"{profile}\", "
            "provide a 2-3 sentence explanation of why this candidate is a match. "
            "Mention specific strengths (e.g., \"5 years of Python experience\") and any potential gaps "
            "(e.g., \"Based in Hyderabad, not Bangalore\"). Be concise and professional.\n\n"
            f"Structured intent: {intent.to_dict()}\n"
            f"Candidate signals: total_years_exp={yoe}, avg_tenure={avg_tenure}, is_job_hopper={is_job_hopper}\n"
            "Return ONLY the explanation text." 
        )

        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return " ".join(text.split())

    @staticmethod
    def _breakdown_text(intent: SearchIntent, candidate_row: pd.Series) -> str:
        base = float(candidate_row.get("base_score", 0.0) or 0.0)
        mult = float(candidate_row.get("score_multiplier", 1.0) or 1.0)
        final = float(candidate_row.get("final_score", 0.0) or 0.0)

        boost_yoe = bool(candidate_row.get("boost_yoe", False))
        boost_location = bool(candidate_row.get("boost_location", False))
        boost_stability = bool(candidate_row.get("boost_stability", False))
        penalty_job_hopper = bool(candidate_row.get("penalty_job_hopper", False))

        applied = []
        if boost_yoe:
            applied.append("YOE boost (x1.4)")
        if boost_location:
            applied.append("Location boost (x1.3)")
        if boost_stability:
            applied.append("Stability boost (x1.5)")
        if penalty_job_hopper:
            applied.append("Job-hopper penalty (x0.5)")

        if applied:
            applied_txt = ", ".join(applied)
        else:
            applied_txt = "None"

        intent_bits = []
        if int(intent.min_yoe or 0) > 0:
            intent_bits.append(f"min_yoe={int(intent.min_yoe)}")
        if (intent.location or "").strip():
            intent_bits.append(f"location=\"{intent.location.strip()}\"")
        if bool(intent.stability_required):
            intent_bits.append("stability_required=true")
        if not intent_bits:
            intent_bits.append("(no structured constraints extracted)")

        return (
            "Ranking breakdown:\n"
            f"- base_score: {base:.4f}\n"
            f"- multiplier: {mult:.2f} ({applied_txt})\n"
            f"- final_score: {final:.4f}\n"
            f"- intent: {', '.join(intent_bits)}"
        )

    @staticmethod
    def _fallback_explain(query: str, intent: SearchIntent, candidate_row: pd.Series) -> str:
        yoe = candidate_row.get("total_years_exp", 0)
        is_job_hopper = bool(candidate_row.get("is_job_hopper", False))
        text = str(candidate_row.get("combined_text", "") or "")

        parts = []
        if intent.min_yoe and float(yoe or 0) >= float(intent.min_yoe):
            parts.append(f"Meets the {intent.min_yoe}+ years experience requirement (approx. {yoe} years).")
        elif intent.min_yoe:
            parts.append(f"May be under the {intent.min_yoe}+ years experience requirement (approx. {yoe} years).")

        if intent.location:
            if intent.location.lower() in text.lower():
                parts.append(f"Location appears to match: {intent.location}.")
            else:
                parts.append(f"Location is not clearly {intent.location} based on the profile text.")

        if intent.stability_required:
            if not is_job_hopper:
                parts.append("Career history suggests stability (not a job hopper).")
            else:
                parts.append("Career history suggests shorter tenures (possible job hopper).")

        if not parts:
            parts.append("Strong semantic match to the query based on the resume content.")

        out = " ".join(parts)
        return out[:500]
