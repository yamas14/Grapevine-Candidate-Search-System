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
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return self._fallback_explain(query, intent, candidate_row)

        try:
            return self._explain_with_gemini(query, intent, candidate_row)
        except Exception:
            return self._fallback_explain(query, intent, candidate_row)

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
