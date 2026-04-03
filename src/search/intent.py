from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchIntent:
    skills: List[str]
    min_yoe: int
    location: str
    is_startup: bool
    stability_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skills": self.skills,
            "min_yoe": int(self.min_yoe or 0),
            "location": self.location or "",
            "is_startup": bool(self.is_startup),
            "stability_required": bool(self.stability_required),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchIntent":
        skills = d.get("skills") or []
        if isinstance(skills, str):
            skills = [skills]
        if not isinstance(skills, list):
            skills = []
        skills = [str(s).strip() for s in skills if str(s).strip()]

        min_yoe = d.get("min_yoe")
        try:
            min_yoe_int = int(min_yoe) if min_yoe is not None else 0
        except Exception:
            min_yoe_int = 0

        location = d.get("location")
        location_str = str(location).strip() if location is not None else ""

        is_startup = bool(d.get("is_startup", False))
        stability_required = bool(d.get("stability_required", False))

        return cls(
            skills=skills,
            min_yoe=min_yoe_int,
            location=location_str,
            is_startup=is_startup,
            stability_required=stability_required,
        )


class IntentExtractor:
    def __init__(
        self,
        *,
        model: str = "gemini-1.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env

    def extract(self, query: str) -> SearchIntent:
        q = str(query or "").strip()
        if not q:
            return SearchIntent(skills=[], min_yoe=0, location="", is_startup=False, stability_required=False)

        # Always detect this locally (requested behavior)
        stability_required = bool(re.search(r"not\s+a\s+job\s+hopper|stable|stability", q, re.IGNORECASE))

        api_key = os.getenv(self.api_key_env)
        if api_key:
            try:
                return self._extract_with_gemini(q, stability_required=stability_required)
            except Exception:
                # Fall back to regex extraction if anything goes wrong
                pass

        intent = self._extract_with_regex(q)
        if stability_required:
            intent.stability_required = True
        return intent

    def _extract_with_gemini(self, query: str, *, stability_required: bool) -> SearchIntent:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv(self.api_key_env))
        model = genai.GenerativeModel(self.model)

        prompt = (
            "Extract hiring constraints from the user query and return ONLY valid JSON. "
            "No markdown, no code fences. JSON schema:\n"
            "{\"skills\": [string], \"min_yoe\": int, \"location\": string, \"is_startup\": bool, \"stability_required\": bool}\n\n"
            "Rules:\n"
            "- skills should be a list of key technical skills explicitly mentioned\n"
            "- min_yoe should be an integer minimum years of experience if present, else 0\n"
            "- location should be a city/region if present, else empty string\n"
            "- stability_required should be true if query implies 'not a job hopper' or stability\n"
            "- is_startup should be true if query mentions startup/early-stage\n\n"
            f"Query: {query}\n"
        )

        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()

        # Attempt to isolate JSON if model returns extra text
        json_text = self._best_effort_json_extract(text)
        data = json.loads(json_text)
        intent = SearchIntent.from_dict(data)

        if stability_required:
            intent.stability_required = True
        return intent

    def _extract_with_regex(self, query: str) -> SearchIntent:
        q = query

        min_yoe = 0
        m = re.search(r"(\d+)\s*\+?\s*(?:years?|yrs?)", q, re.IGNORECASE)
        if m:
            try:
                min_yoe = int(m.group(1))
            except Exception:
                min_yoe = 0

        stability_required = bool(re.search(r"not\s+a\s+job\s+hopper|stable|stability", q, re.IGNORECASE))
        is_startup = bool(re.search(r"\bstartup\b|\bearly[-\s]?stage\b", q, re.IGNORECASE))

        location = ""
        mloc = re.search(r"\b(?:in|at|based\s+in)\s+([A-Za-z][A-Za-z\s\-]{2,40})", q, re.IGNORECASE)
        if mloc:
            location = mloc.group(1).strip().rstrip(".,")
            location = re.split(r"\bwith\b|\band\b|\bfor\b", location, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        skills: List[str] = []
        common_skills = [
            "python",
            "java",
            "golang",
            "go",
            "javascript",
            "typescript",
            "react",
            "angular",
            "node",
            "fastapi",
            "django",
            "flask",
            "aws",
            "gcp",
            "azure",
            "docker",
            "kubernetes",
            "sql",
            "postgres",
            "mongodb",
            "spark",
            "pytorch",
            "tensorflow",
            "llm",
            "nlp",
        ]
        ql = q.lower()
        for sk in common_skills:
            if re.search(rf"\b{re.escape(sk)}\b", ql):
                skills.append("Go" if sk == "go" else sk.title() if sk.isalpha() else sk)

        skills = list(dict.fromkeys(skills))

        return SearchIntent(
            skills=skills,
            min_yoe=min_yoe,
            location=location,
            is_startup=is_startup,
            stability_required=stability_required,
        )

    @staticmethod
    def _best_effort_json_extract(text: str) -> str:
        t = text.strip()
        if t.startswith("{") and t.endswith("}"):
            return t
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1]
        return t
