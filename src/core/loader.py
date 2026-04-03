from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


MONTHS: Dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class ExperienceStats:
    total_years_exp: float
    avg_tenure_years: float
    is_job_hopper: bool


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name


def _coalesce_str(*values: Any) -> str:
    for v in values:
        if not _is_nullish(v):
            return str(v).strip()
    return ""


def _parse_month_token(token: str) -> Optional[int]:
    token = token.strip().lower().strip(".,")
    if token in MONTHS:
        return MONTHS[token]
    return None


def _parse_year_token(token: str) -> Optional[int]:
    token = token.strip().lower().strip(".,")
    if not token:
        return None
    if token.isdigit():
        y = int(token)
        if 1900 <= y <= 2100:
            return y
        if 0 <= y <= 99:
            # Heuristic: 00-29 => 2000s, else 1900s
            return 2000 + y if y <= 29 else 1900 + y
    return None


def _safe_date(year: int, month: int) -> date:
    # Normalize month to [1, 12]
    month = max(1, min(12, int(month)))
    return date(int(year), int(month), 1)


def _months_between(a: date, b: date) -> int:
    # inclusive-ish month difference, clamped to >= 0
    if b < a:
        a, b = b, a
    return (b.year - a.year) * 12 + (b.month - a.month)


# Common date-range patterns found in resumes
_DATE_RANGE_PATTERNS: List[re.Pattern[str]] = [
    # Apr 2021 - Feb 2023
    re.compile(
        r"(?P<m1>[A-Za-z]{3,9})\s+(?P<y1>\d{2,4})\s*(?:-|to|–|—)\s*(?P<m2>[A-Za-z]{3,9}|present|current)\s*(?P<y2>\d{2,4})?",
        re.IGNORECASE,
    ),
    # 2021 - 2023, 2021 - Present
    re.compile(
        r"(?P<y1>\d{4})\s*(?:-|to|–|—)\s*(?P<y2>\d{4}|present|current)",
        re.IGNORECASE,
    ),
]


def extract_date_ranges_from_text(text: str) -> List[Tuple[date, date]]:
    """Extract approximate employment date ranges from unstructured resume text.

    Returns list of (start_date, end_date) pairs, where end_date may be 'today'
    for present/current roles.

    This is heuristic and intentionally permissive.
    """

    if _is_nullish(text):
        return []

    txt = str(text)
    today = date.today()
    ranges: List[Tuple[date, date]] = []

    for pat in _DATE_RANGE_PATTERNS:
        for m in pat.finditer(txt):
            gd = m.groupdict()

            if "m1" in gd and gd.get("m1") and gd.get("y1"):
                m1 = _parse_month_token(gd["m1"])
                y1 = _parse_year_token(gd["y1"])
                if m1 is None or y1 is None:
                    continue

                m2_raw = (gd.get("m2") or "").strip().lower()
                if m2_raw in {"present", "current"}:
                    end = today
                else:
                    m2 = _parse_month_token(gd.get("m2") or "")
                    y2 = _parse_year_token(gd.get("y2") or "")
                    if m2 is None:
                        # if month missing but year present, approximate January
                        if y2 is not None:
                            m2 = 1
                        else:
                            continue
                    if y2 is None:
                        # If year missing but month present, ignore
                        continue
                    end = _safe_date(y2, m2)

                start = _safe_date(y1, m1)
                if end < start:
                    start, end = end, start

                # filter obvious noise (e.g., 1 month ranges from education timelines)
                if _months_between(start, end) >= 1:
                    ranges.append((start, end))

            elif gd.get("y1") and gd.get("y2"):
                y1 = _parse_year_token(gd["y1"])
                if y1 is None:
                    continue
                y2_raw = str(gd["y2"]).strip().lower()
                if y2_raw in {"present", "current"}:
                    end = today
                else:
                    y2 = _parse_year_token(y2_raw)
                    if y2 is None:
                        continue
                    end = _safe_date(y2, 1)

                start = _safe_date(y1, 1)
                if end < start:
                    start, end = end, start

                if _months_between(start, end) >= 3:
                    ranges.append((start, end))

    return ranges


def compute_experience_stats_from_ranges(
    ranges: List[Tuple[date, date]],
    *,
    job_hopper_avg_tenure_threshold_years: float = 1.5,
) -> ExperienceStats:
    """Compute total years of experience and average tenure from date ranges.

    - total_years_exp: sum of non-overlapping months across ranges
    - avg_tenure_years: average months per role / 12
    - is_job_hopper: avg_tenure_years < threshold
    """

    if not ranges:
        return ExperienceStats(total_years_exp=0.0, avg_tenure_years=0.0, is_job_hopper=False)

    # Normalize + sort
    norm = sorted([(min(a, b), max(a, b)) for a, b in ranges], key=lambda x: x[0])

    # Merge overlaps for total experience
    merged: List[Tuple[date, date]] = []
    cur_s, cur_e = norm[0]
    for s, e in norm[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    total_months = sum(_months_between(s, e) for s, e in merged)

    # Avg tenure per role (use original norm to count roles)
    role_months = [_months_between(s, e) for s, e in norm if _months_between(s, e) >= 1]
    if not role_months:
        avg_tenure_years = 0.0
    else:
        avg_tenure_years = (sum(role_months) / len(role_months)) / 12.0

    total_years_exp = total_months / 12.0
    is_job_hopper = avg_tenure_years > 0 and avg_tenure_years < job_hopper_avg_tenure_threshold_years

    return ExperienceStats(
        total_years_exp=round(total_years_exp, 2),
        avg_tenure_years=round(avg_tenure_years, 2),
        is_job_hopper=bool(is_job_hopper),
    )


def compute_experience_stats(work_history: Any, resume_text: Any = None) -> ExperienceStats:
    """Compute experience stats from either a structured Work History field or resume text.

    Supported inputs:
    - work_history as string (free-form). We extract date ranges.
    - resume_text as string, used as fallback.

    If both are present, work_history is preferred.
    """

    src = None
    if not _is_nullish(work_history):
        src = str(work_history)
    elif not _is_nullish(resume_text):
        src = str(resume_text)

    if not src:
        return ExperienceStats(total_years_exp=0.0, avg_tenure_years=0.0, is_job_hopper=False)

    ranges = extract_date_ranges_from_text(src)
    return compute_experience_stats_from_ranges(ranges)


def build_combined_text(row: pd.Series) -> str:
    title = _coalesce_str(row.get("title"), row.get("current title"), row.get("designation"))
    company = _coalesce_str(row.get("company"), row.get("current company"), row.get("company name"))
    skills = _coalesce_str(row.get("skills"), row.get("skill"))
    bio = _coalesce_str(row.get("bio"), row.get("summary"), row.get("resume text"))

    parts: List[str] = []
    if title or company:
        parts.append(f"{title} at {company}".strip())
    if skills:
        parts.append(f"Skills: {skills}")
    if bio:
        parts.append(f"Bio: {bio}")

    return ". ".join([p for p in parts if p]).strip()


def load_candidates_excel(
    excel_path: str | Path,
    *,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """Load candidate profiles from Excel.

    - If sheet_name is None, loads and concatenates any sheets whose name contains 'candidate'
      else loads the first sheet.
    - Normalizes columns to lowercase names.
    - Adds enrichment columns:
        - total_years_exp
        - avg_tenure
        - is_job_hopper
        - combined_text

    Returns a cleaned DataFrame.
    """

    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    xl = pd.ExcelFile(path)

    if sheet_name is None:
        candidate_sheets = [s for s in xl.sheet_names if "candidate" in str(s).lower()]
        if candidate_sheets:
            frames = [xl.parse(s) for s in candidate_sheets]
            df = pd.concat(frames, ignore_index=True)
        else:
            df = xl.parse(xl.sheet_names[0])
    else:
        df = xl.parse(sheet_name)

    # Normalize column names
    df = df.rename(columns={c: _normalize_column_name(c) for c in df.columns})

    # Compute experience stats
    work_hist_col = None
    for candidate in ["work history", "work_history", "experience", "employment history"]:
        if candidate in df.columns:
            work_hist_col = candidate
            break

    resume_col = "resume text" if "resume text" in df.columns else None

    def _row_stats(row: pd.Series) -> ExperienceStats:
        wh = row.get(work_hist_col) if work_hist_col else None
        rt = row.get(resume_col) if resume_col else None
        return compute_experience_stats(wh, rt)

    stats = df.apply(_row_stats, axis=1)
    df["total_years_exp"] = stats.apply(lambda s: s.total_years_exp)
    df["avg_tenure"] = stats.apply(lambda s: s.avg_tenure_years)
    df["is_job_hopper"] = stats.apply(lambda s: s.is_job_hopper)

    # combined_text enrichment
    df["combined_text"] = df.apply(build_combined_text, axis=1)

    return df
