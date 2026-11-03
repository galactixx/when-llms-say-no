"""JSON extraction for CAVE methods and decimal score extraction for PromptAV methods."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

from json_repair import repair_json

from ..shared.constants import PROMPTAV_SCORE_METHODS, PROMPTAV_SCORE_THRESHOLD


def extract_json_blob(text: str) -> Optional[str]:
    fence = re.search(
        r"```(?:json|python)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_cave_json_like_blob(blob: str) -> Optional[dict[str, Any]]:
    """Parse CAVE JSON output.

    Order:
    1. Strict JSON via json.loads
    2. Python literal via ast.literal_eval
    3. json-repair fallback via repair_json + json.loads
    """
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(blob)
            if isinstance(parsed, dict):
                return dict(parsed)
        except Exception:
            continue

    try:
        repaired = repair_json(blob)
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return dict(parsed)
    except Exception:
        pass

    return None


def promptav_decimal_scores(raw: str) -> list[str]:
    """Match the released PromptAV code's confidence-score regex."""
    sentence = raw.lower().strip()
    return re.findall(r"\b0?\.\d+\b", sentence)


def promptav_assign_label(
    score: str, threshold: float = PROMPTAV_SCORE_THRESHOLD
) -> str:
    """PromptAV-style fixed-threshold label assignment."""
    return "YES" if float(score) > threshold else "NO"


def parse_promptav_style_output(
    raw: str,
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[float]]:
    pred_answers = promptav_decimal_scores(raw)
    if not pred_answers:
        return None, None, None
    score_text = pred_answers[-1].replace(" ", "")
    return None, promptav_assign_label(score_text), float(score_text)


def parse_model_output(
    raw: str, method: str
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[float]]:
    if method in PROMPTAV_SCORE_METHODS:
        return parse_promptav_style_output(raw)
    blob = extract_json_blob(raw)
    obj: Optional[dict[str, Any]] = None
    pred_label: Optional[str] = None
    final_score: Optional[float] = None

    if blob is not None:
        obj = parse_cave_json_like_blob(blob)

    if obj is not None:
        output_val = obj.get("output")
        if isinstance(output_val, str):
            lab = output_val.strip().upper()
            if lab in {"YES", "NO", "MAYBE"}:
                pred_label = lab
        score_val = obj.get("final score")
        if score_val is None:
            score_val = obj.get("final_score")
        if score_val is not None:
            try:
                final_score = float(score_val)
            except (TypeError, ValueError):
                final_score = None

    # JSON methods are scored strictly from CAVE-style structured output.
    return obj, pred_label, final_score
