"""Shared dataclasses for examples, results, and consistency rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Example:
    dataset: str
    row_id: int
    text1: str
    text2: str
    gold_label: str


@dataclass
class Result:
    dataset: str
    row_id: int
    method: str
    gold_label: str
    pred_label: Optional[str]
    correct: Optional[bool]
    final_score: Optional[float]
    prompt: str
    raw_response: str
    parsed_json: Optional[dict[str, Any]]
    backend: str
    model: str
    model_revision: str = ""
    error: Optional[str] = None


@dataclass
class ConsistencyRow:
    dataset: str
    row_id: int
    method: str
    gold_label: str
    pred_label: Optional[str]
    final_score: Optional[float]
    cs1: Optional[int]
    cs2: Optional[int]
    consistency: Optional[int]
