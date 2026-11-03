"""Load/save results, resume deduplication, and summary/threshold-sensitivity generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .constants import ZERO_SHOT_METHODS
from .io_utils import iter_jsonl
from .metrics import compute_metrics, threshold_metrics
from .models import Result


def load_seen(result_path: Path) -> set[tuple[str, int, str]]:
    seen: set[tuple[str, int, str]] = set()
    if not result_path.exists():
        return seen
    for row in iter_jsonl(result_path):
        method = row.get("method")
        if method in ZERO_SHOT_METHODS:
            seen.add((row["dataset"], int(row["row_id"]), method))
    return seen


def save_csv(path: Path, rows: list[dict[str, Any]], default_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else default_fields
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def result_from_json(row: dict[str, Any], backend: str, model: str) -> Result:
    pred_label = row.get("pred_label")
    final_score = row.get("final_score")
    return Result(
        dataset=row["dataset"],
        row_id=int(row["row_id"]),
        method=row["method"],
        gold_label=row["gold_label"],
        pred_label=pred_label,
        correct=row.get("correct"),
        final_score=final_score,
        prompt=row.get("prompt", ""),
        raw_response=row.get("raw_response", ""),
        parsed_json=row.get("parsed_json"),
        backend=row.get("backend", backend),
        model=row.get("model", model),
        error=row.get("error"),
    )


def read_all_results(results_path: Path, backend: str, model: str) -> list[Result]:
    if not results_path.exists():
        return []
    rows: list[Result] = []
    for row in iter_jsonl(results_path):
        if row.get("method") not in ZERO_SHOT_METHODS:
            continue
        rows.append(result_from_json(row, backend, model))
    return rows


def summarize_results(results: list[Result]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Result]] = {}
    for r in results:
        by_key.setdefault((r.dataset, r.method), []).append(r)
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in sorted(by_key.items()):
        rows.append({"dataset": dataset, "method": method, **compute_metrics(group)})
    return rows


def make_threshold_sensitivity(
    results: list[Result], step: float
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Result]] = {}
    for r in results:
        by_key.setdefault((r.dataset, r.method), []).append(r)
    thresholds = []
    cur = 0.0
    while cur <= 1.000001:
        thresholds.append(round(cur, 6))
        cur += step
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in sorted(by_key.items()):
        if not any(r.final_score is not None for r in group):
            continue
        for thr in thresholds:
            rows.append(
                {"dataset": dataset, "method": method, **threshold_metrics(group, thr)}
            )
    return rows
