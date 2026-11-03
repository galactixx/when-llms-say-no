"""Accuracy, macro-F1, AUROC, threshold diagnostics, and CAVE consistency."""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from sklearn.metrics import roc_auc_score

from .constants import (
    CAVE_CONSISTENCY_METHODS,
    CAVE_CONSISTENCY_SCORE_THRESHOLD,
)
from .models import ConsistencyRow, Result


def safe_div(num: float, den: float) -> Optional[float]:
    return (num / den) if den else None


def binary_gold(gold_label: str) -> int:
    return 1 if gold_label == "YES" else 0


def f1_score(tp: int, fp: int, fn: int) -> Optional[float]:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else None


def compute_auroc(labels: list[int], scores: list[float]) -> Optional[float]:
    if not labels or len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


manual_auroc = compute_auroc


def threshold_metrics(
    results: list[Result], threshold: float
) -> dict[str, Optional[float]]:
    scored = [r for r in results if r.final_score is not None]
    if not scored:
        return {
            "threshold": threshold,
            "accuracy": None,
            "yes_accuracy": None,
            "no_accuracy": None,
        }

    yes_total = sum(1 for r in scored if r.gold_label == "YES")
    no_total = sum(1 for r in scored if r.gold_label == "NO")
    yes_correct = 0
    no_correct = 0
    correct = 0
    for r in scored:
        pred = "YES" if float(r.final_score) > threshold else "NO"
        if pred == r.gold_label:
            correct += 1
        if r.gold_label == "YES" and pred == "YES":
            yes_correct += 1
        if r.gold_label == "NO" and pred == "NO":
            no_correct += 1
    return {
        "threshold": threshold,
        "accuracy": safe_div(correct, len(scored)),
        "yes_accuracy": safe_div(yes_correct, yes_total),
        "no_accuracy": safe_div(no_correct, no_total),
    }


def compute_optimal_threshold_metrics(
    results: list[Result],
) -> dict[str, Optional[float]]:
    scored = [r for r in results if r.final_score is not None]
    if not scored:
        return {
            "accuracy_optimal_threshold": None,
            "yes_accuracy_optimal_threshold": None,
            "no_accuracy_optimal_threshold": None,
            "optimal_threshold": None,
        }

    candidates = sorted({0.0, 1.0, *[float(r.final_score) for r in scored]})
    best: Optional[dict[str, Optional[float]]] = None
    for thr in candidates:
        cur = threshold_metrics(scored, thr)
        if best is None:
            best = cur
            continue
        cur_acc = cur.get("accuracy")
        best_acc = best.get("accuracy")
        if cur_acc > best_acc:
            best = cur
    assert best is not None
    return {
        "accuracy_optimal_threshold": best.get("accuracy"),
        "yes_accuracy_optimal_threshold": best.get("yes_accuracy"),
        "no_accuracy_optimal_threshold": best.get("no_accuracy"),
        "optimal_threshold": best.get("threshold"),
    }


def extract_intermediate_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    labels = re.findall(r"\b(YES|NO|MAYBE)\b", str(value), flags=re.IGNORECASE)
    return labels[-1].upper() if labels else None


def compute_cave_consistency_for_result(
    result: Result,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Compute Cs-1, Cs-2, and combined consistency for Prompt-CAVE."""
    if result.method not in CAVE_CONSISTENCY_METHODS:
        return None, None, None

    final_label = result.pred_label if result.pred_label in {"YES", "NO"} else None
    if final_label is None:
        return 0, 0, 0

    if result.final_score is None:
        cs1 = 0
    elif final_label == "YES":
        # CAVE CS-1 treats a YES output as score-consistent when final score >= 0.5.
        cs1 = int(float(result.final_score) >= CAVE_CONSISTENCY_SCORE_THRESHOLD)
    else:
        # CAVE CS-1 treats a NO output as score-consistent when final score <= 0.5.
        cs1 = int(float(result.final_score) <= CAVE_CONSISTENCY_SCORE_THRESHOLD)

    obj = result.parsed_json if isinstance(result.parsed_json, dict) else None
    feature_labels: list[str] = []
    if obj is not None:
        for key, evidence in obj.items():
            if key not in {"final_score", "output"}:
                lab = extract_intermediate_label(evidence)
                if lab is not None:
                    feature_labels.append(lab)

    yes_count = sum(1 for lab in feature_labels if lab == "YES")
    no_count = sum(1 for lab in feature_labels if lab == "NO")
    maybe_count = sum(1 for lab in feature_labels if lab == "MAYBE")
    if final_label == "YES":
        cs2 = int((yes_count + maybe_count) > no_count)
    else:
        cs2 = int((no_count + maybe_count) > yes_count)

    return cs1, cs2, int(bool(cs1 and cs2))


def consistency_rows(results: list[Result]) -> list[ConsistencyRow]:
    rows: list[ConsistencyRow] = []
    for r in results:
        cs1, cs2, cons = compute_cave_consistency_for_result(r)
        if cs1 is None and cs2 is None and cons is None:
            continue
        rows.append(
            ConsistencyRow(
                dataset=r.dataset,
                row_id=r.row_id,
                method=r.method,
                gold_label=r.gold_label,
                pred_label=r.pred_label,
                final_score=r.final_score,
                cs1=cs1,
                cs2=cs2,
                consistency=cons,
            )
        )
    return rows


def compute_cave_consistency_metrics(
    results: list[Result],
) -> dict[str, Optional[float]]:
    rows = consistency_rows(results)
    if not rows:
        return {
            "consistency_cs1": None,
            "consistency_cs2": None,
            "consistency": None,
        }
    cs1_vals = [r.cs1 for r in rows if r.cs1 is not None]
    cs2_vals = [r.cs2 for r in rows if r.cs2 is not None]
    cons_vals = [r.consistency for r in rows if r.consistency is not None]
    return {
        "consistency_cs1": safe_div(sum(cs1_vals), len(cs1_vals)) if cs1_vals else None,
        "consistency_cs2": safe_div(sum(cs2_vals), len(cs2_vals)) if cs2_vals else None,
        "consistency": safe_div(sum(cons_vals), len(cons_vals)) if cons_vals else None,
    }


def compute_metrics(results: Iterable[Result]) -> dict[str, Any]:
    results = list(results)
    yes_rows = [r for r in results if r.gold_label == "YES"]
    no_rows = [r for r in results if r.gold_label == "NO"]

    pred_yes = [r for r in results if r.pred_label == "YES"]
    pred_no = [r for r in results if r.pred_label == "NO"]
    pred_maybe = [r for r in results if r.pred_label == "MAYBE"]
    valid_pred_rows = [r for r in results if r.pred_label in {"YES", "NO"}]
    parsed_label_rows = [r for r in results if r.pred_label in {"YES", "NO", "MAYBE"}]
    missing_label_rows = [
        r for r in results if r.pred_label not in {"YES", "NO", "MAYBE"}
    ]

    tp_yes = sum(1 for r in results if r.gold_label == "YES" and r.pred_label == "YES")
    fn_yes = sum(1 for r in results if r.gold_label == "YES" and r.pred_label != "YES")
    fp_yes = sum(1 for r in results if r.gold_label == "NO" and r.pred_label == "YES")

    tp_no = sum(1 for r in results if r.gold_label == "NO" and r.pred_label == "NO")
    fn_no = sum(1 for r in results if r.gold_label == "NO" and r.pred_label != "NO")
    fp_no = sum(1 for r in results if r.gold_label == "YES" and r.pred_label == "NO")

    correct = tp_yes + tp_no
    yes_acc = safe_div(tp_yes, len(yes_rows))
    no_acc = safe_div(tp_no, len(no_rows))

    yes_precision = safe_div(tp_yes, len(pred_yes))
    no_precision = safe_div(tp_no, len(pred_no))
    yes_f1 = f1_score(tp_yes, fp_yes, fn_yes)
    no_f1 = f1_score(tp_no, fp_no, fn_no)
    macro_f1 = (
        (yes_f1 + no_f1) / 2 if yes_f1 is not None and no_f1 is not None else None
    )

    scored_rows = [r for r in results if r.final_score is not None]
    scores = [float(r.final_score) for r in scored_rows]
    labels = [binary_gold(r.gold_label) for r in scored_rows]
    yes_scores = [float(r.final_score) for r in scored_rows if r.gold_label == "YES"]
    no_scores = [float(r.final_score) for r in scored_rows if r.gold_label == "NO"]

    metrics: dict[str, Any] = {
        "n_total": len(results),
        "n_with_prediction": len(valid_pred_rows),     # YES/NO only
        "n_parsed_any_label": len(parsed_label_rows),  # YES/NO/MAYBE
        "n_maybe": len(pred_maybe),
        "n_missing_label": len(missing_label_rows),
        "accuracy": safe_div(correct, len(results)),
        "macro_f1": macro_f1,
        "yes_accuracy": yes_acc,
        "no_accuracy": no_acc,
        "yes_precision": yes_precision,
        "no_precision": no_precision,
        "yes_f1": yes_f1,
        "no_f1": no_f1,
        "yes_pred_rate": safe_div(len(pred_yes), len(results)),
        "no_pred_rate": safe_div(len(pred_no), len(results)),
        "parsed_any_label_rate": safe_div(len(parsed_label_rows), len(results)),
        "maybe_rate": safe_div(len(pred_maybe), len(results)),
        "missing_label_rate": safe_div(len(missing_label_rows), len(results)),
        "n_scored": len(scored_rows),
        "mean_score": safe_div(sum(scores), len(scores)) if scores else None,
        "mean_score_gold_yes": (
            safe_div(sum(yes_scores), len(yes_scores)) if yes_scores else None
        ),
        "mean_score_gold_no": (
            safe_div(sum(no_scores), len(no_scores)) if no_scores else None
        ),
        "auroc": compute_auroc(labels, scores),
    }
    metrics.update(compute_optimal_threshold_metrics(results))
    metrics.update(compute_cave_consistency_metrics(results))
    return metrics
