#!/usr/bin/env python3
"""Significance analysis for PromptAV-family AV-bias experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import permutation_test as sp_permutation_test

from ..shared.io_utils import iter_jsonl
from .semantic_analysis import split_text_pair_from_prompt

try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None  # type: ignore

PROMPTAV_BASELINE = "promptav"
PROMPT_CAVE_BASELINE = "prompt_cave"

PROMPTAV_VARIANTS = [
    "promptav_framing_only",
    "promptav_contrastive_only",
    "promptav_framed_contrastive",
]
PROMPT_CAVE_VARIANTS = [
    "prompt_cave_framing_only",
    "prompt_cave_contrastive_only",
    "prompt_cave_framed_contrastive",
]

POS_LABEL = "YES"
NEG_LABEL = "NO"
LABELS = [POS_LABEL, NEG_LABEL]


def read_jsonl_files(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        try:
            for obj in iter_jsonl(path):
                obj["source_file"] = path.name
                rows.append(obj)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON in {path}: {e}") from e
    if not rows:
        raise ValueError("No rows loaded from input files.")
    return standardize_columns(pd.DataFrame(rows))


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["dataset", "row_id", "method", "gold_label", "pred_label", "model"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for c in ["dataset", "method", "gold_label", "pred_label", "model"]:
        out[c] = out[c].astype(str)
    out["gold_label"] = out["gold_label"].str.upper().str.strip()
    out["pred_label"] = out["pred_label"].str.upper().str.strip()
    out["is_valid_binary_pred"] = out["pred_label"].isin(LABELS)
    return out


def encode_labels(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Encode YES=0, NO=1, invalid prediction=2. Assumes y_true is YES/NO."""
    y = np.where(y_true == POS_LABEL, 0, 1).astype(np.int8)
    p = np.full(len(y_pred), 2, dtype=np.int8)
    p[y_pred == POS_LABEL] = 0
    p[y_pred == NEG_LABEL] = 1
    return y, p


def metric_from_encoded(y: np.ndarray, p: np.ndarray, metric_name: str) -> float:
    if len(y) == 0:
        return np.nan
    if metric_name == "accuracy":
        return float(np.mean(y == p))
    yes_mask = y == 0
    no_mask = y == 1
    if metric_name == "yes_accuracy":
        return float(np.mean(p[yes_mask] == 0)) if yes_mask.sum() else np.nan
    if metric_name == "no_accuracy":
        return float(np.mean(p[no_mask] == 1)) if no_mask.sum() else np.nan
    if metric_name == "macro_f1":
        f1s = []
        for c in (0, 1):
            tp = np.sum((y == c) & (p == c))
            fp = np.sum((y != c) & (p == c))
            fn = np.sum((y == c) & (p != c))
            denom = 2 * tp + fp + fn
            f1s.append(float(2 * tp / denom) if denom else 0.0)
        return float(np.mean(f1s))
    raise ValueError(f"Unknown metric: {metric_name}")


METRIC_NAMES = [
    "accuracy",
    "macro_f1",
    "yes_accuracy",
    "no_accuracy",
]


def holm_adjust(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return out
    _, out[valid], _, _ = multipletests(p[valid], method="holm")
    return out


def align_baseline_variant(
    df: pd.DataFrame,
    model: str,
    dataset: str,
    baseline: str,
    variant: str,
) -> pd.DataFrame:
    cols = ["model", "dataset", "row_id", "gold_label", "pred_label"]
    base = df[(df.model == model) & (df.dataset == dataset) & (df.method == baseline)][
        cols
    ].copy()
    var = df[(df.model == model) & (df.dataset == dataset) & (df.method == variant)][
        cols
    ].copy()
    if base.empty or var.empty:
        return pd.DataFrame()

    if base["row_id"].duplicated().any() or var["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row_ids for {model}/{dataset}/{variant}")

    base = base.rename(
        columns={"pred_label": "pred_base", "gold_label": "gold_label_base"}
    )
    var = var.rename(
        columns={"pred_label": "pred_variant", "gold_label": "gold_label_variant"}
    )
    merged = base.merge(
        var[["model", "dataset", "row_id", "gold_label_variant", "pred_variant"]],
        on=["model", "dataset", "row_id"],
        how="inner",
    )
    # An inner join drops unmatched ids without comment.
    if len(merged) != len(base) or len(merged) != len(var):
        raise ValueError(
            f"Join dropped rows for {model}/{dataset}/{variant}: "
            f"{len(base)} base, {len(var)} variant, {len(merged)} matched"
        )
    if merged.empty:
        return merged
    if not (
        merged["gold_label_base"].values == merged["gold_label_variant"].values
    ).all():
        raise ValueError(f"Gold label mismatch for {model}/{dataset}/{variant}")
    return (
        merged.rename(columns={"gold_label_base": "gold_label"})
        .drop(columns=["gold_label_variant"])
        .sort_values("row_id")
        .reset_index(drop=True)
    )


def mcnemar_from_correctness(
    base_correct: np.ndarray,
    variant_correct: np.ndarray,
    alternative: str = "greater",
) -> dict[str, float]:
    """Exact McNemar/binomial test on disagreements.

    b = baseline correct, variant wrong
    c = baseline wrong, variant correct
    For alternative='greater', p_one_sided tests c > b.
    """
    b = int(np.sum(base_correct & ~variant_correct))
    c = int(np.sum(~base_correct & variant_correct))
    n = b + c
    p_one = (
        1.0
        if n == 0
        else float(binomtest(c, n=n, p=0.5, alternative=alternative).pvalue)
    )
    return {
        "mcnemar_b_base_correct_variant_wrong": b,
        "mcnemar_c_base_wrong_variant_correct": c,
        "mcnemar_n_discordant": n,
        "mcnemar_p_one_sided_variant_better": p_one,
    }


def _permutation_pvalue(
    y: np.ndarray,
    p_base: np.ndarray,
    p_variant: np.ndarray,
    metric_name: str,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """One-sided permutation p-value for metric(variant) - metric(base) > 0."""
    if len(p_base) < 2:
        return np.nan

    def statistic(pb, pv):
        return metric_from_encoded(y, pv, metric_name) - metric_from_encoded(
            y, pb, metric_name
        )

    try:
        result = sp_permutation_test(
            (p_base, p_variant),
            statistic=statistic,
            permutation_type="samples",
            alternative="greater",
            n_resamples=n_perm,
            vectorized=False,
            random_state=rng,
        )
        return float(result.pvalue)
    except Exception as exc:
        raise RuntimeError(
            f"permutation test failed for {metric_name} "
            f"(n={len(p_base)}, n_perm={n_perm}): {exc}"
        ) from exc


def paired_cell_test(
    aligned: pd.DataFrame,
    model: str,
    dataset: str,
    baseline: str,
    variant: str,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float | str | int]:
    y = aligned["gold_label"].to_numpy(dtype=object)
    pred_b = aligned["pred_base"].to_numpy(dtype=object)
    pred_v = aligned["pred_variant"].to_numpy(dtype=object)
    correct_b = pred_b == y
    correct_v = pred_v == y
    yes_mask = y == POS_LABEL

    row: dict[str, float | str | int] = {
        "model": model,
        "dataset": dataset,
        "baseline": baseline,
        "variant": variant,
        "n": int(len(aligned)),
        "n_yes": int(yes_mask.sum()),
        "n_no": int((y == NEG_LABEL).sum()),
    }

    y_enc, p_b_enc = encode_labels(y, pred_b)
    _, p_v_enc = encode_labels(y, pred_v)
    for name in METRIC_NAMES:
        base_val = metric_from_encoded(y_enc, p_b_enc, name)
        var_val = metric_from_encoded(y_enc, p_v_enc, name)
        row[f"base_{name}"] = base_val
        row[f"variant_{name}"] = var_val
        row[f"delta_{name}"] = var_val - base_val

        if name == "macro_f1":
            row["perm_macro_f1_p_one_sided"] = _permutation_pvalue(
                y_enc, p_b_enc, p_v_enc, name, n_perm, rng
            )

    m = mcnemar_from_correctness(correct_b, correct_v)
    for k, v in m.items():
        row[f"accuracy_{k}"] = v

    if yes_mask.sum() > 0:
        m_yes = mcnemar_from_correctness(correct_b[yes_mask], correct_v[yes_mask])
        for k, v in m_yes.items():
            row[f"yes_accuracy_{k}"] = v
    else:
        for k in [
            "mcnemar_b_base_correct_variant_wrong",
            "mcnemar_c_base_wrong_variant_correct",
            "mcnemar_n_discordant",
            "mcnemar_p_two_sided",
            "mcnemar_p_one_sided_variant_better",
        ]:
            row[f"yes_accuracy_{k}"] = np.nan

    return row


def run_cell_tests(
    df: pd.DataFrame,
    baseline: str,
    variants: list[str],
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model in sorted(df["model"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            for variant in variants:
                aligned = align_baseline_variant(df, model, dataset, baseline, variant)
                if aligned.empty:
                    continue
                rows.append(
                    paired_cell_test(
                        aligned, model, dataset, baseline, variant, n_perm, rng
                    )
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    p_cols = [
        "accuracy_mcnemar_p_one_sided_variant_better",
        "yes_accuracy_mcnemar_p_one_sided_variant_better",
        "perm_macro_f1_p_one_sided",
    ]
    # Holm family = the model-dataset settings for a single variant-vs-baseline
    # comparison, matching how Table 8 reports counts (one row per variant, k of
    # N settings). Correcting over the pooled table instead would fold the other
    # variants' tests into each row's adjustment, since Holm is step-down.
    for col in p_cols:
        if col in out.columns:
            out[f"{col}_holm"] = np.nan
            for _, idx in out.groupby("variant").groups.items():
                out.loc[idx, f"{col}_holm"] = holm_adjust(
                    out.loc[idx, col].to_numpy()
                )
    return out


def summarize_main_effects(main_cells: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for metric in METRIC_NAMES:
        d = main_cells[f"delta_{metric}"].dropna().to_numpy(dtype=float)
        n = len(d)
        k = int(np.sum(d > 0))
        rows.append(
            {
                "variant": variant,
                "metric": metric,
                "n_settings": n,
                "n_improved": k,
                "pct_improved": k / n if n else np.nan,
                "mean_delta": float(np.mean(d)) if n else np.nan,
                "median_delta": float(np.median(d)) if n else np.nan,
                "min_delta": float(np.min(d)) if n else np.nan,
                "max_delta": float(np.max(d)) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_significance_counts(cells: pd.DataFrame) -> pd.DataFrame:
    # (label, p-value column, delta column n_improved is counted from)
    tests = [
        (
            "accuracy McNemar",
            "accuracy_mcnemar_p_one_sided_variant_better",
            "accuracy",
        ),
        (
            "YES-only McNemar",
            "yes_accuracy_mcnemar_p_one_sided_variant_better",
            "yes_accuracy",
        ),
        ("permutation Δ macro-F1", "perm_macro_f1_p_one_sided", "macro_f1"),
    ]
    rows = []
    for variant, sub in cells.groupby("variant"):
        for label, p_col, metric in tests:
            if p_col not in sub.columns:
                continue
            p_raw = sub[p_col]
            p_holm = (
                sub[f"{p_col}_holm"]
                if f"{p_col}_holm" in sub.columns
                else pd.Series(np.nan, index=sub.index)
            )
            delta_col = f"delta_{metric}"
            if delta_col not in sub.columns:
                raise ValueError(
                    f"{label}: {p_col} is present but {delta_col} is missing, "
                    f"so n_improved cannot be computed. Pass the cells frame "
                    f"from run_cell_tests, which always emits it."
                )
            improved = sub[delta_col] > 0
            rows.append(
                {
                    "variant": variant,
                    "test_group": label,
                    "n_tests": int(p_raw.notna().sum()),
                    "n_improved": int(improved.sum()),
                    "raw_p_lt_0.05": int((p_raw < 0.05).sum()),
                    "raw_p_lt_0.01": int((p_raw < 0.01).sum()),
                    "holm_p_lt_0.05": int((p_holm < 0.05).sum()),
                    "holm_p_lt_0.01": int((p_holm < 0.01).sum()),
                }
            )
    return pd.DataFrame(rows)


def extract_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Add text1/text2 columns by parsing the prompt field; drops rows where texts can't be parsed."""

    def _try_split(p):
        try:
            return split_text_pair_from_prompt(str(p))
        except ValueError:
            return None

    parsed = df["prompt"].apply(_try_split)
    valid = parsed.notna()
    n_skipped = int((~valid).sum())
    if n_skipped:
        print(
            f"  [warn] skipped {n_skipped} rows where Text 1/2 could not be parsed",
            file=sys.stderr,
        )
    out = df[valid].copy().reset_index(drop=True)
    out["text1"] = parsed[valid].apply(lambda t: t[0]).values
    out["text2"] = parsed[valid].apply(lambda t: t[1]).values
    return out


def fit_false_no_logit(
    sub: pd.DataFrame,
    extra_covariates: list[str] | None = None,
) -> dict | None:
    """
    Fit logit(P(false-NO)) ~ semantic_distance_z + C(dataset) + C(model) [+ extras]
    on same-author pairs (gold_label==YES), with SEs clustered by text pair.

    Clustering is necessary because the same (dataset, row_id) pair is scored by
    multiple models, so rows sharing a pair are not independent.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("Install statsmodels: pip install statsmodels") from exc

    sub = sub.dropna(subset=["semantic_cosine_distance"]).reset_index(drop=True)
    if len(sub) < 10:
        return None

    y = (
        ((sub["gold_label"] == POS_LABEL) & (sub["pred_label"] == NEG_LABEL))
        .astype(int)
        .to_numpy()
    )

    # Cluster IDs = same text pair evaluated by multiple models shares a cluster
    cluster_ids = (sub["dataset"] + "||" + sub["row_id"].astype(str)).to_numpy()

    # Standardize semantic distance for interpretable coefficient scale
    sd = sub["semantic_cosine_distance"].to_numpy(dtype=float)
    sd_std = sd.std()
    sd_z = (sd - sd.mean()) / (sd_std if sd_std > 0 else 1.0)

    # Dataset and model fixed effects
    dataset_dummies = pd.get_dummies(
        sub["dataset"], prefix="dataset", drop_first=True
    ).astype(float)
    model_dummies = pd.get_dummies(
        sub["model"], prefix="model", drop_first=True
    ).astype(float)

    X = pd.DataFrame({"semantic_distance_z": sd_z})
    X = pd.concat([X, dataset_dummies, model_dummies], axis=1)

    if extra_covariates:
        for cov in extra_covariates:
            if cov in sub.columns:
                cov_vals = sub[cov].to_numpy(dtype=float)
                cov_s = cov_vals.std()
                X[f"{cov}_z"] = (cov_vals - cov_vals.mean()) / (
                    cov_s if cov_s > 0 else 1.0
                )

    X = sm.add_constant(X)

    try:
        result = sm.Logit(y, X).fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": cluster_ids},
        )
    except Exception as exc:
        print(f"  [warn] logistic regression failed: {exc}", file=sys.stderr)
        return None

    ci = result.conf_int()
    coef_rows = []
    for term in result.params.index:
        coef_rows.append(
            {
                "term": term,
                "coef": float(result.params[term]),
                "std_err": float(result.bse[term]),
                "z": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "ci_low_95": float(ci.loc[term, 0]),
                "ci_high_95": float(ci.loc[term, 1]),
            }
        )
    return {
        "n_obs": int(result.nobs),
        "n_false_no": int(y.sum()),
        "pseudo_r2_mcfadden": float(result.prsquared),
        "aic": float(result.aic),
        "coefs": coef_rows,
    }