import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.significance import (
    _permutation_pvalue,
    align_baseline_variant,
    encode_labels,
    holm_adjust,
    mcnemar_from_correctness,
    metric_from_encoded,
    paired_cell_test,
    read_jsonl_files,
    run_cell_tests,
    standardize_columns,
    summarize_main_effects,
    summarize_significance_counts,
)


def test_encode_labels_yes_no():
    y_true = np.array(["YES", "NO", "YES"], dtype=object)
    y_pred = np.array(["YES", "NO", "NO"], dtype=object)
    y, p = encode_labels(y_true, y_pred)
    assert list(y) == [0, 1, 0]
    assert list(p) == [0, 1, 1]


def test_encode_labels_invalid_pred():
    y_true = np.array(["YES", "NO"], dtype=object)
    y_pred = np.array(["MAYBE", ""], dtype=object)
    _, p = encode_labels(y_true, y_pred)
    assert list(p) == [2, 2]


def test_encode_labels_invalid_counts_as_wrong():
    y_true = np.array(["YES"], dtype=object)
    y_pred = np.array(["INVALID"], dtype=object)
    y, p = encode_labels(y_true, y_pred)
    assert metric_from_encoded(y, p, "accuracy") == 0.0


def test_metric_accuracy_perfect():
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    p = np.array([0, 1, 0, 1], dtype=np.int8)
    assert metric_from_encoded(y, p, "accuracy") == 1.0


def test_metric_accuracy_zero():
    y = np.array([0, 1], dtype=np.int8)
    p = np.array([1, 0], dtype=np.int8)
    assert metric_from_encoded(y, p, "accuracy") == 0.0


def test_metric_yes_accuracy():
    y = np.array([0, 0, 1], dtype=np.int8)
    p = np.array([0, 1, 1], dtype=np.int8)
    assert metric_from_encoded(y, p, "yes_accuracy") == pytest.approx(0.5)


def test_metric_yes_accuracy_no_yes_examples():
    y = np.array([1, 1], dtype=np.int8)
    p = np.array([1, 1], dtype=np.int8)
    assert np.isnan(metric_from_encoded(y, p, "yes_accuracy"))


def test_metric_no_accuracy():
    y = np.array([0, 1, 1], dtype=np.int8)
    p = np.array([0, 1, 0], dtype=np.int8)
    assert metric_from_encoded(y, p, "no_accuracy") == pytest.approx(0.5)


def test_metric_macro_f1_perfect():
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    p = np.array([0, 0, 1, 1], dtype=np.int8)
    assert metric_from_encoded(y, p, "macro_f1") == pytest.approx(1.0)


def test_metric_macro_f1_all_wrong():
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    p = np.array([1, 1, 0, 0], dtype=np.int8)
    assert metric_from_encoded(y, p, "macro_f1") == pytest.approx(0.0)


def test_metric_unknown_raises():
    y = np.array([0, 1], dtype=np.int8)
    p = np.array([0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="Unknown metric"):
        metric_from_encoded(y, p, "nonexistent")


def test_metric_empty_returns_nan():
    y = np.array([], dtype=np.int8)
    p = np.array([], dtype=np.int8)
    assert np.isnan(metric_from_encoded(y, p, "accuracy"))


def test_mcnemar_no_discordants():
    base = np.array([True, True, False])
    variant = np.array([True, True, False])
    result = mcnemar_from_correctness(base, variant)
    assert result["mcnemar_n_discordant"] == 0
    assert result["mcnemar_p_one_sided_variant_better"] == 1.0


def test_mcnemar_variant_clearly_better():
    base = np.array([False] * 20 + [True] * 10)
    variant = np.array([True] * 20 + [True] * 10)
    result = mcnemar_from_correctness(base, variant)
    assert result["mcnemar_c_base_wrong_variant_correct"] == 20
    assert result["mcnemar_b_base_correct_variant_wrong"] == 0
    assert result["mcnemar_p_one_sided_variant_better"] < 0.05


def test_mcnemar_equal_discordants():
    base = np.array([True, False])
    variant = np.array([False, True])
    result = mcnemar_from_correctness(base, variant)
    assert result["mcnemar_n_discordant"] == 2
    assert result["mcnemar_p_one_sided_variant_better"] == pytest.approx(0.75)


def test_mcnemar_returns_expected_keys():
    base = np.array([True, False])
    variant = np.array([True, True])
    result = mcnemar_from_correctness(base, variant)
    assert "mcnemar_p_one_sided_variant_better" in result
    assert "mcnemar_n_discordant" in result
    assert "mcnemar_p_two_sided" not in result


def _make_df(**kwargs):
    defaults = {
        "dataset": ["imdb62"],
        "row_id": [0],
        "method": ["promptav"],
        "gold_label": ["yes"],
        "pred_label": ["YES"],
        "model": ["m"],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def test_standardize_columns_uppercases_gold_label():
    df = standardize_columns(_make_df(gold_label=["yes"]))
    assert df["gold_label"].iloc[0] == "YES"


def test_standardize_columns_valid_pred_flag():
    df = standardize_columns(_make_df(pred_label=["YES"]))
    assert df["is_valid_binary_pred"].iloc[0] == True


def test_standardize_columns_invalid_pred_flag():
    df = standardize_columns(_make_df(pred_label=["MAYBE"]))
    assert df["is_valid_binary_pred"].iloc[0] == False


def test_standardize_columns_missing_column_raises():
    df = pd.DataFrame({"dataset": ["d"], "row_id": [0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        standardize_columns(df)


def _make_alignment_df():
    return pd.DataFrame(
        {
            "model": ["m"] * 6,
            "dataset": ["d"] * 6,
            "row_id": [0, 1, 2, 0, 1, 2],
            "method": ["base"] * 3 + ["var"] * 3,
            "gold_label": ["YES", "NO", "YES", "YES", "NO", "YES"],
            "pred_label": ["YES", "NO", "NO", "YES", "YES", "YES"],
        }
    )


def test_align_baseline_variant_basic():
    df = _make_alignment_df()
    aligned = align_baseline_variant(df, "m", "d", "base", "var")
    assert len(aligned) == 3
    assert set(aligned.columns) >= {"pred_base", "pred_variant", "gold_label"}


def test_align_baseline_variant_empty_when_missing():
    df = _make_alignment_df()
    aligned = align_baseline_variant(df, "m", "d", "base", "nonexistent")
    assert aligned.empty


def test_align_baseline_variant_gold_label_mismatch_raises():
    df = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "dataset": ["d"] * 4,
            "row_id": [0, 1, 0, 1],
            "method": ["base", "base", "var", "var"],
            "gold_label": ["YES", "NO", "NO", "NO"],  # row 0 mismatch
            "pred_label": ["YES", "NO", "YES", "NO"],
        }
    )
    with pytest.raises(ValueError, match="Gold label mismatch"):
        align_baseline_variant(df, "m", "d", "base", "var")


def test_holm_adjust_preserves_nan():
    p = [0.01, np.nan, 0.05]
    out = holm_adjust(p)
    assert np.isnan(out[1])
    assert not np.isnan(out[0])


def test_holm_adjust_all_nan():
    out = holm_adjust([np.nan, np.nan])
    assert all(np.isnan(out))


def test_holm_adjust_increases_pvalues():
    p = [0.01, 0.02, 0.03]
    out = holm_adjust(p)
    for raw, adj in zip(p, out):
        assert adj >= raw


def _make_encoded(n_yes=30, n_no=20, base_yes_correct=20, variant_yes_correct=25):
    y = np.array([0] * n_yes + [1] * n_no, dtype=np.int8)
    p_base = np.array(
        [0] * base_yes_correct + [1] * (n_yes - base_yes_correct) + [1] * n_no,
        dtype=np.int8,
    )
    p_variant = np.array(
        [0] * variant_yes_correct + [1] * (n_yes - variant_yes_correct) + [1] * n_no,
        dtype=np.int8,
    )
    return y, p_base, p_variant


def test_permutation_pvalue_too_few_obs():
    p = _permutation_pvalue(
        np.array([0], dtype=np.int8),
        np.array([0], dtype=np.int8),
        np.array([0], dtype=np.int8),
        "macro_f1",
        99,
        np.random.default_rng(0),
    )
    assert np.isnan(p)


def test_permutation_pvalue_in_range():
    y, pb, pv = _make_encoded()
    p = _permutation_pvalue(y, pb, pv, "macro_f1", 199, np.random.default_rng(3))
    assert 0.0 <= p <= 1.0


def test_permutation_pvalue_significant_when_large_effect():
    y, pb, pv = _make_encoded(base_yes_correct=5, variant_yes_correct=29)
    p = _permutation_pvalue(y, pb, pv, "macro_f1", 999, np.random.default_rng(4))
    assert p < 0.05


def _make_aligned_df(n=40):
    y = ["YES"] * (n // 2) + ["NO"] * (n // 2)
    pb = ["YES"] * (n // 4) + ["NO"] * (n // 4) + ["YES"] * 2 + ["NO"] * (n // 2 - 2)
    pv = (
        ["YES"] * (n // 4 + 4)
        + ["NO"] * (n // 4 - 4)
        + ["YES"] * 2
        + ["NO"] * (n // 2 - 2)
    )
    return pd.DataFrame(
        {"gold_label": y, "pred_base": pb, "pred_variant": pv, "row_id": range(n)}
    )


def test_paired_cell_test_output_keys():
    aligned = _make_aligned_df()
    row = paired_cell_test(
        aligned,
        "m",
        "d",
        "base",
        "var",
        n_perm=49,
        rng=np.random.default_rng(5),
    )
    assert "perm_macro_f1_p_one_sided" in row
    assert "accuracy_mcnemar_p_one_sided_variant_better" in row
    assert "yes_accuracy_mcnemar_p_one_sided_variant_better" in row


def test_paired_cell_test_no_for_no_accuracy():
    aligned = _make_aligned_df()
    row = paired_cell_test(
        aligned,
        "m",
        "d",
        "base",
        "var",
        n_perm=49,
        rng=np.random.default_rng(6),
    )
    assert "perm_no_accuracy_p_one_sided" not in row


def test_paired_cell_test_delta_correct():
    aligned = _make_aligned_df()
    row = paired_cell_test(
        aligned,
        "m",
        "d",
        "base",
        "var",
        n_perm=49,
        rng=np.random.default_rng(7),
    )
    assert row["delta_accuracy"] == pytest.approx(
        row["variant_accuracy"] - row["base_accuracy"]
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _minimal_row(**kwargs) -> dict:
    defaults = {
        "dataset": "imdb62",
        "row_id": 0,
        "method": "promptav",
        "gold_label": "YES",
        "pred_label": "YES",
        "model": "test-model",
    }
    defaults.update(kwargs)
    return defaults


def test_read_jsonl_files_basic(tmp_path):
    p = tmp_path / "results.jsonl"
    _write_jsonl(p, [_minimal_row(row_id=0), _minimal_row(row_id=1)])
    df = read_jsonl_files([p])
    assert len(df) == 2
    assert set(df.columns) >= {
        "dataset",
        "row_id",
        "method",
        "gold_label",
        "pred_label",
        "model",
    }


def test_read_jsonl_files_uppercases_gold_label(tmp_path):
    p = tmp_path / "results.jsonl"
    _write_jsonl(p, [_minimal_row(gold_label="yes")])
    df = read_jsonl_files([p])
    assert df["gold_label"].iloc[0] == "YES"


def test_read_jsonl_files_multiple_files(tmp_path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    _write_jsonl(p1, [_minimal_row(row_id=0, model="m1")])
    _write_jsonl(p2, [_minimal_row(row_id=1, model="m2")])
    df = read_jsonl_files([p1, p2])
    assert len(df) == 2
    assert set(df["model"]) == {"m1", "m2"}


def test_read_jsonl_files_empty_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="No rows loaded"):
        read_jsonl_files([p])


def test_read_jsonl_files_bad_json_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not valid json}\n")
    with pytest.raises(ValueError, match="Could not parse JSON"):
        read_jsonl_files([p])


def test_read_jsonl_files_adds_source_file(tmp_path):
    p = tmp_path / "myrun.jsonl"
    _write_jsonl(p, [_minimal_row()])
    df = read_jsonl_files([p])
    assert df["source_file"].iloc[0] == "myrun.jsonl"


def _make_cells_df(deltas: list[float], variant: str = "v") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": [variant] * len(deltas),
            "delta_accuracy": deltas,
            "delta_macro_f1": deltas,
            "delta_yes_accuracy": deltas,
            "delta_no_accuracy": deltas,
        }
    )


def test_summarize_main_effects_one_row_per_metric():
    cells = _make_cells_df([0.05, -0.02, 0.10])
    result = summarize_main_effects(cells, "v")
    assert set(result["metric"]) == {
        "accuracy",
        "macro_f1",
        "yes_accuracy",
        "no_accuracy",
    }
    assert len(result) == 4


def test_summarize_main_effects_n_improved():
    cells = _make_cells_df([0.05, -0.02, 0.10])
    result = summarize_main_effects(cells, "v")
    row = result[result["metric"] == "accuracy"].iloc[0]
    assert row["n_improved"] == 2
    assert row["n_settings"] == 3
    assert row["pct_improved"] == pytest.approx(2 / 3)


def test_summarize_main_effects_mean_delta():
    deltas = [0.10, 0.20]
    cells = _make_cells_df(deltas)
    result = summarize_main_effects(cells, "v")
    row = result[result["metric"] == "accuracy"].iloc[0]
    assert row["mean_delta"] == pytest.approx(0.15)


def test_summarize_main_effects_empty_delta_nan():
    cells = pd.DataFrame(
        {
            "variant": ["v"],
            "delta_accuracy": [np.nan],
            "delta_macro_f1": [np.nan],
            "delta_yes_accuracy": [np.nan],
            "delta_no_accuracy": [np.nan],
        }
    )
    result = summarize_main_effects(cells, "v")
    row = result[result["metric"] == "accuracy"].iloc[0]
    assert row["n_settings"] == 0
    assert np.isnan(row["mean_delta"])


def _make_sig_cells(
    variant: str = "v",
    p_mcnemar: float = 0.03,
    p_perm: float = 0.04,
    delta_accuracy: float = 0.05,
    delta_macro_f1: float = 0.05,
    delta_yes_accuracy: float = 0.05,
    n: int = 3,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": [variant] * n,
            "accuracy_mcnemar_p_one_sided_variant_better": [p_mcnemar] * n,
            "yes_accuracy_mcnemar_p_one_sided_variant_better": [p_mcnemar] * n,
            "perm_macro_f1_p_one_sided": [p_perm] * n,
            "perm_macro_f1_p_one_sided_holm": [p_perm * 3] * n,
            "delta_accuracy": [delta_accuracy] * n,
            "delta_macro_f1": [delta_macro_f1] * n,
            "delta_yes_accuracy": [delta_yes_accuracy] * n,
        }
    )


def test_summarize_significance_counts_columns():
    cells = _make_sig_cells()
    result = summarize_significance_counts(cells)
    assert "raw_p_lt_0.05" in result.columns
    assert "raw_p_lt_0.01" in result.columns
    assert "holm_p_lt_0.05" in result.columns
    assert "holm_p_lt_0.01" in result.columns


def test_summarize_significance_counts_one_row_per_test_group():
    cells = _make_sig_cells()
    result = summarize_significance_counts(cells)
    assert set(result["test_group"]) == {
        "accuracy McNemar",
        "YES-only McNemar",
        "permutation Δ macro-F1",
    }


def test_summarize_significance_counts_p_lt_0_05():
    cells = _make_sig_cells(p_mcnemar=0.03, delta_accuracy=0.05, n=3)
    result = summarize_significance_counts(cells)
    row = result[result["test_group"] == "accuracy McNemar"].iloc[0]
    assert row["raw_p_lt_0.05"] == 3


def test_summarize_significance_counts_ignores_direction():
    # The p-threshold columns count significance alone; direction is reported
    # separately by n_improved. (A real one-sided McNemar cannot pair p=0.03
    # with a negative delta — this frame is built by hand to prove the columns
    # are independent.)
    cells = _make_sig_cells(p_mcnemar=0.03, delta_accuracy=-0.05, n=3)
    result = summarize_significance_counts(cells)
    row = result[result["test_group"] == "accuracy McNemar"].iloc[0]
    assert row["raw_p_lt_0.05"] == 3
    assert row["n_improved"] == 0


def test_summarize_significance_counts_p_lt_0_01_threshold():
    cells = _make_sig_cells(p_perm=0.005, delta_macro_f1=0.05, n=2)
    result = summarize_significance_counts(cells)
    row = result[result["test_group"] == "permutation Δ macro-F1"].iloc[0]
    assert row["raw_p_lt_0.01"] == 2
    assert row["raw_p_lt_0.05"] == 2


def test_summarize_significance_counts_missing_col_skipped():
    cells = pd.DataFrame(
        {
            "variant": ["v"],
            "accuracy_mcnemar_p_one_sided_variant_better": [0.03],
            "delta_accuracy": [0.05],
        }
    )
    result = summarize_significance_counts(cells)
    assert len(result) == 1
    assert result["test_group"].iloc[0] == "accuracy McNemar"


def _make_run_df(n_per_method: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for method in ["base", "variant_a"]:
        for i in range(n_per_method):
            gold = "YES" if i < n_per_method // 2 else "NO"
            pred = rng.choice(["YES", "NO"])
            rows.append(
                {
                    "model": "m1",
                    "dataset": "d1",
                    "row_id": i,
                    "method": method,
                    "gold_label": gold,
                    "pred_label": pred,
                    "is_valid_binary_pred": True,
                }
            )
    return pd.DataFrame(rows)


def test_run_cell_tests_returns_dataframe():
    df = _make_run_df()
    out = run_cell_tests(df, "base", ["variant_a"], n_perm=49, seed=0)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1


def test_run_cell_tests_has_holm_columns():
    df = _make_run_df()
    out = run_cell_tests(df, "base", ["variant_a"], n_perm=49, seed=0)
    assert "accuracy_mcnemar_p_one_sided_variant_better_holm" in out.columns
    assert "perm_macro_f1_p_one_sided_holm" in out.columns


def test_run_cell_tests_empty_when_no_overlap():
    df = _make_run_df()
    out = run_cell_tests(df, "base", ["nonexistent"], n_perm=49, seed=0)
    assert out.empty


def test_run_cell_tests_multiple_variants():
    rng = np.random.default_rng(1)
    rows = []
    for method in ["base", "var_a", "var_b"]:
        for i in range(20):
            rows.append(
                {
                    "model": "m1",
                    "dataset": "d1",
                    "row_id": i,
                    "method": method,
                    "gold_label": "YES" if i < 10 else "NO",
                    "pred_label": rng.choice(["YES", "NO"]),
                    "is_valid_binary_pred": True,
                }
            )
    df = pd.DataFrame(rows)
    out = run_cell_tests(df, "base", ["var_a", "var_b"], n_perm=49, seed=1)
    assert len(out) == 2
    assert set(out["variant"]) == {"var_a", "var_b"}
