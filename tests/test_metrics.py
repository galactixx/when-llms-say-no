from src.shared.constants import CAVE_FEATURE_KEYS
from src.shared.metrics import (
    compute_cave_consistency_for_result,
    compute_cave_consistency_metrics,
    compute_metrics,
    compute_optimal_threshold_metrics,
    consistency_rows,
    extract_intermediate_label,
    f1_score,
    manual_auroc,
    safe_div,
    threshold_metrics,
)
from src.shared.models import Result


def make_result(gold, pred, score=None, method="promptav", parsed_json=None):
    return Result(
        dataset="imdb62",
        row_id=0,
        method=method,
        gold_label=gold,
        pred_label=pred,
        correct=(pred == gold if pred in {"YES", "NO"} else None),
        final_score=score,
        prompt="",
        raw_response="",
        parsed_json=parsed_json,
        backend="vllm",
        model="test",
    )


def test_safe_div_normal():
    assert safe_div(3.0, 4.0) == 0.75


def test_safe_div_zero_denominator():
    assert safe_div(1.0, 0.0) is None


def test_f1_score_perfect():
    assert f1_score(tp=4, fp=0, fn=0) == 1.0


def test_f1_score_zero():
    assert f1_score(tp=0, fp=2, fn=2) == 0.0


def test_f1_score_formula():
    # 2*2 / (2*2 + 1 + 1) = 4/6
    assert abs(f1_score(tp=2, fp=1, fn=1) - 4 / 6) < 1e-9


def test_f1_score_no_predictions():
    assert f1_score(tp=0, fp=0, fn=0) is None


def test_manual_auroc_perfect():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.2]
    assert manual_auroc(labels, scores) == 1.0


def test_manual_auroc_worst():
    labels = [1, 1, 0, 0]
    scores = [0.2, 0.3, 0.8, 0.9]
    assert manual_auroc(labels, scores) == 0.0


def test_manual_auroc_tied_scores():
    labels = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert manual_auroc(labels, scores) == 0.5


def test_manual_auroc_single_class_returns_none():
    assert manual_auroc([1, 1, 1], [0.9, 0.8, 0.7]) is None


def test_manual_auroc_empty_returns_none():
    assert manual_auroc([], []) is None


def test_threshold_metrics_basic():
    results = [
        make_result("YES", "YES", score=0.7),
        make_result("YES", "NO", score=0.3),
        make_result("NO", "NO", score=0.2),
        make_result("NO", "YES", score=0.6),
    ]
    m = threshold_metrics(results, 0.5)
    assert m["accuracy"] == 0.5
    assert m["yes_accuracy"] == 0.5
    assert m["no_accuracy"] == 0.5


def test_threshold_metrics_no_scored_rows():
    results = [make_result("YES", "YES", score=None)]
    m = threshold_metrics(results, 0.5)
    assert m["accuracy"] is None


def test_compute_optimal_threshold_finds_best():
    # All YES examples have score 0.9, all NO have score 0.1 — perfect at any threshold between
    results = [
        make_result("YES", "YES", score=0.9),
        make_result("YES", "YES", score=0.8),
        make_result("NO", "NO", score=0.1),
        make_result("NO", "NO", score=0.2),
    ]
    m = compute_optimal_threshold_metrics(results)
    assert m["accuracy_optimal_threshold"] == 1.0


def test_extract_intermediate_label_yes():
    assert extract_intermediate_label("The styles are similar. YES") == "YES"


def test_extract_intermediate_label_maybe():
    assert extract_intermediate_label("Hard to tell. MAYBE") == "MAYBE"


def test_extract_intermediate_label_last_wins():
    assert extract_intermediate_label("First YES then NO") == "NO"


def test_extract_intermediate_label_none():
    assert extract_intermediate_label("the styles are completely different") is None


def test_extract_intermediate_label_null():
    assert extract_intermediate_label(None) is None


def _cave_result(pred_label, score, feature_label="YES"):
    parsed_json = {k: f"Similar. {feature_label}" for k in CAVE_FEATURE_KEYS}
    return make_result(
        "YES", pred_label, score=score, method="prompt_cave", parsed_json=parsed_json
    )


def test_cave_cs1_yes_high_score():
    cs1, _, _ = compute_cave_consistency_for_result(_cave_result("YES", 0.8))
    assert cs1 == 1


def test_cave_cs1_yes_low_score():
    cs1, _, _ = compute_cave_consistency_for_result(_cave_result("YES", 0.3))
    assert cs1 == 0


def test_cave_cs1_no_low_score():
    r = make_result(
        "NO",
        "NO",
        score=0.2,
        method="prompt_cave",
        parsed_json={k: "Different. NO" for k in CAVE_FEATURE_KEYS},
    )
    cs1, _, _ = compute_cave_consistency_for_result(r)
    assert cs1 == 1


def test_cave_cs2_majority_match():
    _, cs2, _ = compute_cave_consistency_for_result(_cave_result("YES", 0.8, "YES"))
    assert cs2 == 1


def test_cave_cs2_majority_mismatch():
    _, cs2, _ = compute_cave_consistency_for_result(_cave_result("YES", 0.8, "NO"))
    assert cs2 == 0


def test_cave_consistency_non_cave_method_returns_none():
    r = make_result("YES", "YES", score=0.8, method="promptav")
    assert compute_cave_consistency_for_result(r) == (None, None, None)


def test_cave_consistency_missing_pred_label():
    r = make_result("YES", None, score=0.8, method="prompt_cave")
    assert compute_cave_consistency_for_result(r) == (0, 0, 0)


def test_compute_metrics_basic():
    results = [
        make_result("YES", "YES", score=0.8),
        make_result("YES", "NO", score=0.3),
        make_result("NO", "NO", score=0.2),
        make_result("NO", "YES", score=0.6),
    ]
    m = compute_metrics(results)
    assert m["n_total"] == 4
    assert m["accuracy"] == 0.5
    assert m["yes_accuracy"] == 0.5
    assert m["no_accuracy"] == 0.5


def test_compute_metrics_all_correct():
    results = [
        make_result("YES", "YES", score=0.9),
        make_result("NO", "NO", score=0.1),
    ]
    m = compute_metrics(results)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_compute_metrics_missing_labels_counted():
    results = [
        make_result("YES", None),
        make_result("NO", None),
    ]
    m = compute_metrics(results)
    assert m["accuracy"] == 0.0
    assert m["n_missing_label"] == 2


def _full_cave_result(gold, pred, score, feature_label="YES"):
    parsed_json = {k: f"Similar. {feature_label}" for k in CAVE_FEATURE_KEYS}
    return make_result(
        gold, pred, score=score, method="prompt_cave", parsed_json=parsed_json
    )


def test_consistency_rows_filters_non_cave():
    results = [
        make_result("YES", "YES", score=0.8, method="promptav"),
        _full_cave_result("YES", "YES", score=0.8),
    ]
    rows = consistency_rows(results)
    assert len(rows) == 1
    assert rows[0].method == "prompt_cave"


def test_consistency_rows_fields():
    r = _full_cave_result("YES", "YES", 0.9)
    rows = consistency_rows([r])
    assert len(rows) == 1
    row = rows[0]
    assert row.cs1 == 1
    assert row.cs2 == 1
    assert row.consistency == 1


def test_consistency_rows_empty_input():
    assert consistency_rows([]) == []


def test_compute_cave_consistency_metrics_aggregates():
    results = [
        _full_cave_result("YES", "YES", 0.9, "YES"),
        _full_cave_result("YES", "YES", 0.8, "YES"),
    ]
    m = compute_cave_consistency_metrics(results)
    assert m["consistency_cs1"] == 1.0
    assert m["consistency_cs2"] == 1.0
    assert m["consistency"] == 1.0


def test_compute_cave_consistency_metrics_non_cave_returns_none():
    results = [make_result("YES", "YES", score=0.8, method="promptav")]
    m = compute_cave_consistency_metrics(results)
    assert m["consistency_cs1"] is None
    assert m["consistency_cs2"] is None
    assert m["consistency"] is None


def test_compute_cave_consistency_metrics_mixed_list():
    # Non-CAVE result first, then CAVE — should still compute from the CAVE result
    results = [
        make_result("YES", "YES", score=0.8, method="promptav"),
        _full_cave_result("YES", "YES", 0.9, "YES"),
    ]
    m = compute_cave_consistency_metrics(results)
    assert m["consistency_cs1"] is not None
    assert m["consistency_cs1"] == 1.0
