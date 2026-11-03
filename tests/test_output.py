import json

from src.shared.models import Result
from src.shared.output import (
    load_seen,
    make_threshold_sensitivity,
    result_from_json,
    save_csv,
    summarize_results,
)


def make_result(gold, pred, score=None, method="promptav", dataset="imdb62", row_id=0):
    return Result(
        dataset=dataset,
        row_id=row_id,
        method=method,
        gold_label=gold,
        pred_label=pred,
        correct=(pred == gold if pred in {"YES", "NO"} else None),
        final_score=score,
        prompt="",
        raw_response="",
        parsed_json=None,
        backend="vllm",
        model="test",
    )


def test_result_from_json_roundtrip():
    r = make_result("YES", "YES", score=0.8, row_id=5)
    row = {
        "dataset": r.dataset,
        "row_id": r.row_id,
        "method": r.method,
        "gold_label": r.gold_label,
        "pred_label": r.pred_label,
        "correct": r.correct,
        "final_score": r.final_score,
        "prompt": r.prompt,
        "raw_response": r.raw_response,
        "parsed_json": r.parsed_json,
        "backend": r.backend,
        "model": r.model,
    }
    restored = result_from_json(row, "vllm", "test")
    assert restored.gold_label == "YES"
    assert restored.final_score == 0.8
    assert restored.row_id == 5


def test_result_from_json_missing_optional_fields():
    row = {
        "dataset": "imdb62",
        "row_id": 0,
        "method": "promptav",
        "gold_label": "YES",
    }
    r = result_from_json(row, "vllm", "test")
    assert r.pred_label is None
    assert r.error is None
    assert r.prompt == ""


def test_load_seen_deduplicates(tmp_path):
    results_file = tmp_path / "results.jsonl"
    rows = [
        {"dataset": "imdb62", "row_id": 0, "method": "promptav", "gold_label": "YES"},
        {"dataset": "imdb62", "row_id": 1, "method": "promptav", "gold_label": "NO"},
        {"dataset": "imdb62", "row_id": 0, "method": "cot", "gold_label": "YES"},
    ]
    with results_file.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    seen = load_seen(results_file)
    assert ("imdb62", 0, "promptav") in seen
    assert ("imdb62", 1, "promptav") in seen
    assert ("imdb62", 0, "cot") in seen
    assert len(seen) == 3


def test_load_seen_missing_file(tmp_path):
    seen = load_seen(tmp_path / "nonexistent.jsonl")
    assert seen == set()


def test_save_csv_writes_headers(tmp_path):
    path = tmp_path / "out.csv"
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    save_csv(path, rows, ["a", "b"])
    lines = path.read_text().splitlines()
    assert lines[0] == "a,b"
    assert len(lines) == 3


def test_save_csv_empty_uses_default_fields(tmp_path):
    path = tmp_path / "out.csv"
    save_csv(path, [], ["x", "y", "z"])
    lines = path.read_text().splitlines()
    assert lines[0] == "x,y,z"


def test_summarize_results_groups_by_dataset_method():
    results = [
        make_result("YES", "YES", score=0.8, dataset="imdb62", method="promptav"),
        make_result("NO", "NO", score=0.2, dataset="imdb62", method="promptav"),
        make_result("YES", "YES", score=0.9, dataset="fanfic", method="cot"),
    ]
    rows = summarize_results(results)
    keys = {(r["dataset"], r["method"]) for r in rows}
    assert ("imdb62", "promptav") in keys
    assert ("fanfic", "cot") in keys
    assert len(rows) == 2


def test_make_threshold_sensitivity_covers_range():
    results = [
        make_result("YES", "YES", score=0.8),
        make_result("NO", "NO", score=0.2),
    ]
    rows = make_threshold_sensitivity(results, step=0.5)
    thresholds = [r["threshold"] for r in rows]
    assert 0.0 in thresholds
    assert 0.5 in thresholds
    assert 1.0 in thresholds


def test_make_threshold_sensitivity_skips_unscored():
    results = [make_result("YES", "YES", score=None)]
    rows = make_threshold_sensitivity(results, step=0.5)
    assert rows == []
