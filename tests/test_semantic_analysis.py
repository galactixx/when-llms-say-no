import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.semantic_analysis import (
    false_no_by_quintile_by_model,
    infer_model_label,
    load_all_results,
    load_result_file,
    make_pair_level_yes,
    split_text_pair_from_prompt,
)


def test_infer_model_label_from_row():
    row = {"model": "google/gemma-3-12b-it"}
    assert infer_model_label(row, Path("results.jsonl")) == "google__gemma-3-12b-it"


def test_infer_model_label_falls_back_to_stem():
    row = {"model": ""}
    assert infer_model_label(row, Path("gemma_12b_run.jsonl")) == "gemma_12b_run"


def test_infer_model_label_slash_replaced():
    row = {"model": "Qwen/Qwen3-14B"}
    label = infer_model_label(row, Path("results.jsonl"))
    assert "/" not in label


def test_split_text_pair_from_prompt_basic():
    prompt = "Task: ...\nText1:Hello world\nText2:Goodbye world"
    t1, t2 = split_text_pair_from_prompt(prompt)
    assert t1 == "Hello world"
    assert t2 == "Goodbye world"


def test_split_text_pair_from_prompt_multiline():
    prompt = "Text1:First line\nsecond line\nText2:Other text"
    t1, t2 = split_text_pair_from_prompt(prompt)
    assert "First line" in t1
    assert t2 == "Other text"


def test_split_text_pair_from_prompt_raises():
    with pytest.raises(ValueError, match="Could not recover"):
        split_text_pair_from_prompt("no markers at all")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_result_file_basic(tmp_path):
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "test-model",
                "dataset": "imdb62",
                "row_id": 0,
                "method": "promptav",
                "gold_label": "YES",
                "pred_label": "YES",
                "correct": True,
                "final_score": 0.8,
                "prompt": "Task: ...\nText1:Hello\nText2:World",
            }
        ],
    )
    rows = load_result_file(path)
    assert len(rows) == 1
    assert rows[0].gold_label == "YES"
    assert rows[0].text1 == "Hello"
    assert rows[0].text2 == "World"


def test_load_result_file_skips_bad_prompt(tmp_path):
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "test-model",
                "dataset": "imdb62",
                "row_id": 0,
                "method": "promptav",
                "gold_label": "YES",
                "pred_label": "YES",
                "correct": True,
                "final_score": 0.8,
                "prompt": "no text markers here",
            }
        ],
    )
    rows = load_result_file(path)
    assert rows == []


def test_load_result_file_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert load_result_file(path) == []


def test_make_pair_level_yes_filters_gold():
    df = pd.DataFrame(
        {
            "gold_label": ["YES", "YES", "NO"],
            "pred_label": ["YES", "NO", "NO"],
            "correct": [True, False, True],
            "correct_int": [1, 0, 1],
            "semantic_cosine_distance": [0.1, 0.5, 0.2],
            "model": ["m", "m", "m"],
            "dataset": ["d", "d", "d"],
            "final_score": [0.8, 0.3, 0.2],
        }
    )
    yes = make_pair_level_yes(df)
    assert (yes["gold_label"] == "YES").all()
    assert len(yes) == 2


def test_make_pair_level_yes_false_no_flag():
    df = pd.DataFrame(
        {
            "gold_label": ["YES", "YES"],
            "pred_label": ["NO", "YES"],
            "correct": [False, True],
            "correct_int": [0, 1],
            "semantic_cosine_distance": [0.4, 0.1],
            "model": ["m", "m"],
            "dataset": ["d", "d"],
            "final_score": [0.3, 0.8],
        }
    )
    yes = make_pair_level_yes(df)
    assert yes.iloc[0]["false_no"] == 1
    assert yes.iloc[1]["false_no"] == 0


def test_false_no_by_quintile_by_model_basic():
    n = 20
    df = pd.DataFrame(
        {
            "gold_label": ["YES"] * n,
            "pred_label": ["YES"] * n,
            "correct": [True] * n,
            "correct_int": [1] * n,
            "semantic_cosine_distance": [i / n for i in range(n)],
            "model": ["m"] * n,
            "dataset": ["d"] * n,
            "method": ["promptav"] * n,
            "false_no": [0] * n,
            "final_score": [0.8] * n,
        }
    )
    result = false_no_by_quintile_by_model(df, n_bins=5)
    assert len(result) == 5
    assert (result["yes_accuracy"] == 1.0).all()
    assert (result["false_no_rate"] == 0.0).all()


def test_false_no_by_quintile_by_model_skips_small_groups():
    df = pd.DataFrame(
        {
            "gold_label": ["YES"] * 3,
            "pred_label": ["YES"] * 3,
            "correct": [True] * 3,
            "correct_int": [1] * 3,
            "semantic_cosine_distance": [0.1, 0.5, 0.9],
            "model": ["m"] * 3,
            "dataset": ["d"] * 3,
            "method": ["promptav"] * 3,
            "false_no": [0] * 3,
            "final_score": [0.8] * 3,
        }
    )
    result = false_no_by_quintile_by_model(df, n_bins=5)
    assert len(result) == 0


def _make_jsonl_row(
    row_id=0,
    model="test-model",
    method="promptav",
    gold="YES",
    pred="YES",
    correct=True,
    score=0.8,
):
    return {
        "model": model,
        "dataset": "imdb62",
        "row_id": row_id,
        "method": method,
        "gold_label": gold,
        "pred_label": pred,
        "correct": correct,
        "final_score": score,
        "prompt": f"Task: ...\nText1:Hello {row_id}\nText2:World {row_id}",
    }


def test_load_all_results_basic(tmp_path):
    p1 = tmp_path / "run1.jsonl"
    p2 = tmp_path / "run2.jsonl"
    _write_jsonl(p1, [_make_jsonl_row(row_id=0), _make_jsonl_row(row_id=1)])
    _write_jsonl(p2, [_make_jsonl_row(row_id=2, model="other-model")])
    df = load_all_results([p1, p2])
    assert len(df) == 3
    assert set(df.columns) >= {
        "model",
        "dataset",
        "method",
        "gold_label",
        "correct_int",
    }


def test_load_all_results_correct_int_mapping(tmp_path):
    p = tmp_path / "results.jsonl"
    _write_jsonl(
        p,
        [
            _make_jsonl_row(row_id=0, correct=True),
            _make_jsonl_row(row_id=1, correct=False),
        ],
    )
    df = load_all_results([p])
    assert df.loc[df["row_id"] == 0, "correct_int"].iloc[0] == 1
    assert df.loc[df["row_id"] == 1, "correct_int"].iloc[0] == 0


def test_load_all_results_empty_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(RuntimeError, match="No usable result rows found"):
        load_all_results([p])
