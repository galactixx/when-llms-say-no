import csv
import json
from pathlib import Path

import pytest

from src.evaluation.data import parse_gold_label, read_examples, split_text_pair


def test_split_text_pair_standard():
    raw = "Text1: Hello world Text2: Goodbye world"
    t1, t2 = split_text_pair(raw)
    assert t1 == "Hello world"
    assert t2 == "Goodbye world"


def test_split_text_pair_multiline():
    raw = "Text1: First line\nsecond line Text2: Other text"
    t1, t2 = split_text_pair(raw)
    assert "First line" in t1
    assert t2 == "Other text"


def test_split_text_pair_fallback():
    raw = "Some preamble Text2: Second part only"
    _, t2 = split_text_pair(raw)
    assert t2 == "Second part only"


def test_split_text_pair_no_split_raises():
    with pytest.raises(ValueError, match="Could not split"):
        split_text_pair("no markers here at all")


def test_parse_gold_label_json_yes():
    raw = json.dumps({"output": "YES"})
    assert parse_gold_label(raw) == "YES"


def test_parse_gold_label_json_no():
    raw = json.dumps({"output": "NO"})
    assert parse_gold_label(raw) == "NO"


def test_parse_gold_label_case_insensitive():
    raw = json.dumps({"output": "yes"})
    assert parse_gold_label(raw) == "YES"


def test_parse_gold_label_regex_fallback():
    raw = '{"output": "YES", "explanation": "same author"}'
    assert parse_gold_label(raw) == "YES"


def test_parse_gold_label_invalid_raises():
    with pytest.raises(ValueError, match="Could not parse"):
        parse_gold_label('{"output": "MAYBE"}')


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(rows)


def test_read_examples_basic(tmp_path):
    csv_path = tmp_path / "test_i2ro.csv"
    _write_csv(
        csv_path,
        [
            {
                "input": "Text1: Hello world Text2: Goodbye world",
                "output": '{"output": "YES"}',
            },
            {"input": "Text1: Foo bar Text2: Baz qux", "output": '{"output": "NO"}'},
        ],
    )
    examples = read_examples(
        csv_path, dataset="imdb62", max_examples=None, random_seed=42
    )
    assert len(examples) == 2
    assert examples[0].gold_label == "YES"
    assert examples[1].gold_label == "NO"
    assert examples[0].text1 == "Hello world"
    assert examples[0].text2 == "Goodbye world"


def test_read_examples_max_examples(tmp_path):
    csv_path = tmp_path / "test_i2ro.csv"
    _write_csv(
        csv_path,
        [
            {"input": f"Text1: A{i} Text2: B{i}", "output": '{"output": "YES"}'}
            for i in range(10)
        ],
    )
    examples = read_examples(csv_path, dataset="imdb62", max_examples=3, random_seed=42)
    assert len(examples) == 3


def test_read_examples_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w") as f:
        f.write("text,label\nhello,YES\n")
    with pytest.raises(ValueError, match="Expected columns"):
        read_examples(csv_path, dataset="imdb62", max_examples=None, random_seed=42)
