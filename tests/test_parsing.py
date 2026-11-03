from src.evaluation.parsing import (
    extract_json_blob,
    parse_cave_json_like_blob,
    parse_model_output,
    parse_promptav_style_output,
    promptav_assign_label,
    promptav_decimal_scores,
)


def test_extract_json_blob_fenced():
    text = '```json\n{"key": "value"}\n```'
    assert extract_json_blob(text) == '{"key": "value"}'


def test_extract_json_blob_raw():
    text = 'Some text before {"key": "value"} and after'
    assert extract_json_blob(text) == '{"key": "value"}'


def test_extract_json_blob_nested():
    text = '{"outer": {"inner": 1}}'
    assert extract_json_blob(text) == '{"outer": {"inner": 1}}'


def test_extract_json_blob_none():
    assert extract_json_blob("no json here") is None


def test_parse_cave_json_blob_valid():
    blob = '{"output": "YES", "final score": 0.8}'
    result = parse_cave_json_like_blob(blob)
    assert result == {"output": "YES", "final score": 0.8}


def test_parse_cave_json_blob_python_dict():
    blob = "{'output': 'NO', 'final score': 0.3}"
    result = parse_cave_json_like_blob(blob)
    assert result == {"output": "NO", "final score": 0.3}


def test_parse_cave_json_blob_invalid():
    assert parse_cave_json_like_blob("not a dict at all") is None


def test_promptav_decimal_scores_finds_decimals():
    assert promptav_decimal_scores("I give this a score of 0.75.") == ["0.75"]


def test_promptav_decimal_scores_multiple():
    scores = promptav_decimal_scores("First 0.3 then 0.8 finally 0.6")
    assert scores == ["0.3", "0.8", "0.6"]


def test_promptav_decimal_scores_none():
    assert promptav_decimal_scores("no score here") == []


def test_promptav_decimal_scores_requires_word_boundary():
    assert promptav_decimal_scores("confidence: .85") == []
    assert promptav_decimal_scores("confidence: 0.85") == ["0.85"]


def test_promptav_assign_label_above_threshold():
    assert promptav_assign_label("0.6") == "YES"


def test_promptav_assign_label_below_threshold():
    assert promptav_assign_label("0.4") == "NO"


def test_promptav_assign_label_at_threshold():
    assert promptav_assign_label("0.5") == "NO"


def test_parse_promptav_style_output_extracts_last_score():
    _, label, score = parse_promptav_style_output("Initial 0.3 then 0.8 confidence")
    assert label == "YES"
    assert score == 0.8


def test_parse_promptav_style_output_no_score():
    _, label, score = parse_promptav_style_output("no score at all")
    assert label is None
    assert score is None


def test_parse_model_output_promptav_method():
    raw = "After analysis, confidence is 0.72."
    _, label, score = parse_model_output(raw, "promptav")
    assert label == "YES"
    assert score == 0.72


def test_parse_model_output_cave_method():
    raw = '{"output": "YES", "final score": 0.9, "writing style": "Similar. YES"}'
    obj, label, score = parse_model_output(raw, "prompt_cave")
    assert label == "YES"
    assert score == 0.9
    assert obj is not None


def test_parse_model_output_cave_no_json():
    _, label, score = parse_model_output("no json here", "prompt_cave")
    assert label is None
    assert score is None
