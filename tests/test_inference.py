import pytest

from src.evaluation.inference import is_qwen3_model, normalize_and_validate_model_name


def test_normalize_qwen_shorthand():
    assert normalize_and_validate_model_name("qwen3-14b") == "Qwen/Qwen3-14B"


def test_normalize_gemma_shorthand():
    assert (
        normalize_and_validate_model_name("gemma-3-12b-it") == "google/gemma-3-12b-it"
    )


def test_normalize_mistral():
    assert (
        normalize_and_validate_model_name("mistral-small-3.2-24b-instruct-2506")
        == "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    )


def test_normalize_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported model"):
        normalize_and_validate_model_name("gpt-4o")


def test_normalize_strips_whitespace():
    assert normalize_and_validate_model_name("  qwen3-14b  ") == "Qwen/Qwen3-14B"


def test_is_qwen3_true():
    assert is_qwen3_model("Qwen/Qwen3-14B") is True


def test_is_qwen3_false():
    assert is_qwen3_model("google/gemma-3-12b-it") is False


def test_is_qwen3_case_insensitive():
    assert is_qwen3_model("qwen3-8b") is True
