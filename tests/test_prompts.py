import pytest

from src.evaluation.prompts import build_prompt, make_chat_messages
from src.shared.constants import ZERO_SHOT_METHODS


def test_build_prompt_all_methods():
    for method in ZERO_SHOT_METHODS:
        prompt = build_prompt(method, "Text one.", "Text two.")
        assert "Text one." in prompt
        assert "Text two." in prompt


def test_build_prompt_unknown_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        build_prompt("nonexistent_method", "a", "b")


def test_build_prompt_cot_contains_step_by_step():
    prompt = build_prompt("cot", "a", "b")
    assert "step by step" in prompt.lower()


def test_build_prompt_cave_contains_json_instruction():
    prompt = build_prompt("prompt_cave", "a", "b")
    assert "JSON" in prompt
    assert "final score" in prompt
    assert "output" in prompt


def test_build_prompt_framing_contains_framing_instruction():
    for method in (
        "promptav_framing_only",
        "promptav_framed_contrastive",
        "prompt_cave_framed_contrastive",
    ):
        prompt = build_prompt(method, "a", "b")
        assert "surface-form habits" in prompt


def test_build_prompt_contrastive_contains_contrastive_instruction():
    for method in (
        "promptav_contrastive_only",
        "promptav_framed_contrastive",
        "prompt_cave_framed_contrastive",
    ):
        prompt = build_prompt(method, "a", "b")
        assert "stylometric evidence" in prompt


def test_make_chat_messages_format():
    msgs = make_chat_messages("my prompt")
    assert msgs == [{"role": "user", "content": "my prompt"}]


def test_make_chat_messages_single_turn():
    msgs = make_chat_messages("hello")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
