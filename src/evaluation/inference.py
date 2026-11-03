"""Model validation and vLLM call helpers."""

from __future__ import annotations

from typing import Any

from ..shared.constants import ALLOWED_MODEL_ALIASES, MODEL_REVISIONS
from .prompts import make_chat_messages


def normalize_and_validate_model_name(model_name: str) -> str:
    key = model_name.strip()
    canonical = ALLOWED_MODEL_ALIASES.get(key.lower())
    if canonical is None:
        raise ValueError(
            "Unsupported model. Allowed models are: "
            "Qwen/Qwen3-{8,14,32}B, google/gemma-3-{1,4,12,27}b-it, "
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506."
        )
    return canonical


def model_revision(model_name: str) -> str:
    """Pinned HF commit SHA for a canonical model id (see config/models.yaml)."""
    try:
        return MODEL_REVISIONS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"No pinned revision for {model_name!r}; add it to config/models.yaml."
        ) from exc


def is_qwen3_model(model_name: str) -> bool:
    return "qwen3" in model_name.lower()


def init_vllm(
    model_name: str,
    dtype: str,
    trust_remote_code: bool,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
):
    try:
        from vllm import LLM
    except Exception as exc:  # pragma: no cover
        raise ImportError("vLLM backend requires vllm.") from exc

    vllm_dtype = dtype if dtype != "auto" else "float16"
    return LLM(
        model=model_name,
        revision=model_revision(model_name),
        dtype=vllm_dtype,
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )


def call_model_vllm_batch(
    llm: Any,
    prompts: list[str],
    temperature: float,
    max_new_tokens: int,
    model_name: str,
    qwen3_enable_thinking: bool = False,
) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    conversations = [make_chat_messages(prompt) for prompt in prompts]

    # For Qwen3, render the chat template manually so --qwen3-enable-thinking works
    # consistently across backends.
    if is_qwen3_model(model_name):
        tokenizer = llm.get_tokenizer()
        rendered_prompts: list[str] = []
        for conversation in conversations:
            chat_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": qwen3_enable_thinking,
            }
            try:
                rendered = tokenizer.apply_chat_template(conversation, **chat_kwargs)
            except TypeError:
                # Fall back to the default Qwen chat template.
                chat_kwargs.pop("enable_thinking", None)
                rendered = tokenizer.apply_chat_template(conversation, **chat_kwargs)
            rendered_prompts.append(rendered)
        outputs = llm.generate(rendered_prompts, sampling_params)
    else:
        outputs = llm.chat(conversations, sampling_params)

    return [out.outputs[0].text if out.outputs else "" for out in outputs]
