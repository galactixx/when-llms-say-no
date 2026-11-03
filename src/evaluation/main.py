#!/usr/bin/env python3
"""
Entry point for zero-shot authorship verification experiments on the CAVE datasets.

Runs 9 prompting methods across up to 4 datasets (imdb62, blog-auth, fanfic, cross-news) using
a local vLLM backend. Results are written incrementally to results.jsonl
and summarized to CSV at the end of each run.

Prompting methods
-----------------
1) cot                              — Chain-of-thought (PromptAV paper)
2) promptav                         — PromptAV zero-shot (PromptAV paper)
3) promptav_framing_only            — PromptAV + topic-blind framing instruction
4) promptav_contrastive_only        — PromptAV + stylometric contrastive instruction
5) promptav_framed_contrastive      — PromptAV + both framing and contrastive
6) prompt_cave                      — Prompt-CAVE structured JSON output (CAVE paper)
7) prompt_cave_framing_only         — Prompt-CAVE + topic-blind framing instruction
8) prompt_cave_contrastive_only     — Prompt-CAVE + stylometric contrastive instruction
9) prompt_cave_framed_contrastive   — Prompt-CAVE + both framing and contrastive

Outputs (written to --output-dir)
----------------------------------
results.jsonl                  — one JSON line per example/method with raw response and parsed fields
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from ..shared.constants import ZERO_SHOT_METHODS
from ..shared.models import Result
from ..shared.output import load_seen, read_all_results, summarize_results
from .data import read_examples
from .inference import (
    call_model_vllm_batch,
    init_vllm,
    model_revision,
    normalize_and_validate_model_name,
)
from .parsing import parse_model_output
from .prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact AV prompt baselines plus empirical audit diagnostics on CAVE datasets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the repo's data/ directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["imdb62", "blog-auth", "fanfic", "cross-news"],
        help="Datasets to evaluate. Expected folders under --data-root.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ZERO_SHOT_METHODS,
        choices=ZERO_SHOT_METHODS,
        help="Which zero-shot prompting methods to run.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap per dataset for random sampling.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when sampling examples.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between API calls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Where to write results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already written to results.jsonl.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Generation cap for local/vLLM models.",
    )
    parser.add_argument(
        "--hf-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype for vLLM open-source runs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face tokenizer/model loading.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for vLLM batched local generation.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length reserved by vLLM.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may use.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vLLM only).",
    )
    parser.add_argument(
        "--qwen3-enable-thinking",
        action="store_true",
        help="For Qwen3 models only: enable Qwen3 thinking mode in apply_chat_template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model = normalize_and_validate_model_name(args.model)

    vllm_state = init_vllm(
        args.model,
        args.hf_dtype,
        args.trust_remote_code,
        args.vllm_max_model_len,
        args.vllm_gpu_memory_utilization,
        args.tensor_parallel_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    seen = load_seen(results_path) if args.resume else set()

    for dataset in args.datasets:
        csv_path = args.data_root / dataset / "test_i2ro.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")
        examples = read_examples(
            csv_path,
            dataset=dataset,
            max_examples=args.max_examples,
            random_seed=args.random_seed,
        )

        for method in args.methods:
            pending: list[tuple] = []
            for ex in examples:
                key = (ex.dataset, ex.row_id, method)
                if key in seen:
                    continue
                pending.append((ex, build_prompt(method, ex.text1, ex.text2)))

            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                prompts = [prompt for _, prompt in batch]
                try:
                    raw_responses = call_model_vllm_batch(
                        vllm_state,
                        prompts,
                        args.temperature,
                        args.max_new_tokens,
                        args.model,
                        args.qwen3_enable_thinking,
                    )
                    errors: list[Optional[str]] = [None] * len(batch)
                except Exception as exc:  # noqa: BLE001
                    raw_responses = [""] * len(batch)
                    errors = [str(exc)] * len(batch)

                for (ex, prompt), raw_response, error in zip(
                    batch, raw_responses, errors
                ):
                    parsed_json = None
                    pred_label = None
                    final_score = None
                    if error is None:
                        try:
                            parsed_json, pred_label, final_score = parse_model_output(
                                raw_response, method
                            )
                        except Exception as exc:  # noqa: BLE001
                            error = str(exc)

                    correct: Optional[bool] = (
                        pred_label == ex.gold_label
                        if pred_label in {"YES", "NO"}
                        else None
                    )
                    result = Result(
                        dataset=ex.dataset,
                        row_id=ex.row_id,
                        method=method,
                        gold_label=ex.gold_label,
                        pred_label=pred_label,
                        correct=correct,
                        final_score=final_score,
                        prompt=prompt,
                        raw_response=raw_response,
                        parsed_json=parsed_json,
                        backend="vllm",
                        model=args.model,
                        model_revision=model_revision(args.model),
                        error=error,
                    )
                    with results_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    print(
                        f"[{dataset}][{method}] row={ex.row_id} gold={ex.gold_label} pred={pred_label} "
                        f"score={final_score} correct={correct}",
                        flush=True,
                    )
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

        # Per-dataset live summary.
        dataset_results = [
            r
            for r in read_all_results(results_path, "vllm", args.model)
            if r.dataset == dataset
        ]
        dataset_summary_rows = summarize_results(dataset_results)
        if dataset_summary_rows:
            print(f"\nSummary for {dataset}:")
            print(pd.DataFrame(dataset_summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
