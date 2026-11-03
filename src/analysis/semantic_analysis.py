#!/usr/bin/env python3
"""
Post-hoc semantic distance analysis for AV results.

Tests whether same-author pairs with high semantic/content distance are more likely
to be misclassified as NO. Reads one or more results.jsonl files produced by main.py
and outputs:

  false_no_by_quintile_by_model.csv   — false-NO rate by quintile per model/dataset/method
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..shared.io_utils import iter_jsonl

TEXT_PAIR_RE = re.compile(
    r"Text\s*1\s*:\s*(.*?)\s*Text\s*2\s*:\s*(.*)", re.DOTALL | re.IGNORECASE
)


@dataclass
class LoadedResult:
    source_file: str
    model: str
    dataset: str
    row_id: int
    method: str
    gold_label: str
    pred_label: Optional[str]
    correct: Optional[bool]
    final_score: Optional[float]
    text1: str
    text2: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal semantic-distance bias analysis for AV result JSONL files."
    )
    p.add_argument(
        "--results",
        nargs="+",
        type=Path,
        required=True,
        help="One or more results.jsonl files produced by main.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for analysis CSV outputs.",
    )
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model used to embed each text.",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    p.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device, e.g. cuda or cpu.",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of semantic-distance bins for binned accuracy tables.",
    )
    return p.parse_args()


def infer_model_label(row: dict[str, Any], source_path: Path) -> str:
    model = str(row.get("model") or "").strip()
    if model:
        return model.replace("/", "__")
    return source_path.stem


def split_text_pair_from_prompt(prompt: str) -> tuple[str, str]:
    matches = list(TEXT_PAIR_RE.finditer(prompt))
    if not matches:
        raise ValueError("Could not recover Text 1/Text 2 from prompt.")
    m = matches[-1]
    return m.group(1).strip(), m.group(2).strip()


def load_result_file(path: Path) -> list[LoadedResult]:
    rows: list[LoadedResult] = []
    for line_no, obj in enumerate(iter_jsonl(path), start=1):
        try:
            text1, text2 = split_text_pair_from_prompt(str(obj.get("prompt", "")))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] skipping {path}:{line_no}; could not parse texts: {exc}",
                file=sys.stderr,
            )
            continue

        method = str(obj.get("method"))
        final_score = (
            float(obj["final_score"]) if obj.get("final_score") is not None else None
        )
        raw_pred = obj.get("pred_label")
        pred_label = str(raw_pred).upper() if raw_pred is not None else None

        rows.append(
            LoadedResult(
                source_file=path.name,
                model=infer_model_label(obj, path),
                dataset=str(obj.get("dataset")),
                row_id=int(obj.get("row_id")),
                method=method,
                gold_label=str(obj.get("gold_label", "")).upper(),
                pred_label=pred_label,
                correct=obj.get("correct"),
                final_score=final_score,
                text1=text1,
                text2=text2,
            )
        )
    return rows


def load_all_results(paths: list[Path]) -> pd.DataFrame:
    all_rows: list[LoadedResult] = []
    for p in paths:
        all_rows.extend(load_result_file(p))
    if not all_rows:
        raise RuntimeError("No usable result rows found.")

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    df["correct_int"] = df["correct"].map(
        lambda x: int(bool(x)) if x is not None and not pd.isna(x) else np.nan
    )
    return df


def compute_embeddings(
    texts: list[str], model_name: str, batch_size: int, device: Optional[str]
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Install sentence-transformers first: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def add_semantic_distances(
    df: pd.DataFrame, embedding_model: str, batch_size: int, device: Optional[str]
) -> pd.DataFrame:
    df = df.copy()
    unique_texts = pd.Index(
        pd.concat([df["text1"], df["text2"]], ignore_index=True).drop_duplicates()
    )
    print(f"Embedding {len(unique_texts):,} unique texts with {embedding_model} ...")
    embs = compute_embeddings(
        unique_texts.tolist(), embedding_model, batch_size, device
    )
    text_to_idx = {text: i for i, text in enumerate(unique_texts)}

    idx1 = df["text1"].map(text_to_idx).to_numpy()
    idx2 = df["text2"].map(text_to_idx).to_numpy()
    sim = np.sum(embs[idx1] * embs[idx2], axis=1)
    df["semantic_cosine_similarity"] = sim
    df["semantic_cosine_distance"] = 1.0 - sim
    return df


def make_pair_level_yes(df: pd.DataFrame) -> pd.DataFrame:
    yes = df[df["gold_label"].eq("YES")].copy()
    yes = yes[yes["pred_label"].isin(["YES", "NO"])]
    yes["false_no"] = (
        (yes["gold_label"] == "YES") & (yes["pred_label"] == "NO")
    ).astype(int)
    return yes


def false_no_by_quintile_by_model(yes: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """False-NO rate by semantic distance quintile, per model/dataset/method."""
    rows: list[dict[str, Any]] = []
    for (model, dataset, method), g in yes.groupby(["model", "dataset", "method"]):
        if len(g) < n_bins:
            continue
        try:
            bins = pd.qcut(g["semantic_cosine_distance"], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        tmp = g.assign(distance_bin=bins)
        for bin_id, (interval, bg) in enumerate(
            tmp.groupby("distance_bin", observed=True), start=1
        ):
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "method": method,
                    "bin_id": bin_id,
                    "distance_bin": str(interval),
                    "n": len(bg),
                    "mean_semantic_distance": float(
                        bg["semantic_cosine_distance"].mean()
                    ),
                    "false_no_rate": float(bg["false_no"].mean()),
                    "yes_accuracy": float(bg["correct_int"].mean()),
                }
            )
    return pd.DataFrame(rows)


def print_output_summary(output_dir: Path, files: list[str]) -> None:
    print("\nWrote outputs to:", output_dir)
    for name in files:
        path = output_dir / name
        print(f"  - {path} ({path.stat().st_size:,} bytes)")

    print("\nHow to read the outputs:")
    print("  false_no_by_quintile_by_model.csv:")
    print(
        "    False-NO rate across low-to-high semantic-distance quintiles per model/dataset/method"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(args.results)
    df = add_semantic_distances(df, args.embedding_model, args.batch_size, args.device)
    yes = make_pair_level_yes(df)

    false_no_by_quintile_by_model(yes, args.n_bins).to_csv(
        args.output_dir / "false_no_by_quintile_by_model.csv", index=False
    )

    print_output_summary(
        args.output_dir,
        ["false_no_by_quintile_by_model.csv"],
    )


if __name__ == "__main__":
    main()
