"""Reads CSV files, splits text pairs, and parses gold labels."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from ..shared.models import Example


def read_examples(
    csv_path: Path, dataset: str, max_examples: Optional[int], random_seed: int
) -> list[Example]:
    df = pd.read_csv(csv_path)
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(
            f"Expected columns 'input' and 'output' in {csv_path}, found {list(df.columns)}"
        )

    row_indices = list(df.index)
    if max_examples is not None:
        rng = random.Random(random_seed)
        row_indices = rng.sample(row_indices, min(max_examples, len(row_indices)))

    examples: list[Example] = []
    for idx in row_indices:
        row = df.loc[idx]
        text1, text2 = split_text_pair(str(row["input"]))
        gold_label = parse_gold_label(str(row["output"]))
        examples.append(
            Example(
                dataset=dataset,
                row_id=int(idx),
                text1=text1,
                text2=text2,
                gold_label=gold_label,
            )
        )
    return examples


def split_text_pair(raw: str) -> tuple[str, str]:
    m = re.search(r"Text1:\s*(.*?)\s*Text2:\s*(.*)", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    pieces = re.split(r"\bText2:\s*", raw, maxsplit=1, flags=re.DOTALL)
    if len(pieces) == 2:
        text1 = re.sub(r"^\s*Text1:\s*", "", pieces[0], flags=re.DOTALL).strip()
        return text1, pieces[1].strip()
    raise ValueError("Could not split input into Text1 and Text2.")


def parse_gold_label(raw: str) -> str:
    try:
        obj = json.loads(raw)
        label = str(obj.get("output", "")).strip().upper()
        if label in {"YES", "NO"}:
            return label
    except json.JSONDecodeError:
        pass
    raise ValueError(f"Could not parse gold label from: {raw[:120]}...")
