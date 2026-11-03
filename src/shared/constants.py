"""
Shared constants used across evaluation and analysis.

Model registries
----------------
ALLOWED_MODEL_ALIASES   — maps user-supplied name variants to canonical model IDs
HF_ALLOWED_MODELS       — canonical IDs routed to the vLLM backend
MODEL_REGISTRY          — config/models.yaml contents: id -> {revision, aliases}
MODEL_REVISIONS         — canonical ID -> pinned HuggingFace commit SHA

Methods
-------
ZERO_SHOT_METHODS        — ordered list of all 9 supported prompting methods
PROMPTAV_SCORE_METHODS   — methods that output a free-text response with a decimal confidence
                           score
CAVE_CONSISTENCY_METHODS — methods that output structured JSON (used for CS-1/CS-2 consistency)

Thresholds
----------
PROMPTAV_SCORE_THRESHOLD         — fixed decision threshold (0.5) for score-only methods
CAVE_CONSISTENCY_SCORE_THRESHOLD — threshold used when computing CAVE CS-1 consistency

CAVE feature keys
-----------------
CAVE_FEATURE_KEYS — the 8 linguistic feature keys expected in Prompt-CAVE JSON output,
                    used to extract intermediate YES/NO/MAYBE labels for CS-2 consistency
"""

from __future__ import annotations

from pathlib import Path

import yaml

_MODELS_YAML = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _load_models() -> dict[str, dict]:
    """Model registry: canonical HF id -> {revision, aliases}."""
    with _MODELS_YAML.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["models"]


MODEL_REGISTRY = _load_models()

# Canonical id -> pinned HF commit SHA. Passed to the loader as `revision` so a
# run is tied to exact weights rather than whatever `main` happens to be.
MODEL_REVISIONS = {mid: spec["revision"] for mid, spec in MODEL_REGISTRY.items()}

# Every accepted spelling -> canonical id. Built from the registry so aliases
# and revisions cannot drift apart.
ALLOWED_MODEL_ALIASES = {}
for _mid, _spec in MODEL_REGISTRY.items():
    ALLOWED_MODEL_ALIASES[_mid.lower()] = _mid
    for _alias in _spec.get("aliases", []):
        ALLOWED_MODEL_ALIASES[_alias.lower()] = _mid

HF_ALLOWED_MODELS = set(MODEL_REGISTRY)

ZERO_SHOT_METHODS = [
    "cot",
    "promptav",
    "promptav_framing_only",
    "promptav_contrastive_only",
    "promptav_framed_contrastive",
    "prompt_cave",
    "prompt_cave_framing_only",
    "prompt_cave_contrastive_only",
    "prompt_cave_framed_contrastive",
]

# Score-only methods use PromptAV-style decimal-score extraction.
PROMPTAV_SCORE_METHODS = {
    "cot",
    "promptav",
    "promptav_framing_only",
    "promptav_contrastive_only",
    "promptav_framed_contrastive",
}

# Main fixed score threshold for score-only methods.
PROMPTAV_SCORE_THRESHOLD = 0.5

# CAVE consistency metrics are computed for fixed-key CAVE-style JSON outputs.
CAVE_CONSISTENCY_METHODS = {
    "prompt_cave",
    "prompt_cave_framing_only",
    "prompt_cave_contrastive_only",
    "prompt_cave_framed_contrastive",
}
CAVE_CONSISTENCY_SCORE_THRESHOLD = 0.5
CAVE_FEATURE_KEYS = [
    "punctuation style",
    "special characters style, capitalization style",
    "acronyms and abbreviations",
    "writing style",
    "expressions and Idioms",
    "tone and mood",
    "sentence structure",
    "any other relevant aspect",
]
