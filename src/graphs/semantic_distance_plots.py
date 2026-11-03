#!/usr/bin/env python3
"""
Semantic distance quintile plots for AV experiment results.

Produces per-model and all-models-averaged line plots of false-negative rate
across semantic distance quintiles, one figure per method group.

A false negative is a gold-YES pair predicted NO.

  {output_dir}/{model_slug}_promptav_quintiles.png
  {output_dir}/{model_slug}_prompt_cave_quintiles.png
  {output_dir}/all_models_by_dataset_promptav_quintiles.png
  {output_dir}/all_models_by_dataset_prompt_cave_quintiles.png
  {output_dir}/legend_promptav.png
  {output_dir}/legend_prompt_cave.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..analysis.semantic_analysis import (
    add_semantic_distances,
    false_no_by_quintile_by_model,
    load_all_results,
    make_pair_level_yes,
)

# Publication style
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
    }
)

COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = [
    "solid",
    (0, (5, 2)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
]

DATASET_ORDER = ["imdb62", "blog-auth", "fanfic", "cross-news"]

PROMPTAV_METHODS = [
    "promptav",
    "promptav_framed_contrastive",
    "promptav_framing_only",
    "promptav_contrastive_only",
]

PROMPT_CAVE_METHODS = [
    "prompt_cave",
    "prompt_cave_framed_contrastive",
    "prompt_cave_framing_only",
    "prompt_cave_contrastive_only",
]

METHOD_GROUPS = {
    "promptav": PROMPTAV_METHODS,
    "prompt_cave": PROMPT_CAVE_METHODS,
}

METHOD_LABELS: dict[str, str] = {
    "promptav": "PromptAV",
    "promptav_framed_contrastive": "PromptAV + Framing & Contrastive",
    "promptav_framing_only": "PromptAV + Framing only",
    "promptav_contrastive_only": "PromptAV + Contrastive only",
    "prompt_cave": "PromptCAVE",
    "prompt_cave_framed_contrastive": "PromptCAVE + Framing & Contrastive",
    "prompt_cave_framing_only": "PromptCAVE + Framing only",
    "prompt_cave_contrastive_only": "PromptCAVE + Contrastive only",
}

GROUP_TITLES = {
    "promptav": "PromptAV",
    "prompt_cave": "PromptCAVE",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Semantic distance quintile plots for AV experiment results."
    )
    p.add_argument("--results", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--n-bins", type=int, default=5)
    return p.parse_args()


def _model_slug(model: str) -> str:
    return model.replace("/", "__")


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "Qwen__Qwen3-14B": "Qwen3 14B",
    "Qwen__Qwen3-32B": "Qwen3 32B",
    "google__gemma-3-12b-it": "Gemma 3 12B",
    "google__gemma-3-27b-it": "Gemma 3 27B",
    "mistralai__Mistral-Small-3.2-24B-Instruct-2506": "Mistral Small 24B",
}


def _model_display(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def _quintile_xlabels(n_bins: int) -> list[str]:
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    labels[0] = f"Q1\n(most similar)"
    labels[-1] = f"Q{n_bins}\n(most different)"
    return labels


def plot_per_model(binned: pd.DataFrame, output_dir: Path, n_bins: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(1, n_bins + 1)
    datasets = [d for d in DATASET_ORDER if d in binned["dataset"].unique()]
    if not datasets:
        return

    for model, model_df in binned.groupby("model"):
        slug = _model_slug(model)
        for group_name, methods in METHOD_GROUPS.items():
            present = [m for m in methods if m in model_df["method"].unique()]
            if not present:
                continue

            fig, axes = plt.subplots(
                1, len(datasets), figsize=(5 * len(datasets), 4.5), sharey=True
            )
            if len(datasets) == 1:
                axes = [axes]

            for col, dataset in enumerate(datasets):
                ax = axes[col]
                dataset_df = model_df[model_df["dataset"] == dataset]

                for i, method in enumerate(present):
                    agg = (
                        dataset_df[dataset_df["method"] == method]
                        .groupby("bin_id")["false_no_rate"]
                        .mean()
                        .reindex(x)
                    )
                    ax.plot(
                        x,
                        agg.values,
                        color=COLORS[i % len(COLORS)],
                        marker=MARKERS[i % len(MARKERS)],
                        linestyle=LINESTYLES[i % len(LINESTYLES)],
                        zorder=3,
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(_quintile_xlabels(n_bins))
                ax.xaxis.grid(False)
                ax.set_xlabel("Semantic distance quintile", labelpad=8)
                ax.set_title(dataset)
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(
                    mticker.PercentFormatter(xmax=1, decimals=0)
                )
                if col == 0:
                    ax.set_ylabel("False-negative rate")

            fig.tight_layout(rect=[0.07, 0, 1, 1])
            fig.text(
                0.02,
                0.5,
                _model_display(model),
                va="center",
                ha="center",
                rotation="vertical",
                fontsize=11,
                fontweight="bold",
                transform=fig.transFigure,
            )
            path = output_dir / f"{slug}_{group_name}_quintiles.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  {path}")


def save_legend(methods: list[str], group_name: str, output_dir: Path) -> None:
    """Save the legend as a standalone tightly-cropped figure."""
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    for i, method in enumerate(methods):
        ax.plot(
            [],
            [],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_visible(False)
    legend = fig.legend(
        loc="center",
        frameon=True,
        fontsize=10,
        ncol=len(methods),
    )
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    path = output_dir / f"legend_{group_name}.png"
    fig.savefig(path, dpi=300, bbox_inches=bbox)
    plt.close(fig)
    print(f"  {path}")


def plot_all_models_by_dataset(
    binned: pd.DataFrame, output_dir: Path, n_bins: int
) -> None:
    """One figure per group with one subplot per dataset, averaged across all models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(1, n_bins + 1)
    datasets = [d for d in DATASET_ORDER if d in binned["dataset"].unique()]
    if not datasets:
        return

    for group_name, methods in METHOD_GROUPS.items():
        present = [m for m in methods if m in binned["method"].unique()]
        if not present:
            continue

        fig, axes = plt.subplots(
            1, len(datasets), figsize=(5 * len(datasets), 4.5), sharey=True
        )
        if len(datasets) == 1:
            axes = [axes]

        legend_handles: list = []
        legend_labels: list[str] = []

        for col, dataset in enumerate(datasets):
            ax = axes[col]
            dataset_df = binned[binned["dataset"] == dataset]

            for i, method in enumerate(present):
                agg = (
                    dataset_df[dataset_df["method"] == method]
                    .groupby("bin_id")["false_no_rate"]
                    .mean()
                    .reindex(x)
                )
                (line,) = ax.plot(
                    x,
                    agg.values,
                    color=COLORS[i % len(COLORS)],
                    marker=MARKERS[i % len(MARKERS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=METHOD_LABELS.get(method, method),
                    zorder=3,
                )
                if col == 0:
                    legend_handles.append(line)
                    legend_labels.append(METHOD_LABELS.get(method, method))

            ax.set_xticks(x)
            ax.set_xticklabels(_quintile_xlabels(n_bins))
            ax.xaxis.grid(False)
            ax.set_xlabel("Semantic distance quintile", labelpad=8)
            ax.set_title(dataset)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            if col == 0:
                ax.set_ylabel("False-negative rate")

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(present),
            bbox_to_anchor=(0.5, -0.12),
            frameon=True,
        )
        fig.tight_layout()
        path = output_dir / f"all_models_by_dataset_{group_name}_quintiles.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_results(args.results)

    print(f"Computing embeddings with {args.embedding_model} ...")
    df = add_semantic_distances(df, args.embedding_model, args.batch_size, args.device)

    yes = make_pair_level_yes(df)

    print("Binning by semantic distance quintile...")
    binned = false_no_by_quintile_by_model(yes, args.n_bins)

    if binned.empty:
        print(
            "No data after binning — check that result files contain YES-labeled rows."
        )
        return

    print(f"\nGenerating per-model plots → {args.output_dir}/")
    plot_per_model(binned, args.output_dir, args.n_bins)

    print(f"\nGenerating all-models by-dataset plots → {args.output_dir}/")
    plot_all_models_by_dataset(binned, args.output_dir, args.n_bins)

    print(f"\nSaving legend images → {args.output_dir}/")
    for group_name, methods in METHOD_GROUPS.items():
        present = [m for m in methods if m in binned["method"].unique()]
        if present:
            save_legend(present, group_name, args.output_dir)

    print(f"\nDone. All plots saved under {args.output_dir}/")


if __name__ == "__main__":
    main()
