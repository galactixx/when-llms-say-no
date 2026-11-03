# When LLMs Say No

Empirical audit of prompt-based authorship verification (AV) on the CAVE datasets and a randomly stratified sample from the Cross-News article-vs-article dataset. Tests whether LLM prompting methods are biased toward misclassifying same-author pairs that are semantically distant in content.

## What it does

**`src/evaluation/main.py`** runs 9 zero-shot prompting methods on text-pair AV datasets:

| Method | Description |
|---|---|
| `cot` | Chain-of-thought (PromptAV paper) |
| `promptav` | PromptAV zero-shot baseline |
| `promptav_framing_only` | PromptAV + topic-blind framing |
| `promptav_contrastive_only` | PromptAV + stylometric contrastive instruction |
| `promptav_framed_contrastive` | PromptAV + both framing and contrastive |
| `prompt_cave` | Prompt-CAVE (structured JSON output) |
| `prompt_cave_framing_only` | Prompt-CAVE + topic-blind framing |
| `prompt_cave_contrastive_only` | Prompt-CAVE + stylometric contrastive instruction |
| `prompt_cave_framed_contrastive` | Prompt-CAVE + framing and contrastive |

Outputs per run: `results.jsonl`.

**`src/analysis/semantic_analysis.py`** runs post-hoc analysis on one or more `results.jsonl` files, testing whether semantic distance between text pairs predicts misclassification. Outputs binned accuracy tables and logistic regression results.

**`src/graphs/`** turns that analysis into the paper's figures; **`src/tables/`** produces the numbered tables. Both only read results and reuse the analysis code, so neither can drift from what the pipeline computes.

## Setup

```bash
python -m pip install -r requirements.txt
```

For the vLLM backend, install vLLM separately following the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html).

## Source layout

The existing modules are grouped by responsibility without splitting their contents:

```text
src/
  evaluation/  # experiment runner, data, inference, prompts, and parsing
  analysis/    # summaries, significance tests, and semantic analysis
  graphs/      # the paper's figures
  tables/      # the paper's numbered tables (CSV + paste-ready LaTeX)
  shared/      # constants, models, metrics, output, and JSONL I/O
```

Generated artefacts live outside `src/`:

```text
outputs/       # per-model results.jsonl, plus tables/ (CSV and .tex)
plots/         # figure PNGs
```

## Data layout

```
data/
  imdb62/test_i2ro.csv
  blog-auth/test_i2ro.csv
  fanfic/test_i2ro.csv
  cross-news/test_i2ro.csv
```

The first three datasets are from CAVE. `cross-news` is a randomly stratified sample from the Cross-News article-vs-article dataset. Each CSV requires `input` and `output` columns.

## Running evaluation

All models run via vLLM. Example:

```bash
python -m src.evaluation.main \
  --data-root ./data \
  --model Qwen/Qwen3-14B \
  --output-dir ./outputs/qwen3-14b \
  --batch-size 8
```

**Useful flags:**
- `--datasets imdb62 fanfic` — run on a subset of datasets
- `--methods promptav prompt_cave` — run specific methods only
- `--max-examples 250` — cap examples per dataset
- `--resume` — skip already-completed rows in `results.jsonl`
- `--qwen3-enable-thinking` — enable Qwen3 thinking mode

Supported models: `Qwen/Qwen3-{8,14,32}B`, `google/gemma-3-{1,4,12,27}b-it`, `mistralai/Mistral-Small-3.2-24B-Instruct-2506`.

## Experiment runs

Commands used to produce the results in `outputs/`. All runs use vLLM with bfloat16, batch size 8, 4096 max new tokens.

**Qwen3-32B:**
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m src.evaluation.main \
  --data-root ./data \
  --model Qwen/Qwen3-32B \
  --batch-size 8 \
  --max-new-tokens 4096 \
  --hf-dtype bfloat16 \
  --trust-remote-code \
  --vllm-gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --output-dir outputs/qwen3_32b \
  > run_qwen3_32b.log 2>&1 &
```

**Qwen3-14B:**
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m src.evaluation.main \
  --data-root ./data \
  --model Qwen/Qwen3-14B \
  --batch-size 8 \
  --max-new-tokens 4096 \
  --hf-dtype bfloat16 \
  --trust-remote-code \
  --vllm-gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --output-dir outputs/qwen3_14b \
  > run_qwen3_14b.log 2>&1 &
```

**Gemma 3 27B:**
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m src.evaluation.main \
  --data-root ./data \
  --model google/gemma-3-27b-it \
  --batch-size 8 \
  --max-new-tokens 4096 \
  --hf-dtype bfloat16 \
  --trust-remote-code \
  --vllm-gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --output-dir outputs/gemma3_27b \
  > run_gemma3_27b.log 2>&1 &
```

**Gemma 3 12B:**
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m src.evaluation.main \
  --data-root ./data \
  --model google/gemma-3-12b-it \
  --batch-size 8 \
  --max-new-tokens 4096 \
  --hf-dtype bfloat16 \
  --trust-remote-code \
  --vllm-gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --output-dir outputs/gemma3_12b \
  > run_gemma3_12b.log 2>&1 &
```

**Mistral Small 3.2 24B:**
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m src.evaluation.main \
  --data-root ./data \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --batch-size 8 \
  --max-new-tokens 4096 \
  --hf-dtype bfloat16 \
  --trust-remote-code \
  --vllm-gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --output-dir outputs/mistral24b \
  > run_mistral24b.log 2>&1 &
```

## Post-run analysis

### Significance tests

```bash
python -m src.analysis.significance \
  --results outputs/gemma3_12b/results.jsonl \
            outputs/gemma3_27b/results.jsonl \
            outputs/mistral24b/results.jsonl \
            outputs/qwen3_14b/results.jsonl \
            outputs/qwen3_32b/results.jsonl \
  --output-dir outputs/significance_all
```

Key outputs in `outputs/significance_all/`:
- `aggregate_directional_summary.csv` — mean Δ and improvement counts per variant/metric
- `appendix_all_promptav_variant_cell_tests.csv` — per-cell paired test results
- `appendix_significance_counts.csv` — Holm-adjusted significance counts

To also run the pooled logistic regression:

```bash
python -m src.analysis.significance \
  --results outputs/gemma3_12b/results.jsonl \
            outputs/gemma3_27b/results.jsonl \
            outputs/mistral24b/results.jsonl \
            outputs/qwen3_14b/results.jsonl \
            outputs/qwen3_32b/results.jsonl \
  --output-dir outputs/significance_all \
  --run-regression \
  --run-robustness-regression
```

Additional outputs: `regression_false_no_vs_semantic_distance.csv`, `regression_model_summary.csv`. The key result is the `semantic_distance_z` row.

### Semantic distance plots

```bash
python -m src.graphs.semantic_distance_plots \
  --results outputs/gemma3_12b/results.jsonl \
            outputs/gemma3_27b/results.jsonl \
            outputs/mistral24b/results.jsonl \
            outputs/qwen3_14b/results.jsonl \
            outputs/qwen3_32b/results.jsonl \
  --output-dir plots
```

Outputs in `plots/`: per-model quintile plots (`{model}_promptav_quintiles.png`, `{model}_prompt_cave_quintiles.png`), all-models averaged plots, and standalone legend images.

## Running tests

```bash
pytest
```

Tests cover parsing, metrics, data loading, inference utilities, prompt building, and I/O helpers.
