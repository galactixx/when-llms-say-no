# When LLMs Say No

Official code for the EMNLP 2026 Main Conference paper, "[When LLMs Say No: Diagnosing and Mitigating Conservative Bias in Authorship Verification](https://aclanthology.org/2026.emnlp-main.1477/)."

Nicholas Wlezien and Rakesh M. Verma (University of Houston). EMNLP 2026 Main Conference, pp. 33771–33788, Association for Computational Linguistics. DOI: [10.18653/2026.emnlp-main.1477](https://doi.org/10.18653/2026.emnlp-main.1477).

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

## Setup

```bash
python -m pip install -r requirements.txt
```

For the vLLM backend, install vLLM separately following the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html).

## Source layout

```text
src/
  evaluation/  # experiment runner, data, inference, prompts, and parsing
  analysis/    # summaries, significance tests, and semantic analysis
  graphs/      # the paper's figures
  shared/      # constants, models, metrics, output, and JSONL I/O
```

## Data layout

```
data/
  imdb62/test_i2ro.csv
  blog-auth/test_i2ro.csv
  fanfic/test_i2ro.csv
  cross-news/test_i2ro.csv
```

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
- `--qwen3-enable-thinking` — enable Qwen3 thinking mode

Supported models: `Qwen/Qwen3-{14,32}B`, `google/gemma-3-{12,27}b-it`, `mistralai/Mistral-Small-3.2-24B-Instruct-2506`.

## Experiment runs

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

## Citation

```bibtex
@inproceedings{wlezien-verma-2026-llms,
    title     = {When {LLM}s Say No: Diagnosing and Mitigating Conservative Bias in Authorship Verification},
    author    = {Wlezien, Nicholas and Verma, Rakesh M},
    editor    = {Lee, Hung-Yi and Dev, Sunipa and Li, Junyi Jessy and Elliott, Desmond},
    booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
    month     = {October},
    year      = {2026},
    address   = {Budapest, Hungary},
    publisher = {Association for Computational Linguistics},
    pages     = {33771--33788},
    url       = {https://aclanthology.org/2026.emnlp-main.1477/},
    doi       = {10.18653/2026.emnlp-main.1477},
    isbn      = {979-8-89176-459-0}
}
```
