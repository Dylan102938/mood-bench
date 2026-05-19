# MOOD Bench

<p align="center">
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/📄_Paper-arXiv-b31b1b?style=for-the-badge" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/mood-bench/mood-bench"><img src="https://img.shields.io/badge/🤗_Dataset-HuggingFace-ffd21e?style=for-the-badge" alt="Dataset"></a>
  <a href="https://huggingface.co/spaces/mood-bench/leaderboard"><img src="https://img.shields.io/badge/🏆_Leaderboard-HuggingFace_Spaces-6366f1?style=for-the-badge" alt="Leaderboard"></a>
  <br>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square" alt="Python 3.12+"></a>
</p>

<p align="center" style="margin-top: 15px;"><b>A multi-domain out-of-distribution safety benchmark for LLMs.</b></p>

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Installing from PyPI](#installing-from-pypi)
  - [Building from source](#building-from-source)
  - [Verifying installation](#verifying-installation)
- [Usage](#usage)
  - [Running the built-in pipelines](#running-the-built-in-pipelines)
  - [Analyzing pre-scored results](#analyzing-pre-scored-results)
  - [Running your own pipelines](#running-your-own-pipelines)
- [Code structure overview](#code-structure-overview)
- [Further issues and questions](#further-issues-and-questions)

## Introduction

**MOOD Bench** (Misalignment out-of-distribution benchmark) measures how well safety monitors generalize beyond the data they were trained on. MOOD is built to evaluate monitors that are trained on a specific restricted [training split](https://huggingface.co/datasets/mood-bench/mood-bench/). Models are embedded as part of monitoring **pipelines**, and each pipeline is calibrated on a small set of *in-distribution* conversations (helpful, harmless, function-calling) and then evaluated on diverse *out-of-distribution* unsafe behaviors — jailbreaking, sycophancy, scheming, insecure code, controlling responses, missing or inappropriate function calls, etc.

## Installation

### Installing from PyPI

```bash
pip install mood-bench

# With vLLM support (recommended for the instruction-tuned pipeline)
pip install mood-bench[vllm]
```

### Building from source

Clone the repo:

```bash
git clone https://github.com/Dylan102938/mood-bench.git
cd mood-bench
```

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv
uv sync
```

Or with pip:

```bash
pip install -e ".[vllm]"
```

### Verifying installation

Run the CLI without arguments to confirm the `mood` entry point is on your path:

```bash
mood --help
```

To run the unit tests:

```bash
uv run pytest tests/test_cli.py tests/test_analysis.py
```

End-to-end tests that pull real adapters and require a GPU live under `tests/e2e/`:

```bash
uv run pytest tests/e2e/ -m gpu
```

## Usage

### Running the built-in pipelines

The `mood` CLI exposes one subcommand per built-in pipeline under `mood bench` and a separate `mood analyze` for post-hoc analysis.

**Guard model** — a sequence-classification model with a binary safe/unsafe head, optionally with a LoRA adapter:

```bash
mood bench guard \
    --model-id google/gemma-2-2b \
    --adapter-id mood-bench/gemma-2-2b-guard \
    --output-dir results/gemma-2-2b-guard \
    --batch-size 8 \
    --max-length 2048
```

**Perplexity** — token-level negative log-likelihood under a causal LM, with optional LoRA adapter merged on top:

```bash
mood bench perplexity \
    --model-id google/gemma-2-2b \
    --adapter-id mood-bench/gemma-2-2b-causal-lm \
    --output-dir results/perplexity \
    --batch-size 8 \
    --max-length 2048
```

**Mahalanobis distance** — fits a Gaussian on safe in-distribution hidden states and scores each test sample by its distance from that distribution. Stats are cached under `--stats-cache-dir` so subsequent runs are fast:

```bash
mood bench mahalanobis \
    --model-id google/gemma-2-2b \
    --adapter-id mood-bench/gemma-2-2b-guard \
    --pooling cls \
    --stats-cache-dir mahalanobis-stats/ \
    --output-dir results/mahalanobis \
    --batch-size 4 \
    --max-length 2048
```

**Instruction-tuned judge** — an instruction-tuned LLM asked to score each sample. Uses vLLM if installed, falls back to `transformers`:

```bash
mood bench instruction-tuned \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --grading-type alignment \
    --num-few-shot 3 \
    --output-dir results/instruction-tuned
```

Every `mood bench` subcommand accepts a common set of flags (`--use-mini` for a quick sanity-check subset, `--domains` to evaluate a subset of misaligned settings, `--no-figures` to skip plots, `-v` for verbose output, etc.). Run `mood bench <pipeline> --help` for the full list.

Each run writes a versioned directory under `--output-dir` containing `results.jsonl` (per-sample scores), `analysis.json` (group-level AUROC and TPR@FPR), and per-group `score_hist.png` / `auroc.png` figures.

### Analyzing pre-scored results

`mood analyze` consumes one or more `results.jsonl` files (the format produced by `mood bench`) and re-runs the metric / figure step, optionally aggregating across multiple results datasets.

Single run:

```bash
mood analyze results/guard/results.jsonl --output-dir reports/guard
```

Combining multiple monitors with an aggregator (min / mean / lambda):

```bash
mood analyze \
    results/guard/results.jsonl \
    results/perplexity/results.jsonl \
    --aggregator lambda \
    --anchor-index 0 \
    --output-dir reports/guard+ppl
```

For an ensemble of identical-architecture guard runs, take the min:

```bash
mood analyze \
    results/guard-particle-0/results.jsonl \
    results/guard-particle-1/results.jsonl \
    results/guard-particle-2/results.jsonl \
    --aggregator min \
    --output-dir reports/guard-ensemble
```

### Running your own pipelines

The CLI is a thin wrapper around the `mood_bench` Python API. To plug in your own monitors, implement the `Pipeline` protocol — any callable that maps a list of conversation strings to a `(scores, metadata)` tuple — and hand it to `mood_bench()`:

```python
from mood_bench import mood_bench, GuardModelPipeline, load_tokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = load_tokenizer("mood-bench/gemma-2-2b-guard")
model = AutoModelForSequenceClassification.from_pretrained(
    "google/gemma-2-2b", dtype="bfloat16"
)

results, report = mood_bench(
    pipelines=GuardModelPipeline(model, tokenizer, unsafe_label_index=1),
    output_dir="results/my-guard",
    eval_batch_size=8,
    max_length=2048,
)
print(report["groups"]["overall"])
```

`mood_bench(...)` returns the scored `Dataset` and a metrics report dict. Passing `output_dir=None` skips disk writes and returns everything in-memory.

For multi-pipeline runs, pass a list of pipelines plus an `Aggregator`:

```python
from mood_bench import MinAggregate, mood_bench

mood_bench(
    pipelines=[guard_a, guard_b, guard_c],
    aggregator=MinAggregate(),
    output_dir="results/guard-ensemble",
)
```

The `[examples/](examples/)` directory contains complete, self-contained scripts that you can copy and adapt:


| Script                                                                         | What it shows                                                                                                                   |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `[examples/guard.py](examples/guard.py)`                                       | Minimal single-pipeline run with a guard model + LoRA adapter.                                                                  |
| `[examples/guard_ensemble.py](examples/guard_ensemble.py)`                     | Loading five guard adapters in sequence and aggregating with `MinAggregate`.                                                    |
| `[examples/mixture_guard_perplexity.py](examples/mixture_guard_perplexity.py)` | Combining a guard classifier with a perplexity scorer via `LambdaAggregate`, including per-pipeline `predict_safe` orientation. |
| `[examples/analysis.py](examples/analysis.py)`                                 | Standalone re-analysis script — equivalent to `mood analyze` but easier to fork for custom metrics.                             |


To re-run analysis from Python without re-scoring:

```python
from datasets import load_dataset
from mood_bench import mood_bench_analysis

ds = load_dataset("json", data_files="results/guard/results.jsonl", split="train")
scored_ds, analysis_report = mood_bench_analysis(results=ds, output_path="reports/guard")
```

## Code structure overview

The package is laid out as follows:

```
mood_bench/
├── core.py          # mood_bench() and mood_bench_analysis() entry points
├── data.py          # EvalDataset enum, load_mood_dataset(), domain constants
├── aggregator.py    # MinAggregate, MeanAggregate, LambdaAggregate
├── metrics.py       # tpr_at_fpr, ROC + score-histogram plotters
├── tokenize.py      # load_tokenizer() and chat-template rendering
├── pipeline/
│   ├── base.py              # Pipeline protocol
│   ├── guard.py             # GuardModelPipeline
│   ├── perplexity.py        # PerplexityPipeline
│   ├── mahalanobis.py       # MahalanobisPipeline + get_stats_for_model
│   └── instruction_tuned.py # InstructionTunedPipeline (vLLM/HF backends)
└── cli/
    ├── __init__.py          # `mood` entry point
    ├── _common.py           # shared CLI flags
    ├── guard.py             # `mood bench guard`
    ├── perplexity.py        # `mood bench perplexity`
    ├── mahalanobis.py       # `mood bench mahalanobis`
    ├── instruction_tuned.py # `mood bench instruction-tuned`
    └── analyze.py           # `mood analyze`

examples/         # Standalone Python scripts demonstrating the library API
tests/
├── test_cli.py        # Fast CLI tests with stub pipelines
├── test_analysis.py   # Aggregator + analysis tests
└── e2e/               # GPU-only end-to-end tests against real adapters
```

## Further issues and questions

If you run into a problem or have a question, please contact [Dylan Feng](https://dylanfeng.com) at [dfeng102938@berkeley.edu](mailto:dfeng102938@berkeley.edu).