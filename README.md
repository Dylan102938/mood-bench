# Mood-Bench

[Python 3.12+](https://www.python.org/downloads/)
[License: MIT](https://opensource.org/licenses/MIT)
[Hugging Face Dataset](https://huggingface.co/datasets/mood-bench/mood-bench)

A multi-domain out-of-distribution safety benchmark for LLMs.

## Introduction

**Mood-Bench** (Misalignment Out-of-distribution benchmark) measures whether safety monitors generalize beyond the data they were trained on. Each pipeline is calibrated on a small set of *in-distribution* conversations (helpful, harmless, function-calling) and then evaluated on diverse *out-of-distribution* unsafe behaviors — jailbreaking, sycophancy, scheming, insecure code, controlling responses, missing or inappropriate function calls, and more.

Mood-Bench ships four reference monitor pipelines out of the box:

- **Guard model** — fine-tuned binary classifier head (`mood bench guard`)
- **Perplexity** — token-level NLL from a causal LM (`mood bench perplexity`)
- **Mahalanobis distance** — anomaly score in hidden-state space (`mood bench mahalanobis`)
- **Instruction-tuned judge** — an LLM scoring its own outputs (`mood bench instruction-tuned`)

It also ships aggregators (mean / min / lambda mixture) so monitors can be combined, and an `analyze` command that turns scored JSONL files into AUROC and TPR-at-FPR reports with score histograms and ROC plots.

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
git clone https://github.com/shizwick/mood-bench.git
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
    --adapter-id shizwick/google-gemma-2-2b_guard \
    --output-dir results/guard \
    --batch-size 8 --max-length 2048
```

**Perplexity** — token-level negative log-likelihood under a causal LM, with optional LoRA adapter merged on top:

```bash
mood bench perplexity \
    --model-id google/gemma-2-2b \
    --adapter-id shizwick/google-gemma-2-2b_causal-lm \
    --output-dir results/perplexity \
    --batch-size 8 --max-length 2048
```

**Mahalanobis distance** — fits a Gaussian on safe in-distribution hidden states and scores each test sample by its distance from that distribution. Stats are cached under `--stats-cache-dir` so subsequent runs are fast:

```bash
mood bench mahalanobis \
    --model-id google/gemma-2-2b \
    --adapter-id shizwick/google-gemma-2-2b_guard \
    --pooling cls \
    --stats-cache-dir mahalanobis-stats/ \
    --output-dir results/mahalanobis \
    --batch-size 4 --max-length 2048
```

**Instruction-tuned judge** — an instruction-tuned LLM asked to score each sample. Uses vLLM if installed, falls back to `transformers`:

```bash
mood bench instruction-tuned \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --grading-type alignment \
    --num-few-shot 3 \
    --output-dir results/instruction-tuned
```

Every `mood bench` subcommand accepts a common set of flags (`--use-mini` for a quick sanity-check subset, `--domains` to evaluate a subset, `--no-figures` to skip plots, `-v` for verbose output, etc.). Run `mood bench <pipeline> --help` for the full list.

Each run writes a versioned directory under `--output-dir` containing `results.jsonl` (per-sample scores), `analysis.json` (group-level AUROC and TPR@FPR), and per-group `score_hist.png` / `auroc.png` figures.

### Analyzing pre-scored results

`mood analyze` consumes one or more `results.jsonl` files (the format produced by `mood bench`) and re-runs the metric / figure step, optionally aggregating across multiple monitors.

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

The CLI is a thin wrapper around the `mood_bench` Python API. To plug in your own monitor, implement the `Pipeline` protocol — any callable that maps a list of conversation strings to a `(scores, metadata)` tuple — and hand it to `mood_bench()`:

```python
from mood_bench import mood_bench, GuardModelPipeline, load_tokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = load_tokenizer("shizwick/google-gemma-2-2b_guard")
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

## Dataset

Mood-Bench evaluates on conversations from the `[shizwick/mood-bench](https://huggingface.co/datasets/shizwick/mood-bench)` dataset on Hugging Face. Each row carries a `conversation`, a `domain`, and a `malign` label.


| Group                                                     | Domains                                                                                                                               |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **In-distribution (used to calibrate the safe baseline)** | `hh-rlhf-helpful`, `hh-rlhf-harmless`, `function-calling`                                                                             |
| **Out-of-distribution (unsafe behaviors)**                | `controlling`, `insecure-code`, `scheming`, `jailbroken`, `sycophantic`, `function-calling-missing`, `function-calling-inappropriate` |
| **Robustness probe**                                      | `swahili` (non-English variant)                                                                                                       |


The `id` group in any report contains all in-distribution conversations (safe + unsafe). Each remaining group pairs in-distribution *safe* samples with out-of-distribution *unsafe* samples for the named domain, so the reported AUROC and TPR@FPR isolate the monitor's behavior on that single OOD slice. The `overall` group is a macro-average across the per-domain groups.

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

If you run into a problem or have a question, please open an issue on the [GitHub tracker](https://github.com/shizwick/mood-bench/issues).