# Mood-Bench

A multi-domain out-of-distribution safety benchmark for LLMs. Mood-Bench evaluates whether safety monitors can detect misaligned, jailbroken, sycophantic, scheming, and otherwise unsafe model behaviour across diverse conversation domains.

## Installation

```bash
pip install mood-bench

# With vLLM support (recommended for instruction-tuned pipeline):
pip install mood-bench[vllm]
```

## Python API

```python
from mood_bench import mood_bench, GuardModelPipeline, load_tokenizer
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta_toxicity_classifier")
tokenizer = load_tokenizer("s-nlp/roberta_toxicity_classifier")

# Returns (scored_dataset, report). Pass output_dir=... to also persist
# results.jsonl, analysis.json, and figures to disk.
results, report = mood_bench(
    pipelines=GuardModelPipeline(model, tokenizer),
    use_mini=True,
)
print(report["groups"]["overall"])
```

Run analysis on pre-scored results:

```python
from mood_bench import mood_bench_analysis
from datasets import load_dataset

ds = load_dataset("json", data_files="results.jsonl", split="train")
mood_bench_analysis(results=ds, output_path="my-report")
```

## CLI

Mood-Bench ships a `mood` command with two top-level subcommands: `bench` (run a pipeline) and `analyze` (generate reports from scored JSONL).

```bash
# Run a guard classifier
mood bench guard --model-id s-nlp/roberta_toxicity_classifier --use-mini

# Run a perplexity pipeline
mood bench perplexity --model-id gpt2 --use-mini

# Run a Mahalanobis-distance pipeline
mood bench mahalanobis --model-id gpt2 --pooling mean --use-mini

# Run an instruction-tuned LLM judge
mood bench instruction-tuned --adapter-id shizwick/gemma-judge-lora

# Run a guard-model ensemble
mood bench guard-ensemble --config ensemble.json --aggregate mean

# Run a guard + perplexity lambda mixture
mood bench mixture --guard-model-id s-nlp/roberta_toxicity_classifier --ppl-model-id gpt2

# Analyze pre-scored results
mood analyze results.jsonl --output-dir my-report
mood analyze run-a.jsonl run-b.jsonl --aggregator mean --output-dir ensemble-report
```

Use `mood --help`, `mood bench --help`, or `mood bench <pipeline> --help` for full option details.

## Examples

The `examples/` directory contains standalone scripts that demonstrate how to wire custom pipelines and aggregators into `mood_bench` and `mood_bench_analysis`. These are meant as starting points for your own experiments -- copy one and adapt it to your model.

## Dataset

Mood-Bench evaluates on conversations from the [shizwick/mood-bench](https://huggingface.co/datasets/shizwick/mood-bench) dataset on Hugging Face, covering domains including:

- **In-distribution (safe):** hh-rlhf-helpful, hh-rlhf-harmless, function-calling
- **Out-of-distribution (unsafe):** controlling, insecure-code, scheming, jailbroken, sycophantic, function-calling-missing, function-calling-inappropriate
