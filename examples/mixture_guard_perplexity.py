from __future__ import annotations

import argparse
import gc

import torch as t
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from mood_bench import (
    GuardModelPipeline,
    LambdaAggregate,
    PerplexityPipeline,
    Pipeline,
    PipelineResult,
    load_tokenizer,
    mood_bench,
)
from mood_bench.cli._common import infer_adapter_num_labels

MODEL_ID = "google/gemma-2-2b"
GUARD_ADAPTER_ID = "mood-bench/gemma-2-2b-guard"
PERPLEXITY_ADAPTER_ID = "mood-bench/gemma-2-2b-causal-lm"


def guard_pipeline(device: str) -> Pipeline:
    def run(samples: list[str], **kwargs) -> PipelineResult:
        tokenizer = load_tokenizer(GUARD_ADAPTER_ID)
        num_labels = infer_adapter_num_labels(GUARD_ADAPTER_ID)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            dtype=t.bfloat16,
            **({"num_labels": num_labels} if num_labels is not None else {}),
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        model = PeftModel.from_pretrained(base_model, GUARD_ADAPTER_ID).merge_and_unload()
        model = model.to(device)
        model.eval()

        pipe = GuardModelPipeline(model, tokenizer, unsafe_label_index=1)
        result = pipe(samples, **kwargs)

        del pipe, model, base_model, tokenizer
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()

        return result

    return run


def perplexity_pipeline(device: str) -> Pipeline:
    def run(samples: list[str], **kwargs) -> PipelineResult:
        tokenizer = load_tokenizer(PERPLEXITY_ADAPTER_ID)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=t.bfloat16)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        model = PeftModel.from_pretrained(base_model, PERPLEXITY_ADAPTER_ID).merge_and_unload()
        model = model.to(device)
        model.eval()

        pipe = PerplexityPipeline(model, tokenizer, outlier_z_threshold=3.0)
        result = pipe(samples, **kwargs)

        del pipe, model, base_model, tokenizer
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()

        return result

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Directory to write the per-run results.jsonl and analysis.json under.",
    )
    args = parser.parse_args()

    device = "cuda" if t.cuda.is_available() else "cpu"
    mood_bench(
        pipelines=[guard_pipeline(device), perplexity_pipeline(device)],
        aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
        output_dir=args.output_dir,
        eval_batch_size=8,
        max_length=2048,
        predict_safe=[True, False],
    )


if __name__ == "__main__":
    main()
