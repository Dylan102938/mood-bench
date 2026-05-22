from __future__ import annotations

import argparse
import gc
from typing import Iterable

import torch as t
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from mood_bench import (
    GuardModelPipeline,
    MinAggregate,
    Pipeline,
    PipelineResult,
    load_tokenizer,
    mood_bench,
)
from mood_bench.cli._common import infer_adapter_num_labels

MODEL_ID = "google/gemma-2-9b"
ADAPTER_IDS: tuple[str, ...] = (
    "mood-bench/gemma-2-9b-guard-ensemble-p0",
    "mood-bench/gemma-2-9b-guard-ensemble-p1",
    "mood-bench/gemma-2-9b-guard-ensemble-p2",
    "mood-bench/gemma-2-9b-guard-ensemble-p3",
    "mood-bench/gemma-2-9b-guard-ensemble-p4",
)


def guard_pipeline(model_id: str, adapter_id: str, device: str) -> Pipeline:
    def run(samples: list[str], **kwargs) -> PipelineResult:
        tokenizer = load_tokenizer(adapter_id)
        num_labels = infer_adapter_num_labels(adapter_id)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            dtype=t.bfloat16,
            **({"num_labels": num_labels} if num_labels is not None else {}),
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        model = PeftModel.from_pretrained(base_model, adapter_id).merge_and_unload()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Directory to write the per-run results.jsonl and analysis.json under.",
    )
    args = parser.parse_args()

    device = "cuda" if t.cuda.is_available() else "cpu"
    adapter_ids: Iterable[str] = ADAPTER_IDS

    mood_bench(
        pipelines=[guard_pipeline(MODEL_ID, adapter_id, device) for adapter_id in adapter_ids],
        aggregator=MinAggregate(),
        output_dir=args.output_dir,
        eval_batch_size=8,
        max_length=2048,
        predict_safe=True,
    )


if __name__ == "__main__":
    main()
