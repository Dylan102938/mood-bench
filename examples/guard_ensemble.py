from __future__ import annotations

import argparse
import gc
from typing import Iterable

import torch as t
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification

from mood_bench import (
    GuardModelPipeline,
    MinAggregate,
    Pipeline,
    PipelineResult,
    load_tokenizer,
    mood_bench,
)


def _infer_num_labels(adapter_id: str) -> int | None:
    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])
    return None


def guard_pipeline(adapter_id: str, device: str) -> Pipeline:
    def run(samples: list[str], **kwargs) -> PipelineResult:
        tokenizer = load_tokenizer(adapter_id)
        num_labels = _infer_num_labels(adapter_id)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            tokenizer.name_or_path,
            dtype=t.bfloat16,
            **({"num_labels": num_labels} if num_labels is not None else {}),
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        model = PeftModel.from_pretrained(base_model, adapter_id)
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
    adapter_ids: Iterable[str] = (
        "shizwick/ensemble_google_gemma-2-9b-particle-00",
        "shizwick/ensemble_google_gemma-2-9b-particle-01",
        "shizwick/ensemble_google_gemma-2-9b-particle-02",
        "shizwick/ensemble_google_gemma-2-9b-particle-03",
        "shizwick/ensemble_google_gemma-2-9b-particle-04",
    )

    mood_bench(
        pipelines=[guard_pipeline(adapter_id, device) for adapter_id in adapter_ids],
        aggregator=MinAggregate(),
        output_dir=args.output_dir,
        eval_batch_size=8,
        max_length=2048,
        predict_safe=False,
    )


if __name__ == "__main__":
    main()
