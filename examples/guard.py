from __future__ import annotations

import argparse

import torch as t
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from mood_bench import GuardModelPipeline, load_tokenizer, mood_bench
from mood_bench.cli._common import infer_adapter_num_labels

MODEL_ID = "google/gemma-2-2b"
ADAPTER_ID = "mood-bench/gemma-2-2b-guard"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Directory to write the per-run results.jsonl and analysis.json under.",
    )
    args = parser.parse_args()

    device = "cuda" if t.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(ADAPTER_ID)
    num_labels = infer_adapter_num_labels(ADAPTER_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        dtype=t.bfloat16,
        **({"num_labels": num_labels} if num_labels is not None else {}),
    )
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, ADAPTER_ID).merge_and_unload()
    model = model.to(device)
    model.eval()

    mood_bench(
        pipelines=GuardModelPipeline(
            model,
            tokenizer,
            unsafe_label_index=1,
        ),
        output_dir=args.output_dir,
        eval_batch_size=8,
        max_length=2048,
        predict_safe=True,
    )


if __name__ == "__main__":
    main()
