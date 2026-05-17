from __future__ import annotations

import argparse

import torch as t
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification

from mood_bench import GuardModelPipeline, load_tokenizer, mood_bench


def _infer_num_labels(adapter_id: str) -> int | None:
    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Directory to write the per-run results.jsonl and analysis.json under.",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer("shizwick/google-gemma-2-2b_guard")
    num_labels = _infer_num_labels("shizwick/google-gemma-2-2b_guard")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer.name_or_path,
        dtype=t.bfloat16,
        **({"num_labels": num_labels} if num_labels is not None else {}),
    )
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    device = "cuda" if t.cuda.is_available() else "cpu"
    model = PeftModel.from_pretrained(base_model, "shizwick/google-gemma-2-2b_guard")
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
        predict_safe=False,
    )


if __name__ == "__main__":
    main()
