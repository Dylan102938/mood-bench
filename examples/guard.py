"""Evaluate mood-bench with :class:`GuardModelPipeline`.

Wraps any Hugging Face ``AutoModelForSequenceClassification`` checkpoint
as a mood-bench pipeline. Pass ``--adapter-id`` to layer a LoRA / PEFT
classification adapter on top of the base model; when doing so, the
adapter's ``num_labels`` is auto-inferred from its safetensors head so
the base model is instantiated with a compatible classifier.

Usage:
    python examples/guard.py --model-id s-nlp/roberta_toxicity_classifier
    python examples/guard.py \\
        --model-id google/gemma-2-2b \\
        --adapter-id shizwick/google-gemma-2-2b_guard \\
        --predict-unsafe false
"""

from __future__ import annotations

import argparse

import torch as t
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification
from utils import resolve_torch_dtype

from mood_bench.core import mood_bench
from mood_bench.data import EvalDataset
from mood_bench.pipeline.guard import GuardModelPipeline
from mood_bench.tokenize import load_tokenizer


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def infer_adapter_num_labels(adapter_id: str) -> int | None:
    """Read the adapter's saved classifier head to recover ``num_labels``."""
    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", "--model_id", help="Base classifier.")
    parser.add_argument(
        "--adapter-id",
        "--adapter_id",
        default=None,
        help="Optional LoRA/PEFT adapter to merge on top of --model-id.",
    )
    parser.add_argument(
        "--num-labels",
        "--num_labels",
        type=int,
        default=None,
        help="Override the base model's num_labels (auto-inferred from adapter if unset).",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=4)
    parser.add_argument("--max-length", "--max_length", type=int, default=1024)
    parser.add_argument("--output-dir", "--output_dir", default="mood-bench-results")
    parser.add_argument("--use-mini", "--use_mini", action="store_true")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help=(
            "Subset of EvalDataset values to evaluate on (e.g. 'hh-rlhf-helpful "
            "function-calling-missing'). Defaults to all domains."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--device-map",
        "--device_map",
        default=None,
        help=(
            "Optional ``device_map`` for ``from_pretrained`` (e.g. 'auto'). When set, "
            "the model is placed/sharded via Accelerate and we skip the follow-up "
            "``model.to(device)`` call. Use this for models too large for a single GPU."
        ),
    )
    parser.add_argument(
        "--unsafe-label-index",
        "--unsafe_label_index",
        type=int,
        default=1,
        help=(
            "Index of the 'unsafe' class in the classifier's logits. "
            "Only used for multi-label heads; single-logit heads are read directly."
        ),
    )
    parser.add_argument(
        "--predict-unsafe",
        "--predict_unsafe",
        type=parse_bool,
        default=True,
        help=(
            "If false, the pipeline flips the score so that higher still means 'more unsafe'. "
            "Set this to false when your model's target class actually represents 'safe'."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Dtype for model weights (e.g. bfloat16, float16, float32). Default: bfloat16.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = t.device(args.device or ("cuda" if t.cuda.is_available() else "cpu"))
    num_labels = args.num_labels
    if num_labels is None and args.adapter_id is not None:
        num_labels = infer_adapter_num_labels(args.adapter_id)

    adapter_str = f" + adapter {args.adapter_id}" if args.adapter_id else ""
    labels_str = f", num_labels={num_labels}" if num_labels is not None else ""
    print(f"Loading guard classifier {args.model_id}{adapter_str} on {device}{labels_str}")

    ### Load model and tokenizer ###
    tokenizer = load_tokenizer(args.adapter_id or args.model_id)
    from_pretrained_kwargs: dict[str, object] = {"dtype": resolve_torch_dtype(args.dtype)}
    if num_labels is not None:
        from_pretrained_kwargs["num_labels"] = num_labels
    if args.device_map is not None:
        from_pretrained_kwargs["device_map"] = args.device_map
        from_pretrained_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        **from_pretrained_kwargs,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.adapter_id is not None:
        model = PeftModel.from_pretrained(model, args.adapter_id).merge_and_unload()
    if args.device_map is None:
        model = model.to(device)

    model.eval()

    ### Run pipeline ###
    domains = [EvalDataset(d) for d in args.domains] if args.domains else None
    dataset = mood_bench(
        pipelines=GuardModelPipeline(
            model,
            tokenizer,
            unsafe_label_index=args.unsafe_label_index,
            predict_unsafe=args.predict_unsafe,
        ),
        domains=domains,
        eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        max_length=args.max_length,
        include_figures=True,
    )

    print(f"Scored {len(dataset)} samples across domains: {sorted(set(dataset['domain']))}")


if __name__ == "__main__":
    main()
