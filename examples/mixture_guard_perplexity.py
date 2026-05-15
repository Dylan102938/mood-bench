"""Evaluate mood-bench with a lambda-aggregated (guard + perplexity) mixture.

Both models are lazily loaded: we materialise the guard classifier, run it,
free the weights, then do the same for the perplexity LM. Scores are combined
by coordinate-descent :func:`~mood_bench.aggregator.lambda_aggregate` over the
anchor pipeline, using the mood-bench test split's in-distribution rows to
fit coefficients.

Usage:
    python examples/mixture_guard_perplexity.py \\
        --guard-model-id s-nlp/roberta_toxicity_classifier \\
        --ppl-model-id gpt2 \\
        --anchor guard
"""

from __future__ import annotations

import argparse
import gc
from typing import Any

import torch as t
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from utils import resolve_torch_dtype

from mood_bench.aggregator import LambdaAggregate
from mood_bench.core import mood_bench
from mood_bench.data import (
    EvalDataset,
)
from mood_bench.pipeline.base import Pipeline, PipelineResult
from mood_bench.pipeline.guard import GuardModelPipeline
from mood_bench.pipeline.perplexity import PerplexityPipeline
from mood_bench.tokenize import load_tokenizer


def _free_cuda() -> None:
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def _infer_adapter_num_labels(adapter_id: str) -> int | None:
    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])

    return None


def make_guard_pipeline(
    model_id: str,
    adapter_id: str | None,
    device: t.device,
    *,
    num_labels: int | None = None,
    unsafe_label_index: int = 1,
    dtype: t.dtype = t.bfloat16,
) -> Pipeline:
    if num_labels is None and adapter_id:
        num_labels = _infer_adapter_num_labels(adapter_id)

    def run(samples: list[str], **kwargs: Any) -> PipelineResult:
        print(f"[{run.__name__}] loading {model_id} on {device}")
        tokenizer = load_tokenizer(adapter_id or model_id)
        load_kwargs: dict[str, Any] = {"dtype": dtype}
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(model_id, **load_kwargs)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if adapter_id:
            model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload()
        model = model.to(device).eval()
        try:
            return GuardModelPipeline(
                model,
                tokenizer,
                unsafe_label_index=unsafe_label_index,
            )(samples, **kwargs)
        finally:
            del model, tokenizer
            _free_cuda()

    run.__name__ = f"guard_{model_id.split('/')[-1]}"
    return run


def make_perplexity_pipeline(
    model_id: str,
    adapter_id: str | None,
    device: t.device,
    *,
    outlier_z_threshold: float | None = 3.0,
    dtype: t.dtype = t.bfloat16,
) -> Pipeline:
    def run(samples: list[str], **kwargs: Any) -> PipelineResult:
        print(f"[{run.__name__}] loading {model_id} on {device}")
        tokenizer = load_tokenizer(adapter_id or model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if adapter_id:
            model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload()
        model = model.to(device).eval()
        try:
            return PerplexityPipeline(model, tokenizer, outlier_z_threshold=outlier_z_threshold)(
                samples, **kwargs
            )
        finally:
            del model, tokenizer
            _free_cuda()

    run.__name__ = f"ppl_{model_id.split('/')[-1]}"
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-model-id", "--guard_model_id", required=True)
    parser.add_argument("--guard-adapter-id", "--guard_adapter_id", default=None)
    parser.add_argument("--guard-num-labels", "--guard_num_labels", type=int, default=None)
    parser.add_argument(
        "--guard-unsafe-label-index", "--guard_unsafe_label_index", type=int, default=1
    )
    parser.add_argument(
        "--guard-predict-safe",
        "--guard_predict_safe",
        action="store_true",
        default=False,
    )
    parser.add_argument("--ppl-model-id", "--ppl_model_id", required=True)
    parser.add_argument("--ppl-adapter-id", "--ppl_adapter_id", default=None)
    parser.add_argument(
        "--ppl-outlier-z-threshold", "--ppl_outlier_z_threshold", type=float, default=3.0
    )
    parser.add_argument("--fpr-threshold", "--fpr_threshold", type=float, default=0.01)
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
            "function-calling-missing'). Defaults to all domains. Note: the "
            "lambda-coefficient fit still uses the full in-distribution set."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Dtype for model weights (e.g. bfloat16, float16, float32). Default: bfloat16.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = t.device(args.device or ("cuda" if t.cuda.is_available() else "cpu"))

    dtype = resolve_torch_dtype(args.dtype)
    threshold = args.ppl_outlier_z_threshold if args.ppl_outlier_z_threshold >= 0 else None
    guard_fn = make_guard_pipeline(
        args.guard_model_id,
        args.guard_adapter_id,
        device,
        num_labels=args.guard_num_labels,
        unsafe_label_index=args.guard_unsafe_label_index,
        dtype=dtype,
    )
    ppl_fn = make_perplexity_pipeline(
        args.ppl_model_id,
        args.ppl_adapter_id,
        device,
        outlier_z_threshold=threshold,
        dtype=dtype,
    )

    pipelines = [guard_fn, ppl_fn] if args.anchor == "guard" else [ppl_fn, guard_fn]
    domains = [EvalDataset(d) for d in args.domains] if args.domains else None
    dataset = mood_bench(
        pipelines=pipelines,
        aggregator=LambdaAggregate(
            anchor_index=0,
            fpr_threshold=args.fpr_threshold,
        ),
        domains=domains,
        eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        max_length=args.max_length,
        predict_safe=[args.guard_predict_safe, False],
    )

    print(
        f"Scored {len(dataset)} samples across domains: {sorted(set(dataset['domain']))} "
        f"(anchor={args.anchor}, fpr={args.fpr_threshold})"
    )


if __name__ == "__main__":
    main()
