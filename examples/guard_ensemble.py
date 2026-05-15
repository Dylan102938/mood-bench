"""Evaluate mood-bench with an ensemble of guard pipelines.

Each particle is passed to ``mood_bench`` as its own pipeline function that
loads its model, scores, then frees the weights + CUDA cache -- so only one
particle's worth of weights lives in memory at a time.

Particles are described in a JSON config with a ``particles`` list. Each entry
requires ``model_id`` and optionally takes ``adapter_id``, ``tokenizer_id``,
``num_labels``, ``unsafe_label_index``, ``device``, ``name``.

Usage:
    python examples/guard_ensemble.py --config ensemble.json --aggregate mean
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch as t
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification
from utils import resolve_torch_dtype

from mood_bench.aggregator import Aggregator, MeanAggregate, MinAggregate
from mood_bench.core import mood_bench
from mood_bench.data import EvalDataset
from mood_bench.pipeline.base import Pipeline, PipelineResult
from mood_bench.pipeline.guard import GuardModelPipeline
from mood_bench.tokenize import load_tokenizer

AGGREGATORS: dict[str, Aggregator] = {"mean": MeanAggregate(), "min": MinAggregate()}


def infer_adapter_num_labels(adapter_id: str) -> int | None:
    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])
    return None


def make_guard_particle(
    spec: dict[str, Any],
    fallback_device: t.device,
    dtype: t.dtype,
) -> Pipeline:
    spec = dict(spec)
    adapter_id = spec.get("adapter_id")
    spec["device"] = t.device(spec["device"]) if spec.get("device") else fallback_device
    spec["tokenizer_id"] = spec.get("tokenizer_id") or adapter_id or spec["model_id"]
    if spec.get("num_labels") is None and adapter_id:
        spec["num_labels"] = infer_adapter_num_labels(adapter_id)

    def run_particle(samples: list[str], **kwargs: Any) -> PipelineResult:
        run_particle.__name__ = spec.get("name") or "GuardModelPipeline"
        print(f"[{run_particle.__name__}] loading {spec['model_id']} on {spec['device']}")

        tokenizer = load_tokenizer(spec["tokenizer_id"])

        load_kwargs: dict[str, Any] = {"dtype": dtype}
        if spec.get("num_labels") is not None:
            load_kwargs["num_labels"] = spec["num_labels"]

        model = AutoModelForSequenceClassification.from_pretrained(spec["model_id"], **load_kwargs)
        model = model.to(spec["device"]).eval()
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if adapter_id:
            model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload()

        try:
            return GuardModelPipeline(
                model,
                tokenizer,
                unsafe_label_index=spec.get("unsafe_label_index", 1),
            )(samples, **kwargs)
        finally:
            del model, tokenizer
            gc.collect()
            if t.cuda.is_available():
                t.cuda.empty_cache()

    run_particle.__name__ = spec.get("name") or "GuardModelPipeline"
    return run_particle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--aggregate", choices=sorted(AGGREGATORS), default="mean")
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
    parser.add_argument("--device", default=None, help="Fallback device for particles.")
    parser.add_argument(
        "--predict-safe",
        "--predict_safe",
        action="store_true",
        default=False,
        help=(
            "If set, scores are flipped so that higher still means 'more unsafe'. "
            "Use when your model's target class actually represents 'safe'."
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
    fallback_device = t.device(args.device or ("cuda" if t.cuda.is_available() else "cpu"))

    particles = json.loads(args.config.read_text()).get("particles", [])
    if not particles:
        raise ValueError(f"No particles found in {args.config}")

    dtype = resolve_torch_dtype(args.dtype)
    domains = [EvalDataset(d) for d in args.domains] if args.domains else None
    dataset = mood_bench(
        pipelines=[make_guard_particle(p, fallback_device, dtype) for p in particles],
        aggregator=AGGREGATORS[args.aggregate],
        domains=domains,
        eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        max_length=args.max_length,
        predict_safe=args.predict_safe,
    )

    print(f"Scored {len(dataset)} samples with {len(particles)}-model {args.aggregate}-ensemble")


if __name__ == "__main__":
    main()
