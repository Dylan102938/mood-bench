"""Evaluate mood-bench with :class:`PerplexityPipeline`.

The perplexity pipeline scores each conversation by the average
token-level negative log-likelihood under a causal LM. Samples that the
model finds surprising (e.g. jailbreaks, weird formatting) tend to get
higher scores, making this a useful unsupervised signal.

Optionally layer a LoRA / PEFT adapter on top of the base model by
passing ``--adapter-id``; the adapter is merged into the base weights
before evaluation.

Usage:
    python examples/perplexity.py --model-id gpt2 --use-mini
    python examples/perplexity.py \\
        --model-id google/gemma-2-2b \\
        --adapter-id shizwick/google-gemma-2-2b_causal-lm
"""

from __future__ import annotations

import argparse

import torch as t
from peft import PeftModel
from transformers import AutoModelForCausalLM
from utils import resolve_torch_dtype

from mood_bench.core import mood_bench
from mood_bench.data import EvalDataset
from mood_bench.pipeline.perplexity import PerplexityPipeline
from mood_bench.tokenize import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", "--model_id", help="Base causal LM.")
    parser.add_argument(
        "--adapter-id",
        "--adapter_id",
        default=None,
        help="Optional LoRA/PEFT adapter to merge on top of --model-id.",
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
        "--outlier-z-threshold",
        "--outlier_z_threshold",
        type=float,
        default=3.0,
        help=(
            "Record per-sample tokens whose excess-surprise z-score exceeds this value. "
            "Pass a negative number to disable thresholding and record everything."
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
    threshold = args.outlier_z_threshold if args.outlier_z_threshold >= 0 else None

    adapter_str = f" + adapter {args.adapter_id}" if args.adapter_id else ""
    print(f"Loading causal LM {args.model_id}{adapter_str} on {device}")

    ### Load model and tokenizer ###
    tokenizer = load_tokenizer(args.adapter_id or args.model_id)
    from_pretrained_kwargs: dict[str, object] = {"dtype": resolve_torch_dtype(args.dtype)}
    if args.device_map is not None:
        from_pretrained_kwargs["device_map"] = args.device_map
        from_pretrained_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **from_pretrained_kwargs)
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
        pipelines=PerplexityPipeline(
            model,
            tokenizer,
            outlier_z_threshold=threshold,
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
