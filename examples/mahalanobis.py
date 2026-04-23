"""Evaluate mood-bench with :class:`MahalanobisPipeline`.

Fits a Gaussian over pooled hidden states of safe in-distribution
training conversations, then scores test samples by their Mahalanobis
distance to that distribution (larger = more anomalous).

Optionally layer a LoRA / PEFT adapter on top of the base encoder by
passing ``--adapter-id``; the adapter is merged into the base weights
before stats are fit.

Fitted stats are cached to disk (keyed by model/adapter/pooling/max-samples)
and reused on subsequent runs. Pass ``--refit-stats`` to ignore the cache
and recompute. The cache directory can be set via ``--stats-cache-dir`` or
the ``MOOD_BENCH_STATS_CACHE`` environment variable.

Usage:
    python examples/mahalanobis.py --model-id gpt2 --pooling mean --use-mini
    python examples/mahalanobis.py \\
        --model-id google/gemma-2-2b \\
        --adapter-id shizwick/google-gemma-2-2b_guard \\
        --pooling cls
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch as t
from peft import PeftModel
from transformers import AutoModel

from mood_bench.core import mood_bench
from mood_bench.pipeline.mahalanobis import (
    MahalanobisPipeline,
    PoolingStrategy,
    get_stats_for_model,
)
from mood_bench.tokenize import load_tokenizer

POOLING_CHOICES: tuple[PoolingStrategy, ...] = ("cls", "mean", "max")
DEFAULT_STATS_CACHE_DIR = Path(
    os.environ.get(
        "MOOD_BENCH_STATS_CACHE", Path.home() / ".cache" / "mood-bench" / "mahalanobis-stats"
    )
)


def _stats_cache_path(
    cache_dir: Path,
    model_id: str,
    adapter_id: str | None,
    pooling: PoolingStrategy,
    max_samples: int | None,
) -> Path:
    key = "|".join(
        [
            model_id,
            adapter_id or "",
            pooling,
            str(max_samples) if max_samples is not None else "all",
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    slug = f"{model_id.replace('/', '__')}_{pooling}_{digest}.pt"
    return cache_dir / slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", "--model_id", help="Base encoder.")
    parser.add_argument(
        "--adapter-id",
        "--adapter_id",
        default=None,
        help="Optional LoRA/PEFT adapter to merge on top of --model-id.",
    )
    parser.add_argument("--pooling", choices=POOLING_CHOICES, default="cls")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=4)
    parser.add_argument("--max-length", "--max_length", type=int, default=1024)
    parser.add_argument("--output-dir", "--output_dir", default="mood-bench-results")
    parser.add_argument("--use-mini", "--use_mini", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--stats-batch-size",
        "--stats_batch_size",
        type=int,
        default=16,
        help="Batch size used while fitting the safe-sample Gaussian.",
    )
    parser.add_argument(
        "--stats-max-samples",
        "--stats_max_samples",
        type=int,
        default=None,
        help="Cap on the number of safe training samples used to fit stats (all if unset).",
    )
    parser.add_argument(
        "--stats-cache-dir",
        "--stats_cache_dir",
        type=Path,
        default=DEFAULT_STATS_CACHE_DIR,
        help=(
            "Directory used to cache fitted Mahalanobis stats. "
            "Override with $MOOD_BENCH_STATS_CACHE."
        ),
    )
    parser.add_argument(
        "--refit-stats",
        "--refit_stats",
        action="store_true",
        help="Ignore any cached stats and recompute via get_stats_for_model.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = t.device(args.device or ("cuda" if t.cuda.is_available() else "cpu"))
    adapter_str = f" + adapter {args.adapter_id}" if args.adapter_id else ""
    print(f"Loading encoder {args.model_id}{adapter_str} on {device}")

    ### Load model and tokenizer ###
    tokenizer = load_tokenizer(args.adapter_id or args.model_id)
    model = AutoModel.from_pretrained(args.model_id, torch_dtype="auto")
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.adapter_id is not None:
        model = PeftModel.from_pretrained(model, args.adapter_id).merge_and_unload()

    model = model.to(device)
    model.eval()

    ### Fit stats (with on-disk cache) ###
    cache_path = _stats_cache_path(
        args.stats_cache_dir,
        args.model_id,
        args.adapter_id,
        args.pooling,
        args.stats_max_samples,
    )

    if cache_path.exists() and not args.refit_stats:
        print(f"Loading cached Mahalanobis stats from {cache_path}")
        stats = t.load(cache_path, map_location="cpu", weights_only=True)
    else:
        print(
            f"Fitting Mahalanobis stats ({args.pooling} pooling, "
            f"max_samples={args.stats_max_samples})"
        )
        stats = get_stats_for_model(
            model,
            tokenizer,
            pooling_strategy=args.pooling,
            batch_size=args.stats_batch_size,
            max_samples=args.stats_max_samples,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        t.save({k: v.detach().cpu() for k, v in stats.items()}, cache_path)
        print(f"Saved Mahalanobis stats to {cache_path}")

    ### Run pipeline ###
    dataset = mood_bench(
        pipelines=MahalanobisPipeline(
            model,
            tokenizer,
            mean=stats["mean"],
            inv_cov=stats["inv_cov"],
            pooling_strategy=args.pooling,
        ),
        eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        max_length=args.max_length,
        run_analysis=True,
    )

    print(f"Scored {len(dataset)} samples across domains: {sorted(set(dataset['domain']))}")


if __name__ == "__main__":
    main()
