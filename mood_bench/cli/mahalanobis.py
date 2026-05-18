from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch as t
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from mood_bench._output import print_report_table
from mood_bench.cli._common import (
    add_common_args,
    infer_adapter_num_labels,
    parse_domains,
    resolve_torch_dtype,
)
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


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("mahalanobis", help="Mahalanobis-distance pipeline.")
    parser.add_argument(
        "--num-labels",
        "--num_labels",
        type=int,
        default=None,
        help="Classification head size when --model-type=cls.",
    )
    parser.add_argument("--pooling", choices=POOLING_CHOICES, default="cls")
    parser.add_argument(
        "--device-map",
        "--device_map",
        default=None,
        help="Optional device_map for from_pretrained (e.g. 'auto').",
    )
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
        help="Cap on the number of safe training samples used to fit stats.",
    )
    parser.add_argument(
        "--stats-cache-dir",
        "--stats_cache_dir",
        type=Path,
        default=DEFAULT_STATS_CACHE_DIR,
        help="Directory used to cache fitted Mahalanobis stats.",
    )
    parser.add_argument(
        "--refit-stats",
        "--refit_stats",
        action="store_true",
        help="Ignore any cached stats and recompute.",
    )
    add_common_args(parser)

    parser.set_defaults(func=run)


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


def run(args: argparse.Namespace) -> None:
    ### Define defaults ###
    default_device = "cuda" if t.cuda.is_available() else "cpu"
    device = t.device(args.device or default_device)
    num_labels = args.num_labels
    if num_labels is None and args.adapter_id is not None:
        num_labels = infer_adapter_num_labels(args.adapter_id)

    ### Load tokenizer + model ###
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
        model = PeftModel.from_pretrained(model, args.adapter_id)
    if args.device_map is None:
        model = model.to(device)

    model.eval()

    ### Fit Mahalanobis stats ###
    cache_path = _stats_cache_path(
        args.stats_cache_dir,
        args.model_id,
        args.adapter_id,
        args.pooling,
        args.stats_max_samples,
    )

    if cache_path.exists() and not args.refit_stats:
        from mood_bench._output import info

        info(f"Loading cached Mahalanobis stats from {cache_path}")
        stats = t.load(cache_path, map_location="cpu", weights_only=True)
    else:
        from mood_bench._output import info

        info(
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
        t.save(
            {k: v.detach().cpu() for k, v in stats.items()},
            cache_path,
        )
        info(f"Saved Mahalanobis stats to {cache_path}")

    ### Run mood_bench ###
    domains = parse_domains(args.domains)
    _, report = mood_bench(
        pipelines=MahalanobisPipeline(
            model,
            tokenizer,
            mean=stats["mean"].to(device=device, dtype=t.float64),
            inv_cov=stats["inv_cov"].to(device=device, dtype=t.float64),
            pooling_strategy=args.pooling,
        ),
        domains=domains,
        eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        max_length=args.max_length,
        include_figures=not args.no_figures,
        predict_safe=False,
    )

    print_report_table(report, title=f"Mahalanobis · {args.adapter_id or args.model_id}")
