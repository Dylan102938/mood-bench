from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset

from mood_bench import (
    DEFAULT_IN_DISTR_DOMAINS,
    Aggregator,
    EvalDataset,
    LambdaAggregate,
    MeanAggregate,
    MinAggregate,
    mood_bench_analysis,
)

AGGREGATORS: dict[str, Aggregator] = {
    "min": MinAggregate(),
    "mean": MeanAggregate(),
    "lambda": LambdaAggregate(),
}


def load_scored_jsonl(path: Path) -> Dataset:
    """Load ``path`` as a HF :class:`Dataset` and normalize label columns.

    Accepts either a ``malign`` column (0/1 or bool) or a ``safe`` column
    (0/1 or bool); in the latter case ``malign`` is derived as ``1 - safe``.
    Score-convention handling (``predict_safe``) is delegated to
    :func:`mood_bench.core.mood_bench_analysis`.
    """

    ds = load_dataset("json", data_files=str(path), split="train")

    if "malign" not in ds.column_names:
        if "safe" not in ds.column_names:
            raise ValueError(
                f"{path} is missing both 'malign' and 'safe' columns; "
                "cannot determine per-row labels."
            )
        ds = ds.map(lambda ex: {"malign": int(not bool(ex["safe"]))})

    if "score" not in ds.column_names:
        raise ValueError(f"{path} is missing a 'score' column.")

    for required in ("id", "conversation", "domain"):
        if required not in ds.column_names:
            raise ValueError(f"{path} is missing required column '{required}'.")

    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="One or more scored JSONL files. Multiple files require --aggregator.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where results.jsonl, analysis.json, and figures are written.",
    )
    parser.add_argument(
        "--aggregator",
        choices=sorted(AGGREGATORS),
        default=None,
        help="How to combine multiple scored runs. Required when more than one file is passed.",
    )
    parser.add_argument(
        "--in-distr-domains",
        "--in_distr_domains",
        nargs="+",
        default=None,
        help=(
            "Override the in-distribution domains used when computing the 'id' group. "
            "Defaults to the standard mood-bench in-distribution set."
        ),
    )
    parser.add_argument(
        "--fpr-targets",
        "--fpr_targets",
        nargs="+",
        type=float,
        default=(0.005, 0.01, 0.02),
        help="FPR thresholds at which to report TPR.",
    )
    parser.add_argument(
        "--no-figures",
        "--no_figures",
        action="store_true",
        help="Skip score-histogram and ROC plots; only emit results.jsonl and analysis.json.",
    )
    parser.add_argument(
        "--predict-safe",
        "--predict_safe",
        action="store_true",
        help=(
            "Treat input 'score' columns as safety scores (higher = more safe). "
            "They are negated inside mood_bench_analysis so all written artifacts "
            "use the canonical higher = more malign convention."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.results) > 1 and args.aggregator is None:
        raise SystemExit("--aggregator is required when more than one results file is passed.")
    if len(args.results) == 1 and args.aggregator is not None:
        print("Note: --aggregator is ignored with a single results file.")

    datasets = [load_scored_jsonl(p) for p in args.results]

    in_distr_domains = (
        [EvalDataset(d) for d in args.in_distr_domains]
        if args.in_distr_domains
        else tuple(DEFAULT_IN_DISTR_DOMAINS)
    )

    if args.aggregator == "lambda":
        aggregator = LambdaAggregate(in_distr_domains=in_distr_domains)
    elif args.aggregator == "min":
        aggregator = MinAggregate()
    elif args.aggregator == "mean":
        aggregator = MeanAggregate()
    else:
        aggregator = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mood_bench_analysis(
        results=datasets if len(datasets) > 1 else datasets[0],
        aggregator=aggregator,
        output_path=args.output_dir,
        in_distr_domains=in_distr_domains,
        fpr_targets=tuple(args.fpr_targets),
        include_figures=not args.no_figures,
        predict_safe=args.predict_safe,
    )

    print(f"Wrote mood-bench report to {args.output_dir}")
    print(f"  - {args.output_dir / 'results.jsonl'}")
    print(f"  - {args.output_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
