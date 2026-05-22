from __future__ import annotations

import argparse
from pathlib import Path

AGGREGATOR_NAMES: tuple[str, ...] = ("lambda", "mean", "min")


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "analyze",
        help="Generate a mood-bench report from pre-scored JSONL files.",
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
        choices=sorted(AGGREGATOR_NAMES),
        default=None,
        help="How to combine multiple scored runs. Required when more than one file is passed.",
    )
    parser.add_argument(
        "--in-distr-domains",
        "--in_distr_domains",
        nargs="+",
        default=None,
        help="Override the in-distribution domains used for the 'id' group.",
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
        help="Skip score-histogram and ROC plots.",
    )
    parser.add_argument(
        "--predict-safe",
        "--predict_safe",
        action="store_true",
        help=(
            "If set, treats input 'score' columns as safety scores (higher = more safe) "
            "and inverts them so higher = more unsafe for AUROC."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print additional informational messages.",
    )

    lambda_group = parser.add_argument_group(
        "lambda aggregator options",
        "Only used when --aggregator=lambda.",
    )
    lambda_group.add_argument(
        "--anchor-index",
        "--anchor_index",
        type=int,
        default=0,
        help="Index of the anchor pipeline in the results list. Defaults to 0.",
    )
    lambda_group.add_argument(
        "--fpr-threshold",
        "--fpr_threshold",
        type=float,
        default=0.01,
        help=(
            "FPR threshold used when fitting the lambda coefficient. "
            "Distinct from --fpr-targets, which only affects the report. Defaults to 0.01."
        ),
    )
    lambda_group.add_argument(
        "--lambda-min-exp",
        "--lambda_min_exp",
        type=float,
        default=-2.0,
        help="Lower bound (log10) of the lambda search grid. Defaults to -2.0.",
    )
    lambda_group.add_argument(
        "--lambda-max-exp",
        "--lambda_max_exp",
        type=float,
        default=2.0,
        help="Upper bound (log10) of the lambda search grid. Defaults to 2.0.",
    )
    lambda_group.add_argument(
        "--n-lambdas",
        "--n_lambdas",
        type=int,
        default=21,
        help="Number of grid points between --lambda-min-exp and --lambda-max-exp. Defaults to 21.",
    )

    parser.set_defaults(func=run)


def _verify_and_load_dataset(path: Path):  # -> Dataset
    from datasets import load_dataset

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

    if any(required not in ds.column_names for required in ("id", "conversation", "domain")):
        raise ValueError(f"{path} needs to have `id`, `conversation`, and `domain` columns.")

    return ds


def run(args: argparse.Namespace) -> None:
    from mood_bench._output import info, print_report_table, warn
    from mood_bench.aggregator import LambdaAggregate, MeanAggregate, MinAggregate
    from mood_bench.core import mood_bench_analysis
    from mood_bench.data import DEFAULT_IN_DISTR_DOMAINS, EvalDataset

    if len(args.results) > 1 and args.aggregator is None:
        raise SystemExit("--aggregator is required when more than one results file is passed.")
    if len(args.results) == 1 and args.aggregator is not None:
        warn("--aggregator is ignored with a single results file.")

    datasets = [_verify_and_load_dataset(p) for p in args.results]
    in_distr_domains = (
        [EvalDataset(d) for d in args.in_distr_domains]
        if args.in_distr_domains
        else tuple(DEFAULT_IN_DISTR_DOMAINS)
    )

    if args.aggregator == "lambda":
        aggregator = LambdaAggregate(
            anchor_index=args.anchor_index,
            in_distr_domains=in_distr_domains,
            fpr_threshold=args.fpr_threshold,
            lambda_min_exp=args.lambda_min_exp,
            lambda_max_exp=args.lambda_max_exp,
            n_lambdas=args.n_lambdas,
        )
    elif args.aggregator == "min":
        aggregator = MinAggregate()
    elif args.aggregator == "mean":
        aggregator = MeanAggregate()
    else:
        aggregator = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, report = mood_bench_analysis(
        results=datasets,
        aggregator=aggregator,
        in_distr_domains=in_distr_domains,
        fpr_targets=tuple(args.fpr_targets),
        include_figures=not args.no_figures,
        output_path=args.output_dir,
        predict_safe=args.predict_safe,
    )

    info(f"Wrote report to {args.output_dir}")
    print_report_table(report)
