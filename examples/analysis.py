"""Generate a mood-bench report from one or more pre-scored JSONL files.

Each input JSONL must contain at least the columns ``id``, ``conversation``,
``domain``, ``score``, and a per-row ``malign`` (or ``safe``) label. ``safe`` is
automatically converted to ``malign = 1 - safe``. Any extra columns are kept
and treated as pipeline metadata.

By default ``score`` is interpreted as a malign-ness score (higher = more
malign). Pass ``--predict-safe`` if your scores instead predict safety
(higher = more safe); the score column will be negated inside
:func:`mood_bench.core.mood_bench_analysis` so all downstream artifacts
(``results.jsonl``, ``analysis.json``, figures) use the canonical
"higher = more malign" convention.

Examples:
    # Single scored run -> writes results.jsonl + analysis.json (+ figures)
    python examples/analysis.py mood-bench-results/gemma-9b-mahalanobis.jsonl \\
        --output-dir mood-bench-reports/gemma-9b-mahalanobis

    # Scores are 'higher = safer' (e.g. log-likelihood of safety) -> negate them
    python examples/analysis.py safety-scores.jsonl \\
        --predict-safe \\
        --output-dir mood-bench-reports/safety-scores

    # Aggregate two scored runs and report on the combined score
    python examples/analysis.py run-a.jsonl run-b.jsonl \\
        --aggregator mean \\
        --output-dir mood-bench-reports/ensemble
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset

from mood_bench.aggregator import (
    Aggregator,
    lambda_aggregate,
    mean_aggregate,
    min_aggregate,
)
from mood_bench.core import mood_bench_analysis
from mood_bench.data import DEFAULT_IN_DISTR_DOMAINS, EvalDataset

AGGREGATORS: dict[str, Aggregator] = {
    "min": min_aggregate,
    "mean": mean_aggregate,
    "lambda": lambda_aggregate,
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
    aggregator = AGGREGATORS[args.aggregator] if args.aggregator else None

    in_distr_domains = (
        [EvalDataset(d) for d in args.in_distr_domains]
        if args.in_distr_domains
        else tuple(DEFAULT_IN_DISTR_DOMAINS)
    )

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
