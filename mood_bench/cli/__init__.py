from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    from mood_bench import __version__

    parser = argparse.ArgumentParser(
        prog="mood",
        description="Mood-Bench: a multi-domain out-of-distribution safety benchmark for LLMs.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- mood bench <pipeline> ---
    bench_parser = subparsers.add_parser("bench", help="Run a benchmark pipeline.")
    bench_sub = bench_parser.add_subparsers(dest="pipeline")

    from mood_bench.cli.guard import build_parser as guard_bp
    from mood_bench.cli.instruction_tuned import build_parser as it_bp
    from mood_bench.cli.mahalanobis import build_parser as mahal_bp
    from mood_bench.cli.perplexity import build_parser as ppl_bp

    guard_bp(bench_sub)
    ppl_bp(bench_sub)
    mahal_bp(bench_sub)
    it_bp(bench_sub)

    # --- mood analyze ---
    from mood_bench.cli.analyze import build_parser as analyze_bp

    analyze_bp(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "bench" and getattr(args, "pipeline", None) is None:
        bench_parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)
