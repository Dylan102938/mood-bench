from __future__ import annotations

import argparse

from mood_bench.cli._common import add_common_args


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("perplexity", help="Perplexity (NLL) pipeline.")
    parser.add_argument(
        "--device-map",
        "--device_map",
        default=None,
        help="Optional device_map for from_pretrained (e.g. 'auto').",
    )
    parser.add_argument(
        "--outlier-z-threshold",
        "--outlier_z_threshold",
        type=float,
        default=3.0,
        help="Record per-sample tokens whose excess-surprise z-score exceeds this value.",
    )
    add_common_args(parser)

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import torch as t
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from mood_bench._output import print_report_table
    from mood_bench.cli._common import parse_domains, resolve_torch_dtype
    from mood_bench.core import mood_bench
    from mood_bench.pipeline.perplexity import PerplexityPipeline
    from mood_bench.tokenize import load_tokenizer

    ### Define defaults ###
    default_device = "cuda" if t.cuda.is_available() else "cpu"
    device = t.device(args.device or default_device)

    ### Load tokenizer + model ###
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

    ### Run mood_bench ###
    domains = parse_domains(args.domains)
    threshold = args.outlier_z_threshold if args.outlier_z_threshold >= 0 else None
    _, report = mood_bench(
        pipelines=PerplexityPipeline(model, tokenizer, outlier_z_threshold=threshold),
        domains=domains,
        eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        max_length=args.max_length,
        include_figures=not args.no_figures,
    )

    print_report_table(report, title=f"Perplexity · {args.adapter_id or args.model_id}")
