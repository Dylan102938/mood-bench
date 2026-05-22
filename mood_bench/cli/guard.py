from __future__ import annotations

import argparse

from mood_bench.cli._common import add_common_args


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("guard", help="Guard-model classifier pipeline.")
    parser.add_argument(
        "--num-labels",
        "--num_labels",
        type=int,
        default=None,
        help="Override the base model's num_labels (auto-inferred from adapter if unset).",
    )
    parser.add_argument(
        "--device-map",
        "--device_map",
        default=None,
        help="Optional device_map for from_pretrained (e.g. 'auto').",
    )
    parser.add_argument(
        "--unsafe-label-index",
        "--unsafe_label_index",
        type=int,
        default=1,
        help="Index of the 'unsafe' class in the classifier's logits.",
    )
    parser.add_argument(
        "--predict-safe",
        "--predict_safe",
        action="store_true",
        default=False,
        help=(
            "If set, treats the pipeline's scores as safety scores (higher = more safe) "
            "and inverts them so higher = more unsafe for AUROC."
        ),
    )
    add_common_args(parser)

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import torch as t
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    from mood_bench._output import print_report_table
    from mood_bench.cli._common import (
        infer_adapter_num_labels,
        parse_domains,
        resolve_torch_dtype,
    )
    from mood_bench.core import mood_bench
    from mood_bench.pipeline.guard import GuardModelPipeline
    from mood_bench.tokenize import load_tokenizer

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
        model = PeftModel.from_pretrained(model, args.adapter_id).merge_and_unload()
    if args.device_map is None:
        model = model.to(device)

    model.eval()

    ### Run mood_bench ###
    domains = parse_domains(args.domains)
    _, report = mood_bench(
        pipelines=GuardModelPipeline(model, tokenizer, args.unsafe_label_index),
        domains=domains,
        eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        max_length=args.max_length,
        include_figures=not args.no_figures,
        predict_safe=args.predict_safe,
    )

    print_report_table(report, title=f"Guard · {args.adapter_id or args.model_id}")
