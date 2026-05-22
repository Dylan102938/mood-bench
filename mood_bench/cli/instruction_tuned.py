from __future__ import annotations

import argparse

from mood_bench.cli._common import add_common_args


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "instruction-tuned",
        help="Instruction-tuned LLM judge pipeline.",
    )
    parser.add_argument(
        "--grading-type",
        "--grading_type",
        choices=["alignment", "uncertainty"],
        default="alignment",
    )
    parser.add_argument("--num-few-shot", "--num_few_shot", type=int, default=0)
    parser.add_argument(
        "--icl-seed",
        "--icl_seed",
        type=int,
        default=42,
        help="Seed used to sample in-context negative examples.",
    )
    parser.add_argument(
        "--default-score",
        "--default_score",
        type=float,
        default=0.0,
        help="Fallback score assigned when the judge fails to produce a parseable number.",
    )
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-retries", "--max_retries", type=int, default=3)
    parser.add_argument(
        "--max-lora-rank",
        "--max_lora_rank",
        type=int,
        default=64,
        help="vLLM max_lora_rank for the LoRA adapter (ignored when no adapter is used).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization (ignored for HF backend).",
    )
    parser.add_argument(
        "--enforce-eager",
        "--enforce_eager",
        action="store_true",
        default=False,
        help="Enforce eager mode for vLLM backend.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (auto-detected if unset).",
    )
    add_common_args(parser, base_model_required=True)

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from mood_bench._output import print_report_table
    from mood_bench.cli._common import parse_domains, resolve_torch_dtype
    from mood_bench.core import mood_bench
    from mood_bench.pipeline.instruction_tuned import InstructionTunedPipeline

    ### Run mood_bench ###
    domains = parse_domains(args.domains)
    _, report = mood_bench(
        pipelines=InstructionTunedPipeline(
            args.model_id,
            adapter_id=args.adapter_id,
            grading_type=args.grading_type,
            num_few_shot=args.num_few_shot,
            icl_malign_examples=None,
            icl_seed=args.icl_seed,
            max_retries=args.max_retries,
            default_score=args.default_score,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_lora_rank=args.max_lora_rank,
            enforce_eager=args.enforce_eager,
            torch_dtype=resolve_torch_dtype(args.dtype),
        ),
        domains=domains,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        include_figures=not args.no_figures,
        predict_safe=True,
    )

    print_report_table(report, title=f"Instruction-Tuned · {args.adapter_id or args.model_id}")
