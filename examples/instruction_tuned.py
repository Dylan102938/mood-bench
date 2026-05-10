"""Evaluate mood-bench with :class:`InstructionTunedPipeline`.

Scores conversations using an instruction-tuned LLM judge.  The model is
prompted to rate each conversation's safety on a 0–100 scale.  Supports
both vLLM (preferred) and HuggingFace transformers backends; the backend
is selected automatically based on what is installed.

Usage:
    python examples/instruction_tuned.py \\
        --adapter-id shizwick/gemma-judge-lora
    python examples/instruction_tuned.py \\
        --model-id meta-llama/Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse

from utils import resolve_torch_dtype

from mood_bench.core import mood_bench
from mood_bench.data import EvalDataset
from mood_bench.pipeline.instruction_tuned import InstructionTunedPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        "--model_id",
        default=None,
        help="Full-weight instruction-tuned model. Mutually exclusive with --adapter-id.",
    )
    parser.add_argument(
        "--adapter-id",
        "--adapter_id",
        default=None,
        help="LoRA/PEFT adapter repo or path. Base model is inferred from the adapter config.",
    )
    parser.add_argument(
        "--grading-type",
        "--grading_type",
        choices=["alignment", "uncertainty"],
        default="alignment",
    )
    parser.add_argument("--num-few-shot", "--num_few_shot", type=int, default=0)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-retries", "--max_retries", type=int, default=3)
    parser.add_argument("--output-dir", "--output_dir", default="mood-bench-results")
    parser.add_argument("--use-mini", "--use_mini", action="store_true")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Subset of EvalDataset values to evaluate on. Defaults to all domains.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Dtype for model weights (e.g. bfloat16, float16, float32).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization (ignored for HF backend).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (auto-detected if unset).",
    )

    args = parser.parse_args()
    if args.model_id is None and args.adapter_id is None:
        parser.error("Provide at least one of --model-id or --adapter-id.")
    if args.model_id is not None and args.adapter_id is not None:
        parser.error("--model-id and --adapter-id are mutually exclusive.")
    return args


def main() -> None:
    args = parse_args()

    if args.adapter_id is not None:
        model_name = args.adapter_id
        is_lora_adapter = True
    else:
        model_name = args.model_id
        is_lora_adapter = False

    pipeline = InstructionTunedPipeline(
        model_name,
        is_lora_adapter=is_lora_adapter,
        grading_type=args.grading_type,
        num_few_shot=args.num_few_shot,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        torch_dtype=resolve_torch_dtype(args.dtype),
    )

    domains = [EvalDataset(d) for d in args.domains] if args.domains else None
    dataset = mood_bench(
        pipelines=pipeline,
        domains=domains,
        output_dir=args.output_dir,
        use_mini=args.use_mini,
        include_figures=True,
    )

    print(f"Scored {len(dataset)} samples across domains: {sorted(set(dataset['domain']))}")


if __name__ == "__main__":
    main()
