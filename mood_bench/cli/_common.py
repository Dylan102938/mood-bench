from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch as t

    from mood_bench.data import EvalDataset

_DTYPE_ALIASES: dict[str, str] = {
    "auto": "bfloat16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


def resolve_torch_dtype(name: str) -> "t.dtype":
    import torch as t

    key = name.strip().lower()
    canonical = _DTYPE_ALIASES.get(key, key)
    dtype = getattr(t, canonical, None)
    if isinstance(dtype, t.dtype):
        return dtype
    raise ValueError(
        f"Unknown torch dtype: {name!r}. Valid choices: {', '.join(sorted(_DTYPE_ALIASES))}"
    )


def parse_domains(raw: list[str] | None) -> "list[EvalDataset] | None":
    from mood_bench.data import EvalDataset

    if raw is None:
        return None

    return [EvalDataset(d) for d in raw]


def infer_adapter_num_labels(adapter_id: str) -> int | None:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])
    return None


def add_common_args(parser: argparse.ArgumentParser, base_model_required: bool = False) -> None:
    parser.add_argument(
        "--model-id",
        "--model_id",
        help="Base model path or id",
        required=base_model_required,
    )
    parser.add_argument(
        "--adapter-id",
        "--adapter_id",
        default=None,
        help="Optional LoRA/PEFT adapter to merge on top of --model-id.",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=4)
    parser.add_argument("--max-length", "--max_length", type=int, default=2048)
    parser.add_argument("--output-dir", "--output_dir", default="mood-bench-results")
    parser.add_argument("--use-mini", "--use_mini", action="store_true")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help=(
            "Subset of EvalDataset values to evaluate on (e.g. 'hh-rlhf-helpful "
            "function-calling-missing'). Defaults to all domains."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Dtype for model weights (e.g. bfloat16, float16, float32). Default: bfloat16.",
    )
    parser.add_argument(
        "--no-figures",
        "--no_figures",
        action="store_true",
        default=False,
        help="Run pipeline and analysis only, do not generate accompanying figures.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print additional informational messages (cache hits, GPU detection, etc.).",
    )
