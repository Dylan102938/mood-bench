"""Shared helpers for example scripts."""

from __future__ import annotations

import torch as t

_DTYPE_ALIASES: dict[str, t.dtype] = {
    "auto": t.bfloat16,
    "bf16": t.bfloat16,
    "bfloat16": t.bfloat16,
    "fp16": t.float16,
    "float16": t.float16,
    "half": t.float16,
    "fp32": t.float32,
    "float32": t.float32,
    "float": t.float32,
}


def resolve_torch_dtype(name: str) -> t.dtype:
    key = name.strip().lower()
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    # Fall back to torch attribute lookup
    dtype = getattr(t, key, None)
    if isinstance(dtype, t.dtype):
        return dtype
    raise ValueError(
        f"Unknown torch dtype: {name!r}. Valid choices: {', '.join(sorted(_DTYPE_ALIASES))}"
    )
