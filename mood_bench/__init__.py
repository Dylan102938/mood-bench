from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mood_bench.aggregator import (
        Aggregator,
        LambdaAggregate,
        MeanAggregate,
        MinAggregate,
    )
    from mood_bench.cli._common import resolve_torch_dtype
    from mood_bench.core import mood_bench, mood_bench_analysis
    from mood_bench.data import (
        ALL_EVALS,
        DEFAULT_IN_DISTR_DOMAINS,
        EvalDataset,
        load_mood_dataset,
    )
    from mood_bench.metrics import plot_roc, plot_score_hist, tpr_at_fpr
    from mood_bench.pipeline import (
        GuardModelPipeline,
        InstructionTunedPipeline,
        MahalanobisPipeline,
        PerplexityPipeline,
        Pipeline,
        PipelineResult,
    )
    from mood_bench.tokenize import load_tokenizer

try:
    __version__ = _pkg_version("mood-bench")
except PackageNotFoundError:
    # Running from a source checkout that hasn't been pip-installed.
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    ### Core analysis flow ###
    "mood_bench",
    "mood_bench_analysis",
    ### Data ###
    "ALL_EVALS",
    "DEFAULT_IN_DISTR_DOMAINS",
    "EvalDataset",
    "load_mood_dataset",
    ### Pipelines ###
    "GuardModelPipeline",
    "InstructionTunedPipeline",
    "MahalanobisPipeline",
    "PerplexityPipeline",
    "Pipeline",
    "PipelineResult",
    ### Aggregators ###
    "Aggregator",
    "LambdaAggregate",
    "MeanAggregate",
    "MinAggregate",
    #### Metrics ###
    "plot_roc",
    "plot_score_hist",
    "tpr_at_fpr",
    ### Utils ###
    "load_tokenizer",
    "resolve_torch_dtype",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "mood_bench": ("mood_bench.core", "mood_bench"),
    "mood_bench_analysis": ("mood_bench.core", "mood_bench_analysis"),
    "ALL_EVALS": ("mood_bench.data", "ALL_EVALS"),
    "DEFAULT_IN_DISTR_DOMAINS": ("mood_bench.data", "DEFAULT_IN_DISTR_DOMAINS"),
    "EvalDataset": ("mood_bench.data", "EvalDataset"),
    "load_mood_dataset": ("mood_bench.data", "load_mood_dataset"),
    "GuardModelPipeline": ("mood_bench.pipeline", "GuardModelPipeline"),
    "InstructionTunedPipeline": ("mood_bench.pipeline", "InstructionTunedPipeline"),
    "MahalanobisPipeline": ("mood_bench.pipeline", "MahalanobisPipeline"),
    "PerplexityPipeline": ("mood_bench.pipeline", "PerplexityPipeline"),
    "Pipeline": ("mood_bench.pipeline", "Pipeline"),
    "PipelineResult": ("mood_bench.pipeline", "PipelineResult"),
    "Aggregator": ("mood_bench.aggregator", "Aggregator"),
    "LambdaAggregate": ("mood_bench.aggregator", "LambdaAggregate"),
    "MeanAggregate": ("mood_bench.aggregator", "MeanAggregate"),
    "MinAggregate": ("mood_bench.aggregator", "MinAggregate"),
    "plot_roc": ("mood_bench.metrics", "plot_roc"),
    "plot_score_hist": ("mood_bench.metrics", "plot_score_hist"),
    "tpr_at_fpr": ("mood_bench.metrics", "tpr_at_fpr"),
    "load_tokenizer": ("mood_bench.tokenize", "load_tokenizer"),
    "resolve_torch_dtype": ("mood_bench.cli._common", "resolve_torch_dtype"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val

    raise AttributeError(f"module 'mood_bench' has no attribute {name!r}")
