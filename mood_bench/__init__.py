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

__version__ = "1.0.0"
__all__ = [
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
