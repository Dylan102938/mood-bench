from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets

from mood_bench.aggregator import Aggregator
from mood_bench.data import (
    ALL_EVALS,
    DEFAULT_IN_DISTR_DOMAINS,
    EvalDataset,
    load_mood_dataset,
)
from mood_bench.pipeline.base import Pipeline


def _get_pipeline_name(pipeline: Pipeline) -> str:
    if hasattr(pipeline, "__name__"):
        return pipeline.__name__

    return type(pipeline).__name__


def _get_aggregator_name(aggregator: Aggregator) -> str:
    if hasattr(aggregator, "__name__"):
        return aggregator.__name__

    return type(aggregator).__name__


def mood_bench_analysis(
    pipeline_result: Dataset | list[Dataset],
    output_path: Path,
    in_distr_domains: list[EvalDataset],
) -> None: ...


def mood_bench(
    pipelines: Pipeline | list[Pipeline],
    aggregator: Aggregator | None = None,
    domains: list[EvalDataset] | None = None,
    eval_batch_size: int = 16,
    output_dir: str | None = None,
    use_mini: bool = False,
    in_distr_domains: list[EvalDataset] | None = None,
    max_length: int | None = None,
    max_length_tokenizer: str | None = None,
    run_analysis: bool = True,
    **pipe_kwargs: Any,
) -> Dataset:
    ### Define values robustly ###
    domains = domains or list(ALL_EVALS)
    output_dir = output_dir or "mood-bench-results"
    in_distr_domains = in_distr_domains or list(DEFAULT_IN_DISTR_DOMAINS)
    if not isinstance(pipelines, list):
        pipelines = [pipelines]
    if len(pipelines) > 1:
        assert (
            aggregator is not None
        ), "You must provide an aggregator if passing multiple pipelines"

    assert len(pipelines) > 0

    ### Load eval dataset ###
    dataset = load_mood_dataset(
        "test",
        domains=domains,
        max_length=max_length,
        max_length_tokenizer=max_length_tokenizer,
    )
    if use_mini:
        mini_ds_list: list[Dataset] = []
        for domain in domains:
            ds = dataset.filter(lambda x: x["domain"] == domain.value)
            ds = ds.shuffle().select(range(min(len(ds), 100)))
            mini_ds_list.append(ds)

        dataset = concatenate_datasets(mini_ds_list)

    assert len(dataset) > 0, "No samples loaded"

    ### Run pipelines and aggregate results ###
    results = [
        p(
            dataset["conversation"],
            batch_size=eval_batch_size,
            **pipe_kwargs,
        )
        for p in pipelines
    ]
    if aggregator is not None:
        output, meta = aggregator(results)
    else:
        output, meta = results[0]

    ### Export results ###
    dataset = dataset.add_column("score", output.tolist())
    for key, value in meta.items():
        if isinstance(value, list):
            dataset = dataset.add_column(key, value)

    pipeline_names = "_".join([_get_pipeline_name(p) for p in pipelines])
    agg_name = f"_agg-{_get_aggregator_name(aggregator)}" if aggregator is not None else ""
    output_path = (
        Path(output_dir) / f"{pipeline_names}{agg_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    dataset.to_json(output_path / "results.jsonl", orient="records", lines=True)

    if run_analysis:
        mood_bench_analysis(results, output_path, in_distr_domains)

    return dataset
