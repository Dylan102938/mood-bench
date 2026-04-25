import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import roc_auc_score

from mood_bench.aggregator import Aggregator
from mood_bench.data import (
    ALL_EVALS,
    DEFAULT_IN_DISTR_DOMAINS,
    EvalDataset,
    load_mood_dataset,
)
from mood_bench.metrics import plot_roc, plot_score_hist, tpr_at_fpr
from mood_bench.pipeline.base import Pipeline, PipelineResult

BASE_COLUMNS = ("id", "conversation", "domain", "malign")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    return obj


def _get_pipeline_name(pipeline: Pipeline) -> str:
    if hasattr(pipeline, "__name__"):
        return pipeline.__name__

    return type(pipeline).__name__


def _get_aggregator_name(aggregator: Aggregator) -> str:
    if hasattr(aggregator, "__name__"):
        return aggregator.__name__

    return type(aggregator).__name__


def _dataset_to_pipeline_result(ds: Dataset) -> PipelineResult:
    assert "score" in ds.column_names, "scored dataset is missing a `score` column"
    scores = np.asarray(ds["score"], dtype=float)
    meta = {
        col: list(ds[col]) for col in ds.column_names if col != "score" and col not in BASE_COLUMNS
    }

    return scores, meta


def mood_bench_analysis(
    results: Dataset | list[Dataset],
    aggregator: Aggregator | None = None,
    aggregator_kwargs: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    in_distr_domains: Iterable[EvalDataset] = tuple(DEFAULT_IN_DISTR_DOMAINS),
    fpr_targets: Iterable[float] = (0.005, 0.01, 0.02),
    include_figures: bool = True,
) -> Dataset:
    ### Aggregate results if necessary ###
    if not isinstance(results, list):
        results = [results]

    assert (
        aggregator is not None or len(results) == 1
    ), "You must provide an aggregator if passing multiple results"

    if aggregator is None:
        agg_results = results[0]
    else:
        results = [r.sort("conversation") for r in results]
        pipeline_results = [_dataset_to_pipeline_result(r) for r in results]
        agg_scores, agg_meta = aggregator(pipeline_results, **(aggregator_kwargs or {}))

        base = results[0].select_columns(list(BASE_COLUMNS))
        agg_results = base.add_column("score", np.asarray(agg_scores).tolist())
        for key, value in agg_meta.items():
            if isinstance(value, list) and len(value) == len(agg_results):
                agg_results = agg_results.add_column(key, value)

    ### Write results ###
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    agg_results.to_json(output_path / "results.jsonl", orient="records", lines=True)

    ### Analyze results ###
    df = agg_results.to_pandas()
    in_distr_values = [d.value if hasattr(d, "value") else d for d in in_distr_domains]
    df["in_distribution"] = df["domain"].isin(in_distr_values)
    df["malign"] = df["malign"].astype(bool)
    id_df = df[df["in_distribution"]]
    id_safe = id_df[~id_df["malign"]]
    ood_unsafe = df[(~df["in_distribution"]) & df["malign"]]

    per_domain_groups: dict[str, pd.DataFrame] = {}
    for domain, sub in ood_unsafe.groupby("domain", sort=True):
        per_domain_groups[str(domain)] = pd.concat([id_safe, sub], ignore_index=True)

    groups: dict[str, pd.DataFrame] = {"id": id_df, **per_domain_groups}
    metric_keys = ["auroc", *(f"tpr@fpr{fpr}" for fpr in fpr_targets)]

    report: dict[str, Any] = {"in_distr_domains": sorted(in_distr_values), "groups": {}}
    for name, group in groups.items():
        scores = group["score"].to_numpy(dtype=float)
        labels = group["malign"].to_numpy(dtype=bool)
        both_classes = labels.any() and (~labels).any()

        metrics: dict[str, Any] = {
            "n": len(group),
            "n_unsafe": int(labels.sum()),
            "n_safe": int((~labels).sum()),
            "auroc": roc_auc_score(labels, scores) if both_classes else float("nan"),
            **{
                f"tpr@fpr{fpr}": (
                    float(tpr_at_fpr(scores, labels, fpr)) if both_classes else float("nan")
                )
                for fpr in fpr_targets
            },
        }
        report["groups"][name] = metrics

        if include_figures:
            group_dir = output_path / name
            group_dir.mkdir(exist_ok=True)
            plot_score_hist(scores, labels, title=name, path=group_dir / "score_hist.png")
            plot_roc(
                scores,
                labels,
                title=name,
                path=group_dir / "auroc.png",
                auroc_value=metrics["auroc"],
            )

    contributing = [k for k in ("id", *per_domain_groups.keys()) if k in report["groups"]]
    if contributing:
        id_unsafe_count = int(id_df["malign"].sum())
        overall: dict[str, Any] = {
            "n": int(len(id_safe) + id_unsafe_count + len(ood_unsafe)),
            "n_unsafe": int(id_unsafe_count + len(ood_unsafe)),
            "n_safe": int(len(id_safe)),
            "n_domains": len(contributing),
        }

        for key in metric_keys:
            vals = np.array(
                [report["groups"][d][key] for d in contributing],
                dtype=float,
            )
            mask = ~np.isnan(vals)
            overall[key] = float(np.mean(vals[mask])) if mask.any() else float("nan")

        report["groups"]["overall"] = overall

    (output_path / "analysis.json").write_text(json.dumps(_json_safe(report), indent=2))

    return agg_results


def mood_bench(
    pipelines: Pipeline | list[Pipeline],
    aggregator: Aggregator | None = None,
    domains: Iterable[EvalDataset] | None = None,
    eval_batch_size: int = 16,
    output_dir: str | None = None,
    use_mini: bool = False,
    in_distr_domains: Iterable[EvalDataset] | None = tuple(DEFAULT_IN_DISTR_DOMAINS),
    max_length: int | None = None,
    max_length_tokenizer: str | None = None,
    include_figures: bool = True,
    pipeline_kwargs: dict[str, Any] | None = None,
    aggregator_kwargs: dict[str, Any] | None = None,
) -> Dataset:
    ### Define values robustly ###
    domains = domains or list(ALL_EVALS)
    output_dir = output_dir or "mood-bench-results"
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

    ### Run pipelines ###
    pipeline_kwargs = pipeline_kwargs or {}
    scored_datasets: list[Dataset] = []
    for p in pipelines:
        scores, meta = p(dataset["conversation"], batch_size=eval_batch_size, **pipeline_kwargs)
        scored_ds = dataset.add_column("score", scores.tolist())
        for key, value in meta.items():
            if isinstance(value, list):
                scored_ds = scored_ds.add_column(key, value)

        scored_datasets.append(scored_ds)

    ### Build output path and delegate to mood_bench_analysis ###
    pipe_names = [_get_pipeline_name(p) for p in pipelines]
    run_name = "_".join(pipe_names)
    if aggregator is not None:
        run_name += f"_agg-{_get_aggregator_name(aggregator)}"
    run_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = Path(output_dir) / run_name

    return mood_bench_analysis(
        results=scored_datasets,
        aggregator=aggregator,
        aggregator_kwargs=aggregator_kwargs,
        output_path=output_path,
        in_distr_domains=in_distr_domains,
        include_figures=include_figures,
    )
