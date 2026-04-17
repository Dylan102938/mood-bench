import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

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
from mood_bench.pipeline.base import Pipeline


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


def mood_bench_analysis(
    dataset: Dataset,
    output_path: Path,
    in_distr_domains: list[EvalDataset],
    fpr_targets: Iterable[float, ...] = (0.005, 0.01, 0.02),
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    id_values = {d.value for d in in_distr_domains}

    df = dataset.to_pandas()
    df["malign"] = df["malign"].astype(bool)
    df["score"] = df["score"].astype(float)
    df["in_distribution"] = df["domain"].isin(id_values)

    id_df = df[df["in_distribution"]]
    id_safe = id_df[~id_df["malign"]]
    ood_unsafe = df[(~df["in_distribution"]) & df["malign"]]

    groups: dict[str, pd.DataFrame] = {
        "id": id_df,
        "overall": pd.concat([id_safe, ood_unsafe], ignore_index=True),
    }
    for domain, sub in ood_unsafe.groupby("domain", sort=True):
        groups[domain] = pd.concat([id_safe, sub], ignore_index=True)

    report: dict[str, Any] = {"in_distr_domains": sorted(id_values), "groups": {}}
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

    (output_path / "analysis.json").write_text(json.dumps(_json_safe(report), indent=2))


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

    dataset.to_json(output_path / "out.jsonl", orient="records", lines=True)
    if run_analysis:
        mood_bench_analysis(dataset, output_path, in_distr_domains)

    return dataset
