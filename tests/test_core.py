from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest
from datasets import Dataset

from mood_bench import core as mood_core
from mood_bench.aggregator import min_aggregate
from mood_bench.data import EvalDataset


def _make_dataset(n_per_domain: int, domains: Sequence[EvalDataset]) -> Dataset:
    rows: list[dict[str, Any]] = []
    row_id = 0
    for domain in domains:
        for i in range(n_per_domain):
            rows.append(
                {
                    "id": f"id-{row_id}",
                    "conversation": f"{domain.value}-{i}",
                    "domain": domain.value,
                    "malign": i % 2 == 0,
                }
            )
            row_id += 1
    return Dataset.from_list(rows)


@pytest.fixture
def patched_core(monkeypatch: pytest.MonkeyPatch) -> Callable[[Dataset], None]:
    """Patch Hub/disk side-effects of ``mood_bench.core`` for hermetic testing."""

    def _install(dataset: Dataset) -> None:
        monkeypatch.setattr(mood_core, "load_mood_dataset", lambda *_, **__: dataset)
        monkeypatch.setattr(Dataset, "to_json", lambda self, *_, **__: None)

    return _install


def _named_pipeline(
    name: str, fn: Callable[[list[str]], tuple[np.ndarray, dict[str, Any]]]
) -> Callable[..., tuple[np.ndarray, dict[str, Any]]]:
    def _call(samples: list[str], **_: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return fn(samples)

    _call.__name__ = name
    return _call


def test_mood_bench_single_pipeline_adds_score_column(
    patched_core: Callable[[Dataset], None], tmp_path: Path
) -> None:
    domains = [EvalDataset.HH_HELPFUL, EvalDataset.HH_HARMLESS]
    ds = _make_dataset(3, domains)
    patched_core(ds)

    pipeline = _named_pipeline("myPipe", lambda s: (np.arange(len(s), dtype=float), {"tag": "p1"}))

    out = mood_core.mood_bench(
        pipelines=pipeline,
        domains=domains,
        output_dir=str(tmp_path),
        include_figures=False,
    )

    assert "score" in out.column_names
    assert len(out["score"]) == len(ds)
    assert out["score"] == list(range(len(ds)))


def test_mood_bench_aggregator_combines_results(
    patched_core: Callable[[Dataset], None], tmp_path: Path
) -> None:
    domains = [EvalDataset.HH_HELPFUL]
    ds = _make_dataset(4, domains)
    patched_core(ds)

    p_high = _named_pipeline("hi", lambda s: (np.full(len(s), 5.0), {}))
    p_low = _named_pipeline("lo", lambda s: (np.full(len(s), 2.0), {}))

    out = mood_core.mood_bench(
        pipelines=[p_high, p_low],
        aggregator=min_aggregate,
        domains=domains,
        output_dir=str(tmp_path),
        include_figures=False,
    )

    assert np.allclose(np.asarray(out["score"]), 2.0)


def test_mood_bench_use_mini_truncates_per_domain(
    patched_core: Callable[[Dataset], None], tmp_path: Path
) -> None:
    domains = [EvalDataset.HH_HELPFUL, EvalDataset.HH_HARMLESS]
    ds = _make_dataset(150, domains)
    patched_core(ds)

    pipeline = _named_pipeline("mini", lambda s: (np.zeros(len(s)), {}))

    out = mood_core.mood_bench(
        pipelines=pipeline,
        domains=domains,
        output_dir=str(tmp_path),
        use_mini=True,
        include_figures=False,
    )

    assert len(out) <= 100 * len(domains)
    for domain in domains:
        rows = [row for row in out if row["domain"] == domain.value]
        assert len(rows) <= 100


def test_mood_bench_output_dir_name_contains_pipeline_and_aggregator(
    patched_core: Callable[[Dataset], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domains = [EvalDataset.HH_HELPFUL]
    patched_core(_make_dataset(2, domains))

    captured: dict[str, Path] = {}

    original_to_json = Dataset.to_json

    def capture_to_json(self: Dataset, path: Any, *args: Any, **kwargs: Any) -> Any:
        captured["path"] = Path(path)
        return None

    monkeypatch.setattr(Dataset, "to_json", capture_to_json, raising=False)
    try:
        p_anchor = _named_pipeline("anchorPipe", lambda s: (np.zeros(len(s)), {}))
        p_aux = _named_pipeline("auxPipe", lambda s: (np.zeros(len(s)), {}))

        mood_core.mood_bench(
            pipelines=[p_anchor, p_aux],
            aggregator=min_aggregate,
            domains=domains,
            output_dir=str(tmp_path),
            include_figures=False,
        )
    finally:
        monkeypatch.setattr(Dataset, "to_json", original_to_json, raising=False)

    assert "path" in captured
    # results.jsonl is written directly into the run dir: {run_dir}/results.jsonl.
    run_dir = captured["path"].parent.name
    assert "anchorPipe" in run_dir
    assert "auxPipe" in run_dir
    assert "min_aggregate" in run_dir
