"""CLI-level tests.

Tests are grouped into classes by concern:

* ``TestGuardCLI``   - ``mood bench guard`` happy paths & ``--predict-safe``
* ``TestPerplexityCLI`` - ``mood bench perplexity``
* ``TestMahalanobisCLI`` - ``mood bench mahalanobis``
* ``TestInstructionTunedCLI`` - ``mood bench instruction-tuned``
* ``TestAnalyzeCLI``  - ``mood analyze`` (single / multi-file, aggregators,
                         ``--predict-safe``, ``--in-distr-domains``, ``--fpr-targets``)
* ``TestOutputArtifacts`` - file-system side-effects (``results.jsonl``,
                            ``analysis.json``, figure directories)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
import torch as t
from datasets import Dataset

from mood_bench.cli import main as run_cli
from mood_bench.data import load_mood_dataset as _real_load

# ---------------------------------------------------------------------------
# Tiny real Llama-3.2 checkpoints (~MB each, fast CPU load).
# ---------------------------------------------------------------------------
CLS_MODEL = "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2"
LM_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"

# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------
_N = NormalDist()
_AUROC_ABS_TOL = 0.03
_TPR_ABS_TOL = 0.05
DEFAULT_FPR_TARGETS: tuple[float, ...] = (0.005, 0.01, 0.02)

ID_DOMAIN = "hh-rlhf-harmless"
OOD_DOMAINS = ("jailbroken", "controlling")
DOMAINS = (ID_DOMAIN, *OOD_DOMAINS)
_DOMAIN_CLI_ARGS = ["--domains", *DOMAINS]

_BASE_CLI_ARGS = [
    "--no-figures",
    "--device",
    "cpu",
    "--dtype",
    "float32",
    "--batch-size",
    "32",
    *_DOMAIN_CLI_ARGS,
]


@dataclass(frozen=True)
class DomainSpec:
    safe: tuple[float, float]  # (mu, sigma)
    unsafe: tuple[float, float]


# ---------------------------------------------------------------------------
# Label capture + dataset wrapper + stub pipeline
# ---------------------------------------------------------------------------


class LabelCapture:
    def __init__(self) -> None:
        self.conversations: list[str] = []
        self.domains: list[str] = []
        self.malign: list[int] = []

    def lookup(self) -> dict[str, tuple[str, int]]:
        return {c: (d, m) for c, d, m in zip(self.conversations, self.domains, self.malign)}


def _wrap_dataset_loader(monkeypatch: pytest.MonkeyPatch, capture: LabelCapture) -> None:
    def _wrapper(*args: Any, **kwargs: Any) -> Dataset:
        ds = _real_load(*args, **kwargs)
        capture.conversations = list(ds["conversation"])
        capture.domains = list(ds["domain"])
        capture.malign = [int(m) for m in ds["malign"]]
        return ds

    monkeypatch.setattr("mood_bench.core.load_mood_dataset", _wrapper)


def _make_fake_pipeline(
    capture: LabelCapture,
    specs: dict[str, DomainSpec],
    rng: np.random.Generator,
):
    class _StubPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, samples: list[str], **_: Any):
            lut = capture.lookup()
            scores = np.empty(len(samples), dtype=float)
            for i, conv in enumerate(samples):
                domain, mal = lut[conv]
                spec = specs[domain]
                mu, sigma = spec.unsafe if mal else spec.safe
                scores[i] = rng.normal(mu, sigma)

            return scores, {}

    return _StubPipeline


def _install_pipeline_stub(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_path: str,
    specs: dict[str, DomainSpec],
    rng: np.random.Generator,
) -> None:
    capture = LabelCapture()
    _wrap_dataset_loader(monkeypatch, capture)
    monkeypatch.setattr(pipeline_path, _make_fake_pipeline(capture, specs, rng))


# ---------------------------------------------------------------------------
# Report / metric helpers
# ---------------------------------------------------------------------------


def _read_report(output_dir: Path) -> dict:
    paths = list(output_dir.glob("*/analysis.json"))
    if not paths:
        path = output_dir / "analysis.json"
        assert path.exists(), f"no analysis.json under {output_dir}"
        return json.loads(path.read_text())
    assert len(paths) == 1, f"expected one analysis.json under {output_dir}, got {paths}"
    return json.loads(paths[0].read_text())


def _analytic_auroc(neg: tuple[float, float], pos: tuple[float, float]) -> float:
    mu_n, sig_n = neg
    mu_p, sig_p = pos
    return _N.cdf((mu_p - mu_n) / math.sqrt(sig_p**2 + sig_n**2))


def _analytic_tpr_at_fpr(
    neg: tuple[float, float],
    pos: tuple[float, float],
    fpr: float,
) -> float:
    mu_n, sig_n = neg
    mu_p, sig_p = pos
    tau = mu_n + sig_n * _N.inv_cdf(1 - fpr)
    return 1 - _N.cdf((tau - mu_p) / sig_p)


def _expected_groups(
    specs: dict[str, DomainSpec],
    *,
    flip: bool = False,
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    def _flip(d: tuple[float, float]) -> tuple[float, float]:
        return (-d[0], d[1]) if flip else d

    id_spec = specs[ID_DOMAIN]
    groups: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
        "id": (_flip(id_spec.safe), _flip(id_spec.unsafe)),
    }
    for ood in OOD_DOMAINS:
        groups[ood] = (_flip(id_spec.safe), _flip(specs[ood].unsafe))
    return groups


def _assert_metrics_match_analytic(
    report: dict,
    specs: dict[str, DomainSpec],
    *,
    flip: bool = False,
    fpr_targets: tuple[float, ...] = DEFAULT_FPR_TARGETS,
) -> None:
    groups = _expected_groups(specs, flip=flip)
    aurocs: list[float] = []

    for name, (neg, pos) in groups.items():
        expected_auroc = _analytic_auroc(neg, pos)
        actual_auroc = report["groups"][name]["auroc"]
        assert actual_auroc == pytest.approx(
            expected_auroc, abs=_AUROC_ABS_TOL
        ), f"{name}: AUROC {actual_auroc:.3f} != {expected_auroc:.3f} ± {_AUROC_ABS_TOL}"
        aurocs.append(expected_auroc)
        for fpr in fpr_targets:
            expected_tpr = _analytic_tpr_at_fpr(neg, pos, fpr)
            actual_tpr = report["groups"][name][f"tpr@fpr{fpr}"]
            assert actual_tpr == pytest.approx(
                expected_tpr, abs=_TPR_ABS_TOL
            ), f"{name}: tpr@fpr{fpr} {actual_tpr:.3f} != {expected_tpr:.3f} ± {_TPR_ABS_TOL}"

    expected_overall = sum(aurocs) / len(aurocs)
    actual_overall = report["groups"]["overall"]["auroc"]
    assert actual_overall == pytest.approx(
        expected_overall, abs=_AUROC_ABS_TOL
    ), f"overall AUROC {actual_overall:.3f} != {expected_overall:.3f} ± {_AUROC_ABS_TOL}"


# ---------------------------------------------------------------------------
# Scored-JSONL generation helper (for analyze tests)
# ---------------------------------------------------------------------------


def _write_scored_jsonl(
    path: Path,
    specs: dict[str, DomainSpec],
    rng: np.random.Generator,
    n_per_class: int = 400,
) -> None:
    rows: list[dict[str, Any]] = []
    idx = 0
    for domain, spec in specs.items():
        for malign in (0, 1):
            mu, sigma = spec.unsafe if malign else spec.safe
            for _ in range(n_per_class):
                rows.append(
                    {
                        "id": f"row-{idx}",
                        "conversation": f"<conv {idx}>",
                        "domain": domain,
                        "malign": malign,
                        "score": float(rng.normal(mu, sigma)),
                    }
                )
                idx += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


# ===================================================================
# Test classes
# ===================================================================


class TestGuardCLI:
    SPECS: dict[str, DomainSpec] = {
        ID_DOMAIN: DomainSpec(safe=(0.40, 0.07), unsafe=(0.70, 0.07)),
        "jailbroken": DomainSpec(safe=(0.40, 0.07), unsafe=(0.75, 0.07)),
        "controlling": DomainSpec(safe=(0.40, 0.07), unsafe=(0.72, 0.07)),
    }

    def test_metrics(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.guard.GuardModelPipeline",
            self.SPECS,
            np.random.default_rng(seed=42),
        )
        run_cli(
            [
                "bench",
                "guard",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(tmp_path),
                *_BASE_CLI_ARGS,
            ]
        )
        _assert_metrics_match_analytic(_read_report(tmp_path), self.SPECS, flip=False)

    def test_predict_safe_flips_scores(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        inverted: dict[str, DomainSpec] = {
            ID_DOMAIN: DomainSpec(safe=(0.70, 0.07), unsafe=(0.40, 0.07)),
            "jailbroken": DomainSpec(safe=(0.70, 0.07), unsafe=(0.30, 0.07)),
            "controlling": DomainSpec(safe=(0.70, 0.07), unsafe=(0.33, 0.07)),
        }
        no_flip = tmp_path / "no_flip"
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.guard.GuardModelPipeline",
            inverted,
            np.random.default_rng(seed=7),
        )
        run_cli(
            [
                "bench",
                "guard",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(no_flip),
                *_BASE_CLI_ARGS,
            ]
        )
        _assert_metrics_match_analytic(_read_report(no_flip), inverted, flip=False)

        flip = tmp_path / "flip"
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.guard.GuardModelPipeline",
            inverted,
            np.random.default_rng(seed=7),
        )
        run_cli(
            [
                "bench",
                "guard",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(flip),
                "--predict-safe",
                *_BASE_CLI_ARGS,
            ]
        )
        _assert_metrics_match_analytic(_read_report(flip), inverted, flip=True)

    def test_use_mini(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.guard.GuardModelPipeline",
            self.SPECS,
            np.random.default_rng(seed=99),
        )
        run_cli(
            [
                "bench",
                "guard",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(tmp_path),
                "--use-mini",
                *_BASE_CLI_ARGS,
            ]
        )
        report = _read_report(tmp_path)
        assert "overall" in report["groups"]
        assert report["groups"]["overall"]["n"] <= 100 * len(DOMAINS) * 2


class TestPerplexityCLI:
    SPECS: dict[str, DomainSpec] = {
        ID_DOMAIN: DomainSpec(safe=(8.0, 0.7), unsafe=(11.0, 0.7)),
        "jailbroken": DomainSpec(safe=(8.0, 0.7), unsafe=(13.0, 0.7)),
        "controlling": DomainSpec(safe=(8.0, 0.7), unsafe=(12.0, 0.7)),
    }

    def test_metrics(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.perplexity.PerplexityPipeline",
            self.SPECS,
            np.random.default_rng(seed=11),
        )
        run_cli(
            [
                "bench",
                "perplexity",
                "--model-id",
                LM_MODEL,
                "--output-dir",
                str(tmp_path),
                *_BASE_CLI_ARGS,
            ]
        )
        _assert_metrics_match_analytic(_read_report(tmp_path), self.SPECS, flip=False)

    def test_use_mini(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.perplexity.PerplexityPipeline",
            self.SPECS,
            np.random.default_rng(seed=55),
        )
        run_cli(
            [
                "bench",
                "perplexity",
                "--model-id",
                LM_MODEL,
                "--output-dir",
                str(tmp_path),
                "--use-mini",
                *_BASE_CLI_ARGS,
            ]
        )
        report = _read_report(tmp_path)
        assert report["groups"]["overall"]["n"] <= 100 * len(DOMAINS) * 2


class TestMahalanobisCLI:
    SPECS: dict[str, DomainSpec] = {
        ID_DOMAIN: DomainSpec(safe=(1.0, 0.20), unsafe=(2.0, 0.20)),
        "jailbroken": DomainSpec(safe=(1.0, 0.20), unsafe=(2.5, 0.20)),
        "controlling": DomainSpec(safe=(1.0, 0.20), unsafe=(2.2, 0.20)),
    }

    def _install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seed: int = 13,
    ) -> None:
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.mahalanobis.MahalanobisPipeline",
            self.SPECS,
            np.random.default_rng(seed=seed),
        )
        monkeypatch.setattr(
            "mood_bench.pipeline.mahalanobis.get_stats_for_model",
            lambda *a, **kw: {"mean": t.zeros(8), "inv_cov": t.eye(8)},
        )

    def test_metrics(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._install(monkeypatch)
        run_cli(
            [
                "bench",
                "mahalanobis",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(tmp_path / "out"),
                "--stats-cache-dir",
                str(tmp_path / "stats"),
                "--refit-stats",
                "--pooling",
                "mean",
                *_BASE_CLI_ARGS,
            ]
        )
        _assert_metrics_match_analytic(
            _read_report(tmp_path / "out"),
            self.SPECS,
            flip=False,
        )

    def test_stats_cache_reuse(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Second run with same cache dir re-uses saved stats."""
        stats_dir = tmp_path / "stats"
        self._install(monkeypatch)
        run_cli(
            [
                "bench",
                "mahalanobis",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(tmp_path / "run1"),
                "--stats-cache-dir",
                str(stats_dir),
                "--refit-stats",
                "--pooling",
                "mean",
                *_BASE_CLI_ARGS,
            ]
        )
        cached_files = list(stats_dir.glob("*.pt"))
        assert len(cached_files) == 1

        self._install(monkeypatch, seed=77)
        run_cli(
            [
                "bench",
                "mahalanobis",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(tmp_path / "run2"),
                "--stats-cache-dir",
                str(stats_dir),
                "--pooling",
                "mean",
                *_BASE_CLI_ARGS,
            ]
        )
        report = _read_report(tmp_path / "run2")
        assert "overall" in report["groups"]


class TestInstructionTunedCLI:
    SPECS: dict[str, DomainSpec] = {
        ID_DOMAIN: DomainSpec(safe=(70.0, 6.0), unsafe=(40.0, 6.0)),
        "jailbroken": DomainSpec(safe=(70.0, 6.0), unsafe=(30.0, 6.0)),
        "controlling": DomainSpec(safe=(70.0, 6.0), unsafe=(33.0, 6.0)),
    }

    def test_metrics(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.instruction_tuned.InstructionTunedPipeline",
            self.SPECS,
            np.random.default_rng(seed=17),
        )
        run_cli(
            [
                "bench",
                "instruction-tuned",
                "--model-id",
                LM_MODEL,
                "--output-dir",
                str(tmp_path),
                *_BASE_CLI_ARGS,
            ]
        )
        _assert_metrics_match_analytic(
            _read_report(tmp_path),
            self.SPECS,
            flip=True,
        )


class TestAnalyzeCLI:
    SPECS: dict[str, DomainSpec] = {
        ID_DOMAIN: DomainSpec(safe=(0.40, 0.07), unsafe=(0.70, 0.07)),
        "jailbroken": DomainSpec(safe=(0.40, 0.07), unsafe=(0.75, 0.07)),
        "controlling": DomainSpec(safe=(0.40, 0.07), unsafe=(0.72, 0.07)),
    }

    def test_single_file(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "scored.jsonl"
        out = tmp_path / "analysis_out"

        _write_scored_jsonl(jsonl, self.SPECS, np.random.default_rng(seed=1))
        run_cli(
            [
                "analyze",
                str(jsonl),
                "--output-dir",
                str(out),
                "--no-figures",
            ]
        )
        report = json.loads((out / "analysis.json").read_text())
        _assert_metrics_match_analytic(report, self.SPECS)

    def test_multi_file_mean_aggregator(self, tmp_path: Path) -> None:
        f1 = tmp_path / "run1.jsonl"
        f2 = tmp_path / "run2.jsonl"
        out = tmp_path / "agg_out"

        _write_scored_jsonl(f1, self.SPECS, np.random.default_rng(seed=2))
        _write_scored_jsonl(f2, self.SPECS, np.random.default_rng(seed=3))
        run_cli(
            [
                "analyze",
                str(f1),
                str(f2),
                "--aggregator",
                "mean",
                "--output-dir",
                str(out),
                "--no-figures",
            ]
        )
        report = json.loads((out / "analysis.json").read_text())
        assert "overall" in report["groups"]
        assert report["groups"]["overall"]["auroc"] > 0.5

    def test_multi_file_min_aggregator(self, tmp_path: Path) -> None:
        f1 = tmp_path / "run1.jsonl"
        f2 = tmp_path / "run2.jsonl"
        out = tmp_path / "agg_out"

        _write_scored_jsonl(f1, self.SPECS, np.random.default_rng(seed=4))
        _write_scored_jsonl(f2, self.SPECS, np.random.default_rng(seed=5))
        run_cli(
            [
                "analyze",
                str(f1),
                str(f2),
                "--aggregator",
                "min",
                "--output-dir",
                str(out),
                "--no-figures",
            ]
        )
        report = json.loads((out / "analysis.json").read_text())
        assert report["groups"]["overall"]["auroc"] > 0.5

    def test_predict_safe(self, tmp_path: Path) -> None:
        inverted: dict[str, DomainSpec] = {
            ID_DOMAIN: DomainSpec(safe=(0.70, 0.07), unsafe=(0.40, 0.07)),
            "jailbroken": DomainSpec(safe=(0.70, 0.07), unsafe=(0.30, 0.07)),
            "controlling": DomainSpec(safe=(0.70, 0.07), unsafe=(0.33, 0.07)),
        }
        jsonl = tmp_path / "scored.jsonl"
        out = tmp_path / "out"

        _write_scored_jsonl(jsonl, inverted, np.random.default_rng(seed=6))
        run_cli(
            [
                "analyze",
                str(jsonl),
                "--output-dir",
                str(out),
                "--no-figures",
                "--predict-safe",
            ]
        )
        report = json.loads((out / "analysis.json").read_text())
        _assert_metrics_match_analytic(report, inverted, flip=True)

    def test_custom_fpr_targets(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "scored.jsonl"
        out = tmp_path / "out"

        _write_scored_jsonl(jsonl, self.SPECS, np.random.default_rng(seed=7))
        run_cli(
            [
                "analyze",
                str(jsonl),
                "--output-dir",
                str(out),
                "--no-figures",
                "--fpr-targets",
                "0.05",
                "0.10",
            ]
        )
        report = json.loads((out / "analysis.json").read_text())
        overall = report["groups"]["overall"]

        assert "tpr@fpr0.05" in overall
        assert "tpr@fpr0.1" in overall
        assert "tpr@fpr0.01" not in overall

    def test_custom_in_distr_domains(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "scored.jsonl"
        out = tmp_path / "out"

        _write_scored_jsonl(jsonl, self.SPECS, np.random.default_rng(seed=8))
        run_cli(
            [
                "analyze",
                str(jsonl),
                "--output-dir",
                str(out),
                "--no-figures",
                "--in-distr-domains",
                ID_DOMAIN,
            ]
        )
        report = json.loads((out / "analysis.json").read_text())
        assert report["in_distr_domains"] == [ID_DOMAIN]
        assert "jailbroken" in report["groups"]
        assert "controlling" in report["groups"]

    def test_multi_file_requires_aggregator(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        _write_scored_jsonl(f1, self.SPECS, np.random.default_rng(seed=9))
        _write_scored_jsonl(f2, self.SPECS, np.random.default_rng(seed=10))
        with pytest.raises(SystemExit):
            run_cli(
                [
                    "analyze",
                    str(f1),
                    str(f2),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--no-figures",
                ]
            )


class TestOutputArtifacts:
    SPECS: dict[str, DomainSpec] = {
        ID_DOMAIN: DomainSpec(safe=(0.40, 0.07), unsafe=(0.70, 0.07)),
        "jailbroken": DomainSpec(safe=(0.40, 0.07), unsafe=(0.75, 0.07)),
        "controlling": DomainSpec(safe=(0.40, 0.07), unsafe=(0.72, 0.07)),
    }

    def test_bench_creates_results_and_analysis(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.guard.GuardModelPipeline",
            self.SPECS,
            np.random.default_rng(seed=50),
        )
        run_cli(
            [
                "bench",
                "guard",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(tmp_path),
                *_BASE_CLI_ARGS,
            ]
        )
        run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert (run_dir / "results.jsonl").exists()
        assert (run_dir / "analysis.json").exists()

    def test_figures_created_without_no_figures(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _install_pipeline_stub(
            monkeypatch,
            "mood_bench.pipeline.guard.GuardModelPipeline",
            self.SPECS,
            np.random.default_rng(seed=51),
        )
        cli_args_with_figs = [a for a in _BASE_CLI_ARGS if a != "--no-figures"]
        run_cli(
            [
                "bench",
                "guard",
                "--model-id",
                CLS_MODEL,
                "--output-dir",
                str(tmp_path),
                *cli_args_with_figs,
            ]
        )
        run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        run_dir = run_dirs[0]
        pngs = list(run_dir.rglob("*.png"))
        assert len(pngs) >= 2, f"expected at least 2 PNGs, got {pngs}"

    def test_analyze_creates_results_and_analysis(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "scored.jsonl"
        _write_scored_jsonl(jsonl, self.SPECS, np.random.default_rng(seed=52))
        out = tmp_path / "out"
        run_cli(
            [
                "analyze",
                str(jsonl),
                "--output-dir",
                str(out),
                "--no-figures",
            ]
        )
        assert (out / "results.jsonl").exists()
        assert (out / "analysis.json").exists()
