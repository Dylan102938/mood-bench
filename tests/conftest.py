from __future__ import annotations

import gc
import json
import os
import queue
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator

import pytest
import torch as t
from datasets import Dataset, load_dataset

RELEASE_REPO = "Dylan102938/mood-bench"
RELEASE_TAG = "test-fixtures-v1"
RESULTS_DIR = Path(__file__).parent / "results"


class GPUPool:
    """Thread-safe pool of GPU IDs. Tests acquire/release GPUs from this pool."""

    def __init__(self, gpu_ids: list[int]) -> None:
        self._all_ids = list(gpu_ids)
        self._queue: queue.Queue[int] = queue.Queue()
        for gid in self._all_ids:
            self._queue.put(gid)
        self._semaphore = threading.Semaphore(len(self._all_ids))

    @property
    def total(self) -> int:
        return len(self._all_ids)

    def acquire(self, n: int = 1) -> list[int]:
        ids: list[int] = []
        for _ in range(n):
            self._semaphore.acquire()
            ids.append(self._queue.get())
        return ids

    def release(self, ids: list[int]) -> None:
        for gid in ids:
            self._queue.put(gid)
            self._semaphore.release()


def _discover_gpu_ids() -> list[int]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip():
        return [int(x) for x in env.split(",")]

    n = t.cuda.device_count()
    return list(range(n))


def _get_worker_id(request: pytest.FixtureRequest) -> str:
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]

    return "master"


@pytest.fixture(scope="session")
def gpu_pool(request: pytest.FixtureRequest) -> GPUPool:
    all_ids = _discover_gpu_ids()
    if not all_ids:
        pytest.skip("No GPUs available")

    wid = _get_worker_id(request)
    if wid == "master":
        return GPUPool(all_ids)

    worker_idx = int(wid.replace("gw", ""))
    gpu_id = all_ids[worker_idx % len(all_ids)]
    return GPUPool([gpu_id])


@pytest.fixture()
def gpu(gpu_pool: GPUPool) -> Generator[list[int], None, None]:
    ids = gpu_pool.acquire(n=1)
    old_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
    try:
        yield ids
    finally:
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()
        if old_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_env
        gpu_pool.release(ids)


@pytest.fixture(scope="session")
def results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def load_analysis(results_path: Path) -> dict:
    return json.loads((results_path / "analysis.json").read_text())


def get_metric(analysis: dict, group: str, metric: str) -> float:
    return analysis["groups"][group][metric]


def assert_tpr_metrics(
    analysis: dict,
    expected: dict[str, float],
    *,
    tolerance: float,
    fpr: float = 0.01,
) -> None:
    """Assert that ``tpr@fpr{fpr}`` matches ``expected[group]`` (percent) within ``tolerance``."""
    metric = f"tpr@fpr{fpr}"
    for group, expected_tpr in expected.items():
        actual = get_metric(analysis, group, metric) * 100
        assert actual == pytest.approx(
            expected_tpr, abs=tolerance
        ), f"{group}: {actual:.2f} != {expected_tpr} ± {tolerance}"


def _download_release_asset(repo: str, tag: str, asset_name: str, dest: Path) -> bool:
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            release = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise

    asset = next((a for a in release.get("assets", []) if a["name"] == asset_name), None)
    if asset is None:
        return False

    dl_headers = dict(headers)
    dl_headers["Accept"] = "application/octet-stream"
    dl_req = urllib.request.Request(asset["url"], headers=dl_headers)
    with urllib.request.urlopen(dl_req) as resp:
        dest.write_bytes(resp.read())
    return True


def _release_asset_or_skip(asset_name: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download ``<asset_name>.jsonl`` from the configured release or skip.

    Skips the requesting test if the asset is missing or the download fails.
    Setting ``MOOD_BENCH_FIXTURE_<ASSET_NAME_UPPER>`` to a local path bypasses the
    download and uses that file directly (useful for running against alternative
    score files).
    """
    env_key = f"MOOD_BENCH_FIXTURE_{asset_name.upper()}"
    override = os.environ.get(env_key)
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            pytest.skip(f"Fixture override {env_key}={override} does not exist")
        return path

    dest = tmp_path_factory.mktemp(f"release_{asset_name}") / f"{asset_name}.jsonl"
    try:
        ok = _download_release_asset(RELEASE_REPO, RELEASE_TAG, f"{asset_name}.jsonl", dest)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Fixture download failed for {asset_name}.jsonl: {e}")

    if not ok:
        pytest.skip(f"Release fixture {asset_name!r} not available")

    return dest


def _load_scored_dataset(path: Path) -> Dataset:
    ds = load_dataset("json", data_files=str(path), split="train")
    if "malign" not in ds.column_names and "safe" in ds.column_names:
        ds = ds.map(lambda ex: {"malign": int(not bool(ex["safe"]))})
    return ds


@pytest.fixture(scope="session")
def guard_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _load_scored_dataset(_release_asset_or_skip("guard", tmp_path_factory))


@pytest.fixture(scope="session")
def perplexity_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _load_scored_dataset(_release_asset_or_skip("perplexity", tmp_path_factory))


@pytest.fixture(scope="session")
def mahalanobis_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _load_scored_dataset(_release_asset_or_skip("mahalanobis", tmp_path_factory))


@pytest.fixture(scope="session")
def it_alignment_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _load_scored_dataset(_release_asset_or_skip("it_alignment", tmp_path_factory))


@pytest.fixture(scope="session")
def it_uncertainty_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _load_scored_dataset(_release_asset_or_skip("it_uncertainty", tmp_path_factory))
