from __future__ import annotations

import gc
import json
import os
import queue
import threading
from pathlib import Path
from typing import Generator

import pytest
import torch as t


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


RESULTS_DIR = Path(__file__).parent / "results"


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
    """Acquire 1 GPU, set CUDA_VISIBLE_DEVICES, yield, then release."""
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
