from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import numpy as np

PipelineResult: TypeAlias = tuple[np.ndarray, dict[str, Any]]
"""Return type of :class:`Pipeline`: per-input ``scores`` and run ``metadata``."""


class Pipeline(Protocol):
    def __call__(self, samples: list[str], **kwargs: Any) -> PipelineResult:
        """
        Process samples and return scores for each sample.

        Args:
            samples: List of text samples to process.
            **kwargs: Additional pipeline-specific arguments.

        Returns:
            Tuple of (scores, metadata_dict) in the order of samples.
        """
        ...
