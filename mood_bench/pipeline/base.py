from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Pipeline(Protocol):
    def __call__(self, samples: list[str], **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Process samples and return scores for each sample.

        Args:
            samples: List of text samples to process.
            **kwargs: Additional pipeline-specific arguments.

        Returns:
            Tuple of (scores, metadata_dict) in the order of samples.
        """
        ...
