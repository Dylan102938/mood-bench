"""Dataset loaders for mood-bench."""

from datasets import Dataset


def load_mood_dataset(split: str = "train") -> Dataset:
    """Load MOOD benchmark data (not implemented yet)."""
    raise NotImplementedError(f"load_mood_dataset is not implemented yet ({split=})")
