"""Load and render Jinja prompt templates from the repo `prompts/` directory."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_REPO_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(loader=FileSystemLoader(_REPO_PROMPTS), autoescape=False)


def render_prompt(name: str, **variables: str) -> str:
    """Render a template under `prompts/` (e.g. ``instruction_grading_no_icl.jinja``)."""
    return _environment().get_template(name).render(**variables)
