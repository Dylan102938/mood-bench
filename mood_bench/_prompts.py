from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from jinja2 import Environment, FunctionLoader


def _resource_loader(name: str) -> str | None:
    resource = files("mood_bench.prompts").joinpath(name)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(loader=FunctionLoader(_resource_loader), autoescape=False)


def render_prompt(name: str, **variables: str) -> str:
    return _environment().get_template(name).render(**variables)


def read_prompt_file(name: str) -> str:
    return files("mood_bench.prompts").joinpath(name).read_text(encoding="utf-8")
