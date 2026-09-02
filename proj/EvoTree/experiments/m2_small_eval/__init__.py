"""M2 small-scale OSWorld evaluation helpers."""

from typing import Any

__all__ = ["summarize_experiment"]


def __getattr__(name: str) -> Any:
    if name == "summarize_experiment":
        from .report import summarize_experiment

        return summarize_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
