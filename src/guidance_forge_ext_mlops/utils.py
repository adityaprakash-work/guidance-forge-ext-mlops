# ---INFO-------------------------------------------------------------------------------
"""Utility functions for the package"""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

from contextlib import contextmanager
from textwrap import dedent, wrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from guidance.models import Model


# --------------------------------------------------------------------------------------
@contextmanager
def toggle_echo(lm: Model | list[Model]):
    """Toggle the echo attribute of a model or a list of models."""
    lm = [lm] if isinstance(lm, Model) else lm
    original_st = {model: model.echo for model in lm}
    for model in lm:
        model.echo = not model.echo
    try:
        yield
    finally:
        for model in lm:
            model.echo = original_st[model]


def prompt_wrap(text: str, width: int = 120) -> str:
    """Wrap text to a given width, preserving existing newlines."""
    dedented_text = dedent(text)
    wrapped_lines = []
    for line in dedented_text.splitlines():
        wrapped_lines.extend(wrap(line, width=width, replace_whitespace=False))
    return "\n".join(wrapped_lines)
