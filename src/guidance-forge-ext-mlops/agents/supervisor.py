# ---INFO-------------------------------------------------------------------------------
"""Supervisor agents for the MLOps pipeline."""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

from guidance import role, select, system

from ..utils import toggle_echo
from .base import BaseAgent

if TYPE_CHECKING:
    from guidance import RawFunction
    from guidance.models import Model

    from .base import Agent


# --------------------------------------------------------------------------------------
def supervisor(text=None, **kwargs):
    """Creates a supervisor role block. Does not enforce the Supervior agent strictly,
    that means any text can be generated under this role."""
    return role("supervisor", text=text, **kwargs)


class Supervisor(BaseAgent):
    """Supervises other agents and acts as the primary point of contact for the pipeline
    as a conversational router."""

    def __init__(
        self, name: str, system_prompt: str, lm: Model, sub_agents: list[Agent] = None
    ):
        super().__init__()
        self.name = name
        self.system_prompt = system_prompt
        self._lm = lm
        # TODO: `text` and `**kwargs` are not yet supported in the `role` function.
        # Can this be used in the future?
        self._role = role(f"supervisor_{name}")
        self._sub_agents = sub_agents or []

        # LM Setup
        with system():
            self._lm += self.system_prompt + "\n\n###Sub Agents:\n"
            agent_info = ...

    def add(self, other: str | RawFunction): ...
