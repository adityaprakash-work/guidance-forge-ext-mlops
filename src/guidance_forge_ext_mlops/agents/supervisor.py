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
# TODO: Custom `role` blocks might not be supported, verify.
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
            self._lm += self.system_prompt + "\n\n### Sub Agents:\n"
            self._lm += "\n".join(
                f"-\tAgent:{agent.name}\n\t{agent.info.replace('\n', '\n\t')}"
                for agent in self._sub_agents
            )

    def add(self, other: str | RawFunction): ...

    @property
    def info(self):
        # TODO: Improve.
        return (
            "Primary point of contact between the user and the agents. "
            "Serves as a router and delegates appropriate actions towards "
            "various agents under its command."
        )
