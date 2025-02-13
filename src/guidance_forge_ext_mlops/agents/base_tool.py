# ---INFO-------------------------------------------------------------------------------
"""Base Tool Agent."""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import guidance
from guidance import assistant, role, system

from .base import BaseAgent

if TYPE_CHECKING:
    from guidance import RawFunction
    from guidance._guidance import GuidanceFunction
    from guidance.models import Model


# --------------------------------------------------------------------------------------
class BaseToolAgent(BaseAgent, ABC):
    """Base tool agent for the MLOps pipeline."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        lm: Model,
        echo: bool = True,
        tools: list[GuidanceFunction] = None,
    ):
        super().__init__(name, system_prompt, lm, echo=echo)
        self._tools = {
            **(
                {
                    tool.name: tool
                    for tool in tools
                    # TODO: Change this to proper logs regarding incompatible tools.
                    if hasattr(tool, "name") and hasattr(tool, "info")
                }
                if tools
                else {}
            ),
            "self": self,
        }

        # LM Setup
        with system():
            self._lm += self.system_prompt + "\n\n### Available Tools:\n"
            self._lm += "\n".join(
                f"-\tTool: {tool.name}\n\t{tool.info.replace('\n', '\n\t')}"
                for tool in self._tools
            )
            self._lm += (
                "### Role Block"
                + f"Consider `assistant` and `{self._role.name}` roles to be identical."
            )

    def add(self, other: str | RawFunction, source: str = None) -> ToolAgent:
        """Add text and raw functions to the agent."""
        if isinstance(other, str):
            with role(source):
                self._lm += other

    @guidance
    def determine_tool(self, lm: Model) -> Model:
        lm_t = lm.copy()
        with assistant():
            ...

    @property
    @abstractmethod
    def info(self) -> str:
        """Provides a brief overview of the tool-using agent. This information should be
        sufficient to guide routing decisions and should not be verbose."""
        return NotImplementedError


ToolAgent = TypeVar("ToolAgent", bound=BaseToolAgent)
