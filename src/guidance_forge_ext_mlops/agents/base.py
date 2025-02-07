# ---INFO-------------------------------------------------------------------------------
"""Base agents."""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from guidance import RawFunction


# --------------------------------------------------------------------------------------
class BaseAgent(ABC):
    """Base agent class for the MLOps pipeline."""

    def __add__(self, other: str | RawFunction):
        """Add text and raw functions to the agent."""
        # TODO: Can we possibly do some pre/post-processing here?
        return self.add(other)

    @abstractmethod
    def add(self, other: str | RawFunction):
        """Add text and raw functions to the agent."""
        # NOTE: Default logic considerations
        # - Not returning `self`, could there be use-cases where we want to return
        #   something else?
        # - Not adding directly to `self._lm`, to keep the implementation flexible.
        # - We can possibly return other constructs. For example, we can iterate
        #   over Agents by invoking them with various raw functions and each time
        #   getting a different agent based on some heuristic.
        raise NotImplementedError

    @property
    @abstractmethod
    def info(self):
        """Get information about the agent."""
        raise NotImplementedError


Agent = TypeVar("Agent", bound=BaseAgent)
