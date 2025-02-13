# ---INFO-------------------------------------------------------------------------------
"""Base agents."""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from guidance import role

if TYPE_CHECKING:
    from guidance import RawFunction
    from guidance.models import Model


# --------------------------------------------------------------------------------------
class BaseAgent(ABC):
    """Base agent class for the MLOps pipeline."""

    def __init__(self, name: str, system_prompt: str, lm: Model, echo: bool = False):
        self.name = name
        self.system_prompt = system_prompt
        self._lm = lm.copy()
        self.echo = echo
        # TODO: `text` and `**kwargs` are not yet supported in the `role` function.
        # Can this be used in the future?
        self._role = role(self.name)
        self._role_start_tag = getattr(
            self._lm, "get_role_start", self._lm.chat_template.get_role_start
        )(self.name)

    @property
    def echo(self) -> bool:
        return self._lm.echo

    @echo.setter
    def echo(self, value: bool):
        self._lm.echo = value

    def __add__(self, other: str | RawFunction):
        """Add text and raw functions to the agent."""
        # TODO: Can we possibly do some more pre/post-processing here?
        # search for source between <gfem:agent:source> and </gfem:agent:source> tags.
        if isinstance(other, str):
            source = (
                m.group(1)
                if (
                    m := re.search(
                        r"<gfem:agent:source>(.*?)</gfem:agent:source>", other
                    )
                )
                else "user"
            )
            other = re.sub(r"<gfem:agent:source>.*?</gfem:agent:source>", "", other)
        else:
            source = None
        return self.add(other, source=source)

    def format_relay(self, text: str):
        """Format the text to be relayed by the agent with source identifier."""
        return f"<gfem:agent:source>{self._role.name}</gfem:agent:source>\n{text}"

    @abstractmethod
    def add(self, other: str | RawFunction, source: str = None):
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
    def info(self) -> str:
        """Get information about the agent."""
        raise NotImplementedError

    @property
    def last_response(self) -> str:
        """Get the last response from the agent."""
        # TODO: Very inefficient way to get the last response. Can we do better?
        # Search for the last occurence of role_start_tag in the _lm and return the text
        # after that.
        _lm_str = str(self._lm)
        last_occurence = _lm_str.rfind(self._role_start_tag)
        return _lm_str[last_occurence:] if last_occurence != -1 else ""


Agent = TypeVar("Agent", bound=BaseAgent)
