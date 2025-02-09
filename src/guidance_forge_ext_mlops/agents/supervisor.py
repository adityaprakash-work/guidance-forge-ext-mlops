# ---INFO-------------------------------------------------------------------------------
"""Supervisor agents for the MLOps pipeline."""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

import guidance
from guidance import assistant, gen, role, select, system, user

from ..utils import prompt_wrap
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
        self,
        name: str,
        system_prompt: str,
        lm: Model,
        echo: bool = True,
        sub_agents: list[Agent] = None,
        sa_verbose: bool = False,
    ):
        super().__init__(name, system_prompt, lm, echo=echo)
        self._sub_agents = {**{agent.name: agent for agent in sub_agents}, "self": self}
        # Switch off echo for all the sub-agents so that the Supervisor is the only one
        # echoing the conversation state.
        self.set_sa_verbose(sa_verbose)

        # LM Setup
        with system():
            self._lm += self.system_prompt + "\n\n### Sub Agents:\n"
            self._lm += "\n".join(
                f"-\tAgent:{agent.name}\n\t{agent.info.replace('\n', '\n\t')}"
                for agent in self._sub_agents
            )
            self._lm += (
                "### Role Block"
                + f"Consider `assistant` and `{self._role.name}` roles to be identical."
            )

    def set_sa_verbose(self, value: bool):
        """Sets echo for the sub-agents."""
        for agent in self._sub_agents.values():
            if agent != self:
                agent.echo = value

    def add(self, other: str | RawFunction, source: str = None):
        """Invokes the Supervisor agent by adding text or raw functions to it."""
        # NOTE: `str` is the default `other` to converse with the agent (for now). All
        # other RawFunctions must implement their own logic.
        if isinstance(other, str):
            with role(source):
                self._lm += other
            self._lm += self.route()

            if self._lm["route_to"] == "self":
                with self._role:
                    self._lm + gen()
            else:
                with self._role:
                    self._lm += f"Routing to {self._lm['route_to']} > "
                r_subagent = self._sub_agents[self._lm["route_to"]]
                r_subagent += self.format_relay(f"{self._lm['route_context']}")
                with r_subagent._role:
                    self._lm += r_subagent.last_response
        else:
            # For more complex conversation interfaces.
            self._lm += other
        return self

    @guidance
    def route(self, lm: Model):
        """Routes the user query to the appropriate agent."""
        lm_t = lm.copy()
        with assistant():
            lm_t += (
                "Routing to the appropriate sub agent as per the current requirement: "
            ) + select(self._sub_agents.keys(), name="route_to")
            if lm_t["route_to"] != "self":
                lm_t += (
                    "\nRelaying relevant context and data to the sub-agent as per its "
                    "specification: " + gen(name="route_context")
                )
        lm = lm.set("route_to", lm_t["route_to"])
        lm = lm.set("route_context", lm_t["route_context"])
        return lm

    @property
    def info(self):
        """Provides detailed information about the Supervisor agent."""
        return prompt_wrap(
            """
        The Supervisor agent is the main contact in the MLOps pipeline, routing tasks 
        to sub-agents based on their expertise. If no routing is needed, it interacts 
        directly with the user for efficient task management.
        """
        )
