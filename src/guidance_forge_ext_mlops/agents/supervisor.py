# ---INFO-------------------------------------------------------------------------------
"""Supervisor agents for the MLOps pipeline."""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING

import guidance
from guidance import assistant, gen, role, select, system
from guidance import json as gj
from pydantic import BaseModel, ConfigDict, Field

from ..prompts.agents import sv_beta_cds_plan
from ..utils import prompt_wrap
from .base import BaseAgent

if TYPE_CHECKING:
    from guidance import RawFunction
    from guidance.library._block import ContextBlock
    from guidance.models import Model

    from .base import Agent


# --------------------------------------------------------------------------------------
# TODO: Custom `role` blocks might not be supported, verify.
def supervisor(text=None, **kwargs) -> ContextBlock:
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
                f"-\tAgent: {agent.name}\n\t{agent.info.replace('\n', '\n\t')}"
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

    def get_planning_schema(self) -> tuple[type[BaseModel], type[BaseModel]]:
        """Returns the planning schema for the Supervisor agent."""
        kk_dict = {key: key for key in self._sub_agents}
        SubAgentsEnumType = StrEnum("SubAgentsEnum", kk_dict)

        class AgentCDPlan(BaseModel):
            """Model for context delegation strategy."""

            model_config = ConfigDict(extra="forbid")

            sa_name: SubAgentsEnumType = Field(
                ..., description="Name of the sub-agent."
            )
            sa_required: bool = Field(
                ..., description="Whether the sub-agent is required or not."
            )
            sa_context_relay: str | None = Field(
                None, description="Context to be relayed to the sub-agent."
            )

        class ContextDelegationStrategy(BaseModel):
            """Model for context delegation strategy."""

            model_config = ConfigDict(extra="forbid")

            cds_plan: list[AgentCDPlan] = Field(
                ..., description="List of all agent's context delegation plans."
            )

        return ContextDelegationStrategy, AgentCDPlan

    @staticmethod
    def cds_json_to_markdown(cds_plan: dict) -> str:
        """Converts a JSON string to a Markdown formatted string."""
        markdown_str = "### Context Delegation Strategy:\n"
        for plan in cds_plan.get("cds_plan", []):
            markdown_str += f"- **Sub-Agent Name**: {plan['sa_name']}\n"
            markdown_str += (
                f"  - **Required**: {'Yes' if plan['sa_required'] else 'No'}\n"
            )
            markdown_str += f"  - **Context Relay**: {plan['sa_context_relay']}\n"
        return markdown_str

    @guidance
    def plan(self, lm: Model) -> Model:
        """Plans a context delegation stategy."""
        lm_t = lm.copy()
        cdss, acdp = self.get_planning_schema()
        with assistant():
            lm_t += (
                sv_beta_cds_plan
                + "\n\n ### JSON Response: \n List of all agent's delegation plans -"
                + f"```json\n{gj('cds_plan', schema=cdss)}\n```"
            )
        cds_plan = json.loads(lm_t["cds_plan"])
        if all(not plan["sa_required]"] for plan in cds_plan):
            lm = lm.set("route_to", "self")
        else:
            with self._role:
                lm += self.cds_json_to_markdown(cds_plan)
        return lm

    # NOTE: This function will be redundant if `plan` works as expected.
    # If `select` function's `recurse` parameter is fixed then this function can be
    # modified to be better suited for the task.
    @guidance
    def route(self, lm: Model) -> Model:
        """Routes the user query to the appropriate agent."""
        lm_t = lm.copy()
        with assistant():
            lm_t += (
                "Routing as per the plan and the best execution order for sub-agents: "
                "Routing to self if all the listed sub-agents as per the plan are "
                "executed."
            ) + select(self._sub_agents.keys(), name="route_to")
            if lm_t["route_to"] != "self":
                lm_t += (
                    f"\nRelaying relevant context and data to {lm_t['route_to']} as per"
                    " its description: " + gen(name="route_context")
                )
        lm = lm.set("route_to", lm_t["route_to"])
        lm = lm.set("route_context", lm_t["route_context"])
        return lm

    def add(self, other: str | RawFunction, source: str = None) -> Supervisor:
        """Invokes the Supervisor agent by adding text or raw functions to it."""
        # NOTE: `str` is the default `other` to converse with the agent (for now). All
        # other RawFunctions must implement their own logic.
        # For now `route()` is used for an additional layer of intelligent relaying of
        # context in case of hierarchical tool-call topologies.
        if isinstance(other, str):
            with role(source):
                self._lm += other
            self._lm += self.plan()

            if self._lm["route_to"] == "self":
                with self._role:
                    self._lm + gen()
            else:
                self._lm = self._lm.set("orchestration_complete", False)
                counter = 0
                while not self._lm["orchestration_complete"] and counter < len(
                    self._sub_agents
                ):
                    # TODO: Revamp routing action, possibly replace with plan.
                    counter += 1
                    self._lm += self.route()
                    if self._lm["route_to"] == "self":
                        with self._role:
                            self._lm += "\n\nEnd of sub-agent orchestration: " + gen()
                            break
                    with self._role:
                        self._lm += f"Routing to {self._lm['route_to']} > "
                    r_subagent = self._sub_agents[self._lm["route_to"]]
                    r_subagent += self.format_relay(f"{self._lm['route_context']}")
                    with r_subagent._role:
                        self._lm += r_subagent.last_response
        else:
            # For more complex conversational interfaces.
            self._lm += other
        return self

    @property
    def info(self) -> str:
        """Provides detailed information about the Supervisor agent."""
        return prompt_wrap(
            """
        The Supervisor agent is the main contact in the MLOps pipeline, routing tasks 
        to sub-agents based on their expertise. If no routing is needed, it interacts 
        directly with the user for efficient task management (it routes back to `self`).
        """
        )
