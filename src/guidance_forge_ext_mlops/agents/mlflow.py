# ---INFO-------------------------------------------------------------------------------
"""MLFlow Agent."""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

import guidance
from guidance import system

from .base import BaseAgent

if TYPE_CHECKING:
    from guidance.models import Model


# --------------------------------------------------------------------------------------
class MLFlowAgent(BaseAgent):
    """MLFlow agent for the MLOps pipeline."""

    def __init__(
        self,
        system_prompt: str,
        lm: Model,
        name: str = "mlflow-agent",
        echo: bool = True,
    ):
        super().__init__(name, system_prompt, lm, echo=echo)

        with system():
            self._lm += self.system_prompt
