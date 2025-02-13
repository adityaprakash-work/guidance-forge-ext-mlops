# ---INFO-------------------------------------------------------------------------------
"""Prompts agentic systems"""

# ---DEPENDENCIES-----------------------------------------------------------------------
from ..utils import prompt_wrap

# --------------------------------------------------------------------------------------
## Context Delegation Strategy Prompt
sv_beta_cds_plan = prompt_wrap(
    """
### Task  
Generate a JSON response that determines:  
1. Whether a sub-agent is required (`sa_required`).  
2. If a sub-agent is required, specify the necessary context to relay (`sa_context_relay`).  

### Guidelines  
- **When Sub-Agents Are Not Required**  
  - If the user query does not indicate any intent that requires a sub-agent's capabilities, set:  
    ```json
    {
      "sa_name": some_agent_name, # Name of the sub-agent
      "sa_required": false,
      "sa_context_relay": null
    }
    ```
  - This applies when the query can be handled entirely within the primary system without delegation.  

- **When Sub-Agents Are Required**  
  - If the query requires a sub-agent, set `sa_required` to `true` and define `sa_context_relay` as a message that includes:  
    - **Usage Context**: Why the sub-agent is needed based on the user's intent.  
    - **Functional Parameters**: Any required inputs for function calls

- **Ensuring Logical Coherence**  
  - The response should be internally consistent, ensuring that if a sub-agent is required, `sa_context_relay` provides sufficient detail for effective delegation.  
  - The decision should align with the intent inferred from the query, avoiding unnecessary delegation or missing necessary sub-agent involvement.  

### Expected JSON Response Format  
```json
{
  "sa_name": "<Sub-Agent Name>",
  "sa_required": <true/false>,
  "sa_context_relay": "<null or detailed delegation context>"
}
"""
)
