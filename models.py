from pydantic import BaseModel, Field
from typing import Dict, Any, Literal


class Observation(BaseModel):
    """What the agent sees after every step."""
    inbox: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current customer email the agent must handle.",
    )
    db_result: str = Field(
        default="",
        description="Rows / feedback returned by the last QUERY_DB action.",
    )
    system_feedback: str = Field(
        default="",
        description="System message about the last action's outcome.",
    )
    is_done: bool = Field(
        default=False,
        description="True when the episode has ended.",
    )


class Action(BaseModel):
    """What the agent can do each turn."""
    action_type: Literal[
        "SEARCH_BOOKINGS",
        "FETCH_FLIGHT_STATUS",
        "PROCESS_REFUND",
        "SEARCH_POLICY_DOCS",
        "MODIFY_BOOKING",
        "SEND_REPLY",
        "END_EPISODE",
    ] = Field(..., description="The type of backend action to execute.")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the chosen action_type.",
    )
