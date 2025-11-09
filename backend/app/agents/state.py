from typing import TypedDict, List, Optional, Dict, Any, Annotated
from datetime import datetime
from app.models.schemas import (
    MissionStep, 
    RoverPosition, 
    MissionStatus, 
    AgentType,
    AgentStatus,
    MissionLog
)

class MissionGraphState(TypedDict):
    """State that flows through the LangGraph"""
    mission_id: str
    goal: str
    status: MissionStatus
    steps: List[MissionStep]
    current_step_index: int
    rover_position: RoverPosition
    obstacles: List[RoverPosition]
    goal_positions: List[RoverPosition]
    logs: Annotated[List[MissionLog], "append"]
    agent_states: Dict[str, AgentStatus]
    nasa_images: List[str]
    weather_data: Optional[Dict[str, Any]]
    current_action: Optional[Dict[str, Any]]  # Current action being executed
    safety_approved: Optional[bool]  # Safety validation result
    execution_complete: bool  # Whether all steps are complete
    error: Optional[str]  # Error message if any

