from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class AgentType(str, Enum):
    PLANNER = "planner"
    ROVER = "rover"
    SAFETY = "safety"
    REPORTER = "reporter"
    SUPERVISOR = "supervisor"
    SYSTEM = "system"

class AgentStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    REPORTING = "reporting"
    COMPLETE = "complete"
    ERROR = "error"

class MissionStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETE = "complete"
    ABORTED = "aborted"
    ERROR = "error"

class RoverPosition(BaseModel):
    x: int = Field(ge=0, le=9, description="X coordinate (0-9)")
    y: int = Field(ge=0, le=9, description="Y coordinate (0-9)")

class MissionGoal(BaseModel):
    goal: str = Field(..., description="Natural language mission goal")
    user_id: Optional[str] = None

class MissionStep(BaseModel):
    step_number: int
    action: str = Field(..., description="Action to perform (move, explore, return, etc.)")
    target_position: Optional[RoverPosition] = None
    description: str
    completed: bool = False
    nasa_image_url: Optional[str] = None

class MissionPlan(BaseModel):
    mission_id: str
    goal: str
    steps: List[MissionStep]
    created_at: datetime = Field(default_factory=datetime.now)

class AgentAction(BaseModel):
    agent_type: AgentType
    action: str
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Any] = {}
    status: AgentStatus = AgentStatus.EXECUTING

class MissionLog(BaseModel):
    mission_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_type: AgentType
    message: str
    level: Literal["info", "warning", "error", "success"] = "info"
    data: Optional[Dict[str, Any]] = None

class MissionState(BaseModel):
    mission_id: str
    goal: str
    status: MissionStatus = MissionStatus.PENDING
    current_step: int = 0
    rover_position: RoverPosition = Field(default_factory=lambda: RoverPosition(x=0, y=0))
    obstacles: List[RoverPosition] = []
    goal_positions: List[RoverPosition] = []
    steps: List[MissionStep] = []
    logs: List[MissionLog] = []
    agent_states: Dict[AgentType, AgentStatus] = {
        AgentType.PLANNER: AgentStatus.IDLE,
        AgentType.ROVER: AgentStatus.IDLE,
        AgentType.SAFETY: AgentStatus.IDLE,
        AgentType.REPORTER: AgentStatus.IDLE,
    }
    nasa_images: List[str] = []
    weather_data: Optional[Dict[str, Any]] = None
    collected_data: List[Dict[str, Any]] = []  # Store collected samples and findings
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class MissionReport(BaseModel):
    mission_id: str
    goal: str
    status: MissionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    steps_completed: int
    total_steps: int
    steps: List[MissionStep]
    logs: List[MissionLog]
    nasa_images: List[str] = []
    weather_data: Optional[Dict[str, Any]] = None
    summary: str = ""
    outcome: str = ""

class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type: log, position, status, complete, error")
    mission_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = {}

# Request/Response models
class StartMissionRequest(BaseModel):
    goal: str

class StartMissionResponse(BaseModel):
    mission_id: str
    status: str
    message: str

class MissionStatusResponse(BaseModel):
    mission_id: str
    status: MissionStatus
    state: Optional[MissionState] = None

