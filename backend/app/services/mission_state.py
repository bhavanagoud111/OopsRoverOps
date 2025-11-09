from typing import Dict, Optional, List
import uuid
from datetime import datetime
import random

from app.models.schemas import (
    MissionState, 
    MissionStatus, 
    RoverPosition, 
    MissionStep,
    MissionLog,
    AgentType,
    AgentStatus
)

class MissionStateManager:
    def __init__(self):
        self.missions: Dict[str, MissionState] = {}
        self.grid_size = 10  # 10x10 grid
        # CRITICAL: Track navigation state for loop detection
        self.navigation_state: Dict[str, Dict] = {}  # mission_id -> {visited_positions, stuck_count, tick_count, prev_pos}

    def create_mission(self, goal: str, obstacles: Optional[List[RoverPosition]] = None) -> str:
        """Create a new mission and return mission_id"""
        mission_id = str(uuid.uuid4())
        
        # Generate obstacles if not provided
        if obstacles is None:
            obstacles = self._generate_obstacles(num_obstacles=5)
        
        mission_state = MissionState(
            mission_id=mission_id,
            goal=goal,
            status=MissionStatus.PENDING,
            rover_position=RoverPosition(x=0, y=0),  # Start at origin
            obstacles=obstacles,
            goal_positions=[],  # Will be set by planner
            steps=[],
            logs=[],
            agent_states={
                AgentType.PLANNER: AgentStatus.IDLE,
                AgentType.ROVER: AgentStatus.IDLE,
                AgentType.SAFETY: AgentStatus.IDLE,
                AgentType.REPORTER: AgentStatus.IDLE,
            }
        )
        
        self.missions[mission_id] = mission_state
        
        # CRITICAL: Initialize navigation state tracking for loop detection
        self.navigation_state[mission_id] = {
            "visited_positions": set(),  # Track visited positions per step
            "stuck_count": 0,  # Consecutive no-progress iterations
            "tick_count": 0,  # Total navigation iterations
            "prev_pos": None,  # Previous position
            "current_step_visited": set()  # Positions visited for current step
        }
        
        # CRITICAL: Save mission to database
        try:
            from app.database.connection import get_db_context
            from app.database.repository import MissionRepository
            with get_db_context() as db:
                repo = MissionRepository(db)
                repo.create_mission(mission_id, goal, MissionStatus.PENDING)
                # Save obstacles to database
                for obstacle in obstacles:
                    repo.add_obstacle(mission_id, obstacle.x, obstacle.y)
                print(f"✅ Mission {mission_id} saved to database")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save mission to database: {e}")
            # Continue without database save (non-blocking)
        
        return mission_id

    def get_mission(self, mission_id: str) -> Optional[MissionState]:
        """Get mission state by ID"""
        return self.missions.get(mission_id)

    def update_mission_status(self, mission_id: str, status: MissionStatus):
        """Update mission status"""
        if mission_id in self.missions:
            self.missions[mission_id].status = status
            self.missions[mission_id].updated_at = datetime.now()
        
        # CRITICAL: Update mission status in database
        try:
            from app.database.connection import get_db_context
            from app.database.repository import MissionRepository
            with get_db_context() as db:
                repo = MissionRepository(db)
                rover_pos = None
                if mission_id in self.missions:
                    rover_pos = {
                        "x": self.missions[mission_id].rover_position.x,
                        "y": self.missions[mission_id].rover_position.y
                    }
                repo.update_mission_status(mission_id, status, rover_pos)
        except Exception as e:
            print(f"⚠️  Warning: Failed to update mission status in database: {e}")
            # Continue without database save (non-blocking)

    def update_rover_position(self, mission_id: str, position: RoverPosition):
        """Update rover position"""
        if mission_id in self.missions:
            self.missions[mission_id].rover_position = position
            self.missions[mission_id].updated_at = datetime.now()
        
        # CRITICAL: Update rover position in database
        try:
            from app.database.connection import get_db_context
            from app.database.repository import MissionRepository
            with get_db_context() as db:
                repo = MissionRepository(db)
                rover_pos = {"x": position.x, "y": position.y}
                # Get current status
                if mission_id in self.missions:
                    status = self.missions[mission_id].status
                    repo.update_mission_status(mission_id, status, rover_pos)
        except Exception as e:
            print(f"⚠️  Warning: Failed to update rover position in database: {e}")
            # Continue without database save (non-blocking)

    def add_step(self, mission_id: str, step: MissionStep):
        """Add a mission step"""
        if mission_id in self.missions:
            self.missions[mission_id].steps.append(step)
            self.missions[mission_id].updated_at = datetime.now()
        
        # CRITICAL: Save step to database
        try:
            from app.database.connection import get_db_context
            from app.database.repository import MissionRepository
            with get_db_context() as db:
                repo = MissionRepository(db)
                repo.add_step(mission_id, step)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save step to database: {e}")
            # Continue without database save (non-blocking)

    def update_step(self, mission_id: str, step_number: int, completed: bool = True, nasa_image_url: Optional[str] = None):
        """Update a mission step"""
        if mission_id in self.missions:
            for step in self.missions[mission_id].steps:
                if step.step_number == step_number:
                    step.completed = completed
                    if nasa_image_url:
                        step.nasa_image_url = nasa_image_url
                    break
            self.missions[mission_id].updated_at = datetime.now()
        
        # CRITICAL: Update step in database
        try:
            from app.database.connection import get_db_context
            from app.database.repository import MissionRepository
            with get_db_context() as db:
                repo = MissionRepository(db)
                repo.update_step(mission_id, step_number, completed, nasa_image_url)
        except Exception as e:
            print(f"⚠️  Warning: Failed to update step in database: {e}")
            # Continue without database save (non-blocking)

    def add_log(self, mission_id: str, log: MissionLog):
        """Add a log entry"""
        if mission_id in self.missions:
            self.missions[mission_id].logs.append(log)
            self.missions[mission_id].updated_at = datetime.now()
        
        # CRITICAL: Save log to database
        try:
            from app.database.connection import get_db_context
            from app.database.repository import MissionRepository
            with get_db_context() as db:
                repo = MissionRepository(db)
                repo.add_log(mission_id, log)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save log to database: {e}")
            # Continue without database save (non-blocking)

    def update_agent_status(self, mission_id: str, agent_type: AgentType, status: AgentStatus):
        """Update agent status"""
        if mission_id in self.missions:
            self.missions[mission_id].agent_states[agent_type] = status
            self.missions[mission_id].updated_at = datetime.now()

    def set_current_step(self, mission_id: str, step_number: int):
        """Set current step number"""
        if mission_id in self.missions:
            self.missions[mission_id].current_step = step_number
            self.missions[mission_id].updated_at = datetime.now()

    def add_nasa_image(self, mission_id: str, image_url: str):
        """Add NASA image URL to mission"""
        if mission_id in self.missions:
            if image_url not in self.missions[mission_id].nasa_images:
                self.missions[mission_id].nasa_images.append(image_url)
            self.missions[mission_id].updated_at = datetime.now()

    def set_weather_data(self, mission_id: str, weather_data: dict):
        """Set weather data for mission"""
        if mission_id in self.missions:
            self.missions[mission_id].weather_data = weather_data
            self.missions[mission_id].updated_at = datetime.now()

    def set_goal_positions(self, mission_id: str, positions: List[RoverPosition]):
        """Set goal positions for mission"""
        if mission_id in self.missions:
            self.missions[mission_id].goal_positions = positions
            self.missions[mission_id].updated_at = datetime.now()

    def add_collected_data(self, mission_id: str, data: Dict):
        """Add collected data (samples, findings) to mission"""
        if mission_id in self.missions:
            if not self.missions[mission_id].collected_data:
                self.missions[mission_id].collected_data = []
            self.missions[mission_id].collected_data.append(data)
            self.missions[mission_id].updated_at = datetime.now()

    def is_position_valid(self, mission_id: str, position: RoverPosition) -> bool:
        """Check if a position is valid (within bounds and not an obstacle)"""
        # Check bounds
        if position.x < 0 or position.x >= self.grid_size or position.y < 0 or position.y >= self.grid_size:
            return False
        
        # Check obstacles
        mission = self.missions.get(mission_id)
        if mission:
            for obstacle in mission.obstacles:
                if obstacle.x == position.x and obstacle.y == position.y:
                    return False
        
        return True

    def is_position_obstacle(self, mission_id: str, position: RoverPosition) -> bool:
        """Check if a position is an obstacle"""
        mission = self.missions.get(mission_id)
        if mission:
            for obstacle in mission.obstacles:
                if obstacle.x == position.x and obstacle.y == position.y:
                    return True
        return False

    def add_obstacle(self, mission_id: str, position: RoverPosition):
        """Add an obstacle to the mission (if not already present)"""
        if mission_id in self.missions:
            mission = self.missions[mission_id]
            # Check if obstacle already exists
            if not any(o.x == position.x and o.y == position.y for o in mission.obstacles):
                mission.obstacles.append(position)
                mission.updated_at = datetime.now()
                print(f"✅ Added obstacle at ({position.x}, {position.y}) to mission {mission_id}")
        
        # CRITICAL: Save obstacle to database
        try:
            from app.database.connection import get_db_context
            from app.database.repository import MissionRepository
            with get_db_context() as db:
                repo = MissionRepository(db)
                repo.add_obstacle(mission_id, position.x, position.y)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save obstacle to database: {e}")
            # Continue without database save (non-blocking)
    
    def get_navigation_state(self, mission_id: str) -> Dict:
        """Get navigation state for loop detection"""
        return self.navigation_state.get(mission_id, {
            "visited_positions": set(),
            "stuck_count": 0,
            "tick_count": 0,
            "prev_pos": None,
            "current_step_visited": set()
        })
    
    def update_navigation_state(self, mission_id: str, **kwargs):
        """Update navigation state"""
        if mission_id not in self.navigation_state:
            self.navigation_state[mission_id] = {
                "visited_positions": set(),
                "stuck_count": 0,
                "tick_count": 0,
                "prev_pos": None,
                "current_step_visited": set()
            }
        self.navigation_state[mission_id].update(kwargs)
    
    def reset_navigation_state(self, mission_id: str):
        """Reset navigation state (e.g., when step changes)"""
        if mission_id in self.navigation_state:
            self.navigation_state[mission_id]["current_step_visited"] = set()
            self.navigation_state[mission_id]["stuck_count"] = 0

    def get_path_distance(self, pos1: RoverPosition, pos2: RoverPosition) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _generate_obstacles(self, num_obstacles: int = 5) -> List[RoverPosition]:
        """Generate random obstacles (excluding start position 0,0)"""
        obstacles = []
        attempts = 0
        max_attempts = 100
        
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            
            # Don't place obstacle at start position
            if x == 0 and y == 0:
                continue
            
            pos = RoverPosition(x=x, y=y)
            # Check if already added
            if not any(o.x == x and o.y == y for o in obstacles):
                obstacles.append(pos)
            
            attempts += 1
        
        return obstacles

    def get_mission_summary(self, mission_id: str) -> Optional[dict]:
        """Get mission summary"""
        mission = self.missions.get(mission_id)
        if not mission:
            return None
        
        completed_steps = sum(1 for step in mission.steps if step.completed)
        
        return {
            "mission_id": mission_id,
            "goal": mission.goal,
            "status": mission.status.value,
            "current_step": mission.current_step,
            "total_steps": len(mission.steps),
            "completed_steps": completed_steps,
            "rover_position": {
                "x": mission.rover_position.x,
                "y": mission.rover_position.y
            },
            "logs_count": len(mission.logs),
            "images_count": len(mission.nasa_images)
        }

# Global instance
mission_state_manager = MissionStateManager()

