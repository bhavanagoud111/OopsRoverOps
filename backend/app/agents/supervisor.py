import os
import json
from datetime import datetime
from typing import Dict, Any, Literal, Optional, Callable, Awaitable
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from app.agents.state import MissionGraphState
from app.agents.planner import PlannerAgent
from app.agents.rover import RoverAgent
from app.agents.safety import SafetyAgent
from app.agents.reporter import ReporterAgent
from app.models.schemas import (
    AgentType, 
    AgentStatus, 
    MissionStatus, 
    MissionLog,
    RoverPosition,
    MissionStep
)
from app.services.mission_state import mission_state_manager
from app.services.nasa_client import nasa_client
from app.phases import Phase, at_base
from app.pathfinding import astar
from app.navigation_utils import is_at_target, calculate_distance

# CRITICAL: Hard timebox to prevent infinite loops
MAX_TICKS_PER_MISSION = 200
STUCK_LIMIT = 8  # Consecutive no-progress iterations before aborting

class MissionSupervisor:
    """LangGraph-based supervisor that orchestrates all agents"""
    
    def __init__(self, app_instance=None):
        self.planner = PlannerAgent()
        self.rover = RoverAgent()
        self.safety = SafetyAgent()
        self.reporter = ReporterAgent()
        self.graph = self._build_graph()
        self.broadcast_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        
        # Store app instance for phase machine access
        self.app = app_instance
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        workflow = StateGraph(MissionGraphState)

        # Add nodes (add emergency_return early so it's available for conditional edges)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("rover", self._rover_node)
        workflow.add_node("safety", self._safety_node)
        workflow.add_node("reporter", self._reporter_node)
        workflow.add_node("fetch_nasa_data", self._fetch_nasa_data_node)
        workflow.add_node("update_position", self._update_position_node)
        workflow.add_node("emergency_return", self._emergency_return_node)

        # Set entry point
        workflow.set_entry_point("planner")

        # Add edges
        workflow.add_edge("planner", "fetch_nasa_data")
        workflow.add_edge("fetch_nasa_data", "rover")
        workflow.add_conditional_edges(
            "rover",
            self._should_validate_or_complete,
            {
                "validate": "safety",
                "execute": "update_position",
                "complete": "reporter"
            }
        )
        workflow.add_conditional_edges(
            "safety",
            self._safety_decision,
            {
                "approved": "update_position",
                "rejected": "rover",  # Re-plan move
                "abort": "emergency_return"
            }
        )
        workflow.add_conditional_edges(
            "update_position",
            self._should_continue,
            {
                "continue": "rover",
                "complete": "reporter",
                "abort": "emergency_return"
            }
        )
        workflow.add_edge("emergency_return", "reporter")
        workflow.add_edge("reporter", END)

        return workflow.compile()
    
    async def _planner_node(self, state: MissionGraphState) -> Dict[str, Any]:
        """Planner agent node"""
        mission_id = state["mission_id"]
        goal = state["goal"]
        
        # Update agent status
        mission_state_manager.update_agent_status(mission_id, AgentType.PLANNER, AgentStatus.PLANNING)
        
        # Add log
        log = MissionLog(
            mission_id=mission_id,
            agent_type=AgentType.PLANNER,
            message=f"Planning mission: {goal}",
            level="info"
        )
        mission_state_manager.add_log(mission_id, log)
        
        # Generate plan
        steps = await self.planner.plan_mission(goal)
        
        # CRITICAL: Validate first step has correct target
        if steps and len(steps) > 0:
            first_step = steps[0]
            goal_coords = self.planner._extract_coordinates_from_goal(goal)
            if goal_coords and (goal_coords["x"] is not None and goal_coords["y"] is not None):
                if first_step.target_position:
                    if first_step.target_position.x != goal_coords["x"] or first_step.target_position.y != goal_coords["y"]:
                        print(f"⚠️  WARNING: First step target ({first_step.target_position.x}, {first_step.target_position.y}) doesn't match goal ({goal_coords['x']}, {goal_coords['y']}). FORCING correction...")
                        # Force correct target
                        first_step.target_position = RoverPosition(x=goal_coords["x"], y=goal_coords["y"])
                        first_step.description = f"Move to target coordinates ({goal_coords['x']}, {goal_coords['y']}) from mission goal"
                        print(f"✅ CORRECTED first step target to ({goal_coords['x']}, {goal_coords['y']})")
                else:
                    print(f"⚠️  WARNING: First step has no target! Adding target ({goal_coords['x']}, {goal_coords['y']})...")
                    first_step.target_position = RoverPosition(x=goal_coords["x"], y=goal_coords["y"])
                    first_step.description = f"Move to target coordinates ({goal_coords['x']}, {goal_coords['y']}) from mission goal"
        
        # Update state
        mission_state_manager.update_agent_status(mission_id, AgentType.PLANNER, AgentStatus.COMPLETE)
        mission_state_manager.update_mission_status(mission_id, MissionStatus.PLANNING)
        
        # Store steps in mission state (clear existing steps first to avoid duplicates)
        mission = mission_state_manager.get_mission(mission_id)
        if mission:
            mission.steps = []
        for step in steps:
            mission_state_manager.add_step(mission_id, step)
            
        # CRITICAL: Log first step target for debugging
        if steps and len(steps) > 0:
            first_step = steps[0]
            if first_step.target_position:
                print(f"🔍 DEBUG: First step stored with target ({first_step.target_position.x}, {first_step.target_position.y})")
            else:
                print(f"❌ ERROR: First step has NO target_position!")
        
        log = MissionLog(
            mission_id=mission_id,
            agent_type=AgentType.PLANNER,
            message=f"Generated {len(steps)} mission steps",
            level="success"
        )
        mission_state_manager.add_log(mission_id, log)
        
        return {
            "steps": steps,
            "current_step_index": 0,
            "status": MissionStatus.EXECUTING,
            "logs": [log]
        }
    
    async def _fetch_nasa_data_node(self, state: MissionGraphState) -> Dict[str, Any]:
        """Fetch NASA data for mission context"""
        mission_id = state["mission_id"]
        
        try:
            # Fetch weather data
            weather_data = await nasa_client.get_mars_weather()
            mission_state_manager.set_weather_data(mission_id, weather_data)
            
            log = MissionLog(
                mission_id=mission_id,
                agent_type=AgentType.SUPERVISOR,
                message="Fetched Mars weather data from NASA API",
                level="info"
            )
            mission_state_manager.add_log(mission_id, log)
            
            return {
                "weather_data": weather_data,
                "logs": [log]
            }
        except Exception as e:
            log = MissionLog(
                mission_id=mission_id,
                agent_type=AgentType.SUPERVISOR,
                message=f"Failed to fetch NASA data: {str(e)}",
                level="warning"
            )
            mission_state_manager.add_log(mission_id, log)
            return {"logs": [log]}
    
    async def _rover_node(self, state: MissionGraphState) -> Dict[str, Any]:
        """Rover agent node"""
        mission_id = state["mission_id"]
        
        # CRITICAL: Get phase machine state from app
        if not self.app:
            # Fallback: try to import app if not passed
            try:
                from app.main import app as app_instance
                self.app = app_instance
            except ImportError:
                # If app not available, use default phase
                phase = Phase.EXECUTE_TASK
                base = (0, 0)
                route_q = None
                # Skip phase logic if app not available
                pass
        
        if self.app:
            phase = self.app.state.phase
            base = self.app.state.base
            route_q = self.app.state.route_q
        else:
            phase = Phase.EXECUTE_TASK
            base = (0, 0)
            route_q = None
        
        # CRITICAL FIX: Get steps from mission_state_manager (source of truth) not graph state
        mission = mission_state_manager.get_mission(mission_id) if mission_id else None
        if mission:
            steps = mission.steps  # Use mission state manager as source of truth
        else:
            steps = state.get("steps", [])
        
        current_step_index = state.get("current_step_index", 0)
        rover_position = state.get("rover_position", RoverPosition(x=0, y=0))
        # CRITICAL FIX: Get obstacles from mission state manager (source of truth) to include newly detected obstacles
        if mission:
            obstacles = mission.obstacles  # Use mission state manager as source of truth
        else:
            obstacles = state.get("obstacles", [])
        
        # CRITICAL: Handle RETURN_TO_BASE phase - consume route queue, no replanning
        if route_q and phase == Phase.RETURN_TO_BASE:
            cur = (rover_position.x, rover_position.y)
            
            # Snap if we're within 1 cell of base to avoid last-meter ping-pong
            if at_base(cur, base, tol=1):
                # Snap to base
                base_pos = RoverPosition(x=base[0], y=base[1])
                mission_state_manager.update_rover_position(mission_id, base_pos)
                mission_state_manager.add_log(
                    mission_id,
                    MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.ROVER,
                        message="Base reached. Mission complete.",
                        level="success"
                    )
                )
                # Mark return step as complete
                return_steps = [s for s in steps if s.action == "return"]
                if return_steps:
                    mission_state_manager.update_step(mission_id, return_steps[0].step_number, completed=True)
                # Update phase to COMPLETE
                if self.app:
                    self.app.state.phase = Phase.COMPLETE
                mission_state_manager.update_mission_status(mission_id, MissionStatus.COMPLETE)
                return {"execution_complete": True, "current_step_index": len(steps) if steps else 0}
            
            # If the next node equals current, pop it and continue
            route_q.discard_until(cur)
            nxt = route_q.peek()
            
            if not nxt:
                # Route exhausted but not at base -> one final plan (blocked mid-way)
                obstacle_coords = [(o.x, o.y) for o in obstacles]
                planned = astar(10, 10, cur, base, obstacle_coords, safety=0, diag=True, turn_penalty=0.08)
                if not planned:
                    planned = astar(10, 10, cur, base, obstacle_coords, safety=0, diag=True, turn_penalty=0.06)
                if not planned:
                    mission_state_manager.add_log(
                        mission_id,
                        MissionLog(
                            mission_id=mission_id,
                            agent_type=AgentType.ROVER,
                            message="Route depleted and replan failed; aborting.",
                            level="error"
                        )
                    )
                    mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
                    if self.app:
                        self.app.state.phase = Phase.ABORTED
                    return {"execution_complete": True, "error": "No route to base"}
                else:
                    route_q.load(planned)
                    nxt = route_q.peek()
            
            if nxt:
                # Move to next position in route
                next_pos = RoverPosition(x=nxt[0], y=nxt[1])
                route_q.pop()  # Consume the coordinate
                return {
                    "current_action": {
                        "next_position": next_pos,
                        "action": "move",
                        "request_image": False,
                        "findings": "",
                        "reasoning": f"Following locked return route to base ({base[0]}, {base[1]})",
                        "status": "success"
                    },
                    "current_step_index": current_step_index,
                    "logs": []
                }
            else:
                # No next position available
                return {"execution_complete": True, "error": "Route queue empty"}
        
        # CRITICAL: When tasks are done, lock RETURN_TO_BASE phase once
        # Check if all non-return steps are completed
        task_steps = [s for s in steps if s.action != "return"]
        tasks_completed = len(task_steps) > 0 and all(step.completed for step in task_steps)
        
        if route_q and phase == Phase.EXECUTE_TASK and tasks_completed:
            # All task steps done - switch to RETURN_TO_BASE phase
            rover_pos = (rover_position.x, rover_position.y)
            obstacle_coords = [(o.x, o.y) for o in obstacles]
            
            # Compute return route
            planned = astar(10, 10, rover_pos, base, obstacle_coords, safety=0, diag=True, turn_penalty=0.10)
            if not planned:
                mission_state_manager.add_log(
                    mission_id,
                    MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.ROVER,
                        message="No safe route to base; will retry with relaxed safety.",
                        level="warning"
                    )
                )
                planned = astar(10, 10, rover_pos, base, obstacle_coords, safety=0, diag=True, turn_penalty=0.06)
            
            if not planned:
                mission_state_manager.add_log(
                    mission_id,
                    MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.ROVER,
                        message="Still blocked; aborting to avoid wandering.",
                        level="error"
                    )
                )
                mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
                if self.app:
                    self.app.state.phase = Phase.ABORTED
                return {"execution_complete": True, "error": "No route to base"}
            else:
                route_q.load(planned)
                if self.app:
                    self.app.state.phase = Phase.RETURN_TO_BASE
                mission_state_manager.add_log(
                    mission_id,
                    MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.ROVER,
                        message=f"Return route locked ({len(planned)} hops).",
                        level="success"
                    )
                )
                # Continue to consume route in next iteration
                # Fall through to RETURN_TO_BASE handling above
                cur = (rover_position.x, rover_position.y)
                route_q.discard_until(cur)
                nxt = route_q.peek()
                if nxt:
                    next_pos = RoverPosition(x=nxt[0], y=nxt[1])
                    route_q.pop()
                    return {
                        "current_action": {
                            "next_position": next_pos,
                            "action": "move",
                            "request_image": False,
                            "findings": "",
                            "reasoning": f"Following locked return route to base ({base[0]}, {base[1]})",
                            "status": "success"
                        },
                        "current_step_index": current_step_index,
                        "logs": []
                    }
        
        # Check if all steps are completed (using mission state manager as source of truth)
        if steps and all(step.completed for step in steps):
            # CRITICAL FIX: Verify return step is actually completed (rover is at base)
            return_steps = [s for s in steps if s.action == "return"]
            if return_steps:
                rover_pos = rover_position
                if isinstance(rover_pos, dict):
                    rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                if rover_pos.x != 0 or rover_pos.y != 0:
                    print(f"⚠️  All steps marked complete but rover at ({rover_pos.x}, {rover_pos.y}), not at base (0, 0). Continuing to return step...")
                    # Find return step and execute it
                    for i, step in enumerate(steps):
                        if step.action == "return" and not step.completed:
                            return {
                                "current_step_index": i,
                                "logs": []
                            }
                    # If return step is marked complete but rover not at base, unmark it
                    for step in return_steps:
                        if step.completed:
                            print(f"⚠️  Return step marked complete but rover not at base. Unmarking return step...")
                            mission_state_manager.update_step(mission_id, step.step_number, completed=False)
                            return {
                                "current_step_index": steps.index(step),
                                "logs": []
                            }
            # CRITICAL FIX: Even if no return step, verify rover is at base before completing
            rover_pos = rover_position
            if isinstance(rover_pos, dict):
                rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
            if rover_pos.x != 0 or rover_pos.y != 0:
                print(f"⚠️  All steps completed but rover at ({rover_pos.x}, {rover_pos.y}), not at base (0, 0). Cannot complete mission.")
                # Find or create return step
                return_steps = [s for s in steps if s.action == "return"]
                if return_steps:
                    return_step = return_steps[0]
                    if return_step.completed:
                        mission_state_manager.update_step(mission_id, return_step.step_number, completed=False)
                    return {
                        "current_step_index": steps.index(return_step),
                        "logs": []
                    }
            return {"execution_complete": True, "current_step_index": len(steps)}
        
        # CRITICAL FIX: Don't complete just because index exceeds length - check if return step is completed
        if current_step_index >= len(steps) and len(steps) > 0:
            # Check if return step exists and is completed
            return_steps = [s for s in steps if s.action == "return"]
            if return_steps:
                # Check if rover is at base
                rover_pos = rover_position
                if isinstance(rover_pos, dict):
                    rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                if rover_pos.x != 0 or rover_pos.y != 0:
                    # Rover not at base - find return step and execute it
                    for i, step in enumerate(steps):
                        if step.action == "return":
                            print(f"⚠️  Step index exceeds length but rover not at base. Executing return step at index {i}...")
                            return {
                                "current_step_index": i,
                                "logs": []
                            }
            # CRITICAL FIX: Even if no return step, verify rover is at base before completing
            rover_pos = rover_position
            if isinstance(rover_pos, dict):
                rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
            if rover_pos.x != 0 or rover_pos.y != 0:
                print(f"⚠️  Step index exceeds length but rover at ({rover_pos.x}, {rover_pos.y}), not at base (0, 0). Cannot complete mission.")
                # Find return step and execute it
                return_steps = [s for s in steps if s.action == "return"]
                if return_steps:
                    return_step = return_steps[0]
                    if return_step.completed:
                        mission_state_manager.update_step(mission_id, return_step.step_number, completed=False)
                    return {
                        "current_step_index": steps.index(return_step),
                        "logs": []
                    }
            # Rover is at base, mission can complete
            return {"execution_complete": True, "current_step_index": len(steps) if steps else 0}
        
        if len(steps) == 0:
            return {"execution_complete": True, "current_step_index": 0}
        
        current_step = steps[current_step_index]
        
        # CRITICAL FIX: If first step is a return step, skip it and find first move/explore step
        if current_step_index == 0 and current_step.action == "return":
            print(f"⚠️  WARNING: First step is a return step! Skipping to find first move/explore step...")
            # Find first non-return step
            found_step = False
            for i, step in enumerate(steps):
                if step.action != "return":
                    print(f"✅ Found first non-return step at index {i}: {step.action} to ({step.target_position.x if step.target_position else 'N/A'}, {step.target_position.y if step.target_position else 'N/A'})")
                    current_step = step
                    current_step_index = i
                    # Update state with correct step index
                    mission_state_manager.set_current_step(mission_id, current_step_index + 1)
                    found_step = True
                    break
            
            # If no non-return step found, return error
            if not found_step:
                print(f"❌ ERROR: All steps are return steps! Cannot execute mission.")
                return {
                    "error": "All steps are return steps. Cannot execute mission.",
                    "execution_complete": True,
                    "current_step_index": len(steps)
                }
        
        # Check if current step is already completed - skip to next step
        if current_step.completed:
            # CRITICAL FIX: Before moving to next step, check if rover needs to return to base
            # If this is not a return step and rover is not at base, ensure return step is executed
            if current_step.action != "return":
                rover_pos = rover_position
                if isinstance(rover_pos, dict):
                    rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                
                # If rover is not at base, check if return step exists and needs to be executed
                if rover_pos.x != 0 or rover_pos.y != 0:
                    return_steps = [s for s in steps if s.action == "return"]
                    if return_steps:
                        return_step = return_steps[0]
                        # If return step is not completed, execute it
                        if not return_step.completed:
                            print(f"⚠️  Non-return step completed but rover at ({rover_pos.x}, {rover_pos.y}), not at base. Switching to return step...")
                            return_step_index = steps.index(return_step)
                            return {
                                "current_step_index": return_step_index,
                                "logs": []
                            }
            
            # Move to next step
            next_index = current_step_index + 1
            if next_index >= len(steps):
                # CRITICAL FIX: Before completing, check if rover is at base
                rover_pos = rover_position
                if isinstance(rover_pos, dict):
                    rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                
                # If rover is not at base, find and execute return step
                if rover_pos.x != 0 or rover_pos.y != 0:
                    return_steps = [s for s in steps if s.action == "return"]
                    if return_steps:
                        return_step = return_steps[0]
                        return_step_index = steps.index(return_step)
                        # Unmark return step if marked complete but rover not at base
                        if return_step.completed:
                            print(f"⚠️  All steps done but rover at ({rover_pos.x}, {rover_pos.y}), not at base. Unmarking return step and continuing...")
                            mission_state_manager.update_step(mission_id, return_step.step_number, completed=False)
                        print(f"⚠️  Switching to return step to bring rover back to base (0,0)...")
                        return {
                            "current_step_index": return_step_index,
                            "logs": []
                        }
                # All steps done and rover is at base
                return {"execution_complete": True, "current_step_index": next_index}
            return {
                "current_step_index": next_index,
                "logs": []
            }
        
        # Update agent status
        mission_state_manager.update_agent_status(mission_id, AgentType.ROVER, AgentStatus.EXECUTING)
        # Update current step (convert 0-indexed to 1-indexed for display)
        mission_state_manager.set_current_step(mission_id, current_step_index + 1)
        
        # Only log the step execution once when starting (to avoid spam when moving multiple times for same step)
        # Check if this is the first execution of this step by checking if step is not completed
        # CRITICAL: Log target position to debug navigation issues
        log_message = f"Executing step {current_step.step_number}: {current_step.action} - {current_step.description}"
        if current_step.target_position:
            log_message += f" (Current: ({rover_position.x}, {rover_position.y}), Target: ({current_step.target_position.x}, {current_step.target_position.y}))"
            # Debug: Print target to console
            print(f"🔍 DEBUG: Step {current_step.step_number} target: ({current_step.target_position.x}, {current_step.target_position.y}), Current: ({rover_position.x}, {rover_position.y})")
        else:
            print(f"⚠️  WARNING: Step {current_step.step_number} has no target_position!")
        
        log = MissionLog(
            mission_id=mission_id,
            agent_type=AgentType.ROVER,
            message=log_message,
            level="info"
        )
        # Only add log on first execution to avoid spam
        if not current_step.completed:
            mission_state_manager.add_log(mission_id, log)
        
        # CRITICAL: Check if target position is an obstacle - if so, abort immediately
        if current_step.target_position:
            target_pos = current_step.target_position
            obstacle_positions = {(o.x, o.y) for o in obstacles}
            if (target_pos.x, target_pos.y) in obstacle_positions:
                print(f"❌ CRITICAL: Target position ({target_pos.x}, {target_pos.y}) is blocked by an obstacle. Aborting mission and returning to base.")
                mission_state_manager.add_log(
                    mission_id,
                    MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.SUPERVISOR,
                        message=f"Target position ({target_pos.x}, {target_pos.y}) is blocked by an obstacle. Mission aborted. Returning to base (0,0).",
                        level="error"
                    )
                )
                mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
                # Switch to return phase to go back to base
                if self.app:
                    self.app.state.phase = Phase.RETURN_TO_BASE
                    # Compute return route
                    rover_pos = (rover_position.x, rover_position.y)
                    obstacle_coords = [(o.x, o.y) for o in obstacles]
                    planned = astar(10, 10, rover_pos, (0, 0), obstacle_coords, safety=0, diag=True, turn_penalty=0.10)
                    if planned:
                        route_q.load(planned)
                        return {
                            "current_action": {
                                "next_position": RoverPosition(x=planned[0][0], y=planned[0][1]) if planned else rover_position,
                                "action": "return",
                                "request_image": False,
                                "findings": "",
                                "reasoning": "Target blocked by obstacle. Returning to base.",
                                "status": "success"
                            },
                            "current_step_index": current_step_index,
                            "logs": []
                        }
                return {"execution_complete": True, "error": "Target position blocked by obstacle"}
        
        # CRITICAL: Detect infinite loops - check if rover is stuck between same positions
        if mission:
            recent_logs = mission.logs[-10:] if len(mission.logs) > 10 else mission.logs
            # Extract positions from recent logs
            recent_positions = []
            for log in recent_logs:
                if "Rover moved to position" in log.message or "Rover at" in log.message:
                    # Try to extract position from log message
                    import re
                    pos_match = re.search(r'\((\d+),\s*(\d+)\)', log.message)
                    if pos_match:
                        recent_positions.append((int(pos_match.group(1)), int(pos_match.group(2))))
            
            # Check if we're stuck in a loop (same 2-3 positions repeating)
            if len(recent_positions) >= 6:
                # Check last 6 positions for repeating pattern
                last_six = recent_positions[-6:]
                # Check if we're alternating between same positions
                if len(set(last_six)) <= 2:  # Only 1-2 unique positions in last 6 moves
                    # Check if target is blocked
                    if current_step.target_position:
                        target_pos = current_step.target_position
                        obstacle_positions = {(o.x, o.y) for o in obstacles}
                        if (target_pos.x, target_pos.y) in obstacle_positions:
                            print(f"❌ CRITICAL: Infinite loop detected. Target ({target_pos.x}, {target_pos.y}) is blocked. Aborting and returning to base.")
                            mission_state_manager.add_log(
                                mission_id,
                                MissionLog(
                                    mission_id=mission_id,
                                    agent_type=AgentType.SUPERVISOR,
                                    message=f"Infinite loop detected. Target position ({target_pos.x}, {target_pos.y}) is blocked by an obstacle. Mission aborted. Returning to base (0,0).",
                                    level="error"
                                )
                            )
                            mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
                            # Switch to return phase
                            if self.app:
                                self.app.state.phase = Phase.RETURN_TO_BASE
                                rover_pos = (rover_position.x, rover_position.y)
                                obstacle_coords = [(o.x, o.y) for o in obstacles]
                                planned = astar(10, 10, rover_pos, (0, 0), obstacle_coords, safety=0, diag=True, turn_penalty=0.10)
                                if planned:
                                    route_q.load(planned)
                                    return {
                                        "current_action": {
                                            "next_position": RoverPosition(x=planned[0][0], y=planned[0][1]) if planned else rover_position,
                                            "action": "return",
                                            "request_image": False,
                                            "findings": "",
                                            "reasoning": "Infinite loop detected. Target blocked. Returning to base.",
                                            "status": "success"
                                        },
                                        "current_step_index": current_step_index,
                                        "logs": []
                                    }
                            return {"execution_complete": True, "error": "Infinite loop detected - target blocked"}
        
        # Execute step - pass mission goal for context-aware findings generation
        goal = state.get("goal", "")
        action_result = await self.rover.execute_step(current_step, rover_position, obstacles, mission_goal=goal)
        
        # CRITICAL FIX: If rover agent returns "completed" status, mark step as complete
        if action_result.get("status") == "completed":
            print(f"✅ Rover agent returned completed status for step {current_step.step_number}")
            mission_state_manager.update_step(mission_id, current_step.step_number, completed=True)
        
        return {
            "current_action": action_result,
            "current_step_index": current_step_index,  # CRITICAL: Return updated step index if we skipped return step
            "logs": [log] if not current_step.completed else []
        }
    
    def _should_validate_or_complete(self, state: MissionGraphState) -> Literal["validate", "execute", "complete"]:
        """Determine if we need safety validation, execution, or completion"""
        # Check if execution is complete
        if state.get("execution_complete"):
            return "complete"
        
        mission_id = state.get("mission_id")
        current_step_index = state.get("current_step_index", 0)
        
        # Get steps from mission state manager (source of truth)
        mission = mission_state_manager.get_mission(mission_id) if mission_id else None
        if mission:
            steps = mission.steps
        else:
            steps = state.get("steps", [])
        
        # Check if all steps are completed (using mission state manager as source of truth)
        if steps and all(step.completed for step in steps):
            return "complete"
        
        if current_step_index >= len(steps) and len(steps) > 0:
            return "complete"
        
        current_action = state.get("current_action")
        if current_action and current_action.get("next_position"):
            return "validate"
        return "execute"
    
    async def _safety_node(self, state: MissionGraphState) -> Dict[str, Any]:
        """Safety agent node"""
        mission_id = state["mission_id"]
        current_action = state.get("current_action", {})
        rover_position = state.get("rover_position", RoverPosition(x=0, y=0))
        obstacles = state.get("obstacles", [])
        weather_data = state.get("weather_data")
        
        next_position_data = current_action.get("next_position")
        if not next_position_data:
            return {"safety_approved": False}
        
        # Handle both dict and RoverPosition object
        if isinstance(next_position_data, dict):
            next_position = RoverPosition(x=next_position_data.get("x", 0), y=next_position_data.get("y", 0))
        elif isinstance(next_position_data, RoverPosition):
            next_position = next_position_data
        else:
            return {"safety_approved": False}
        
        # Update agent status
        mission_state_manager.update_agent_status(mission_id, AgentType.SAFETY, AgentStatus.VALIDATING)
        
        # Validate move
        validation_result = await self.safety.validate_move(
            rover_position, 
            next_position, 
            obstacles, 
            weather_data
        )
        
        log = MissionLog(
            mission_id=mission_id,
            agent_type=AgentType.SAFETY,
            message=f"Safety check: {validation_result.get('reason', 'Unknown')}",
            level="success" if validation_result.get("approved") else "warning"
        )
        mission_state_manager.add_log(mission_id, log)
        
        mission_state_manager.update_agent_status(mission_id, AgentType.SAFETY, AgentStatus.IDLE)
        
        return {
            "safety_approved": validation_result.get("approved", False),
            "current_action": {
                **current_action,
                "validation": validation_result
            },
            "logs": [log]
        }
    
    def _safety_decision(self, state: MissionGraphState) -> Literal["approved", "rejected", "abort"]:
        """Determine next step based on safety validation - Safety First"""
        if state.get("error"):
            return "abort"

        if state.get("safety_approved"):
            return "approved"

        # CRITICAL: Safety First - if safety agent flags move as unsafe, do not apply it
        # Mark as no-progress and proceed with detour logic (never recurse/loop inside safety handling)
        mission_id = state.get("mission_id")
        mission = mission_state_manager.get_mission(mission_id) if mission_id else None
        
        # CRITICAL: Check navigation state for stuck count
        nav_state = mission_state_manager.get_navigation_state(mission_id)
        stuck_count = nav_state.get("stuck_count", 0)
        
        # If safety rejects, increment stuck count (treat as no-progress)
        if mission:
            # Get current navigation state
            current_step_visited = nav_state.get("current_step_visited", set())
            
            # Check if we're stuck (too many rejections)
            recent_logs = mission.logs[-20:] if len(mission.logs) > 20 else mission.logs
            rejection_count = sum(1 for log in recent_logs 
                                 if "Obstacle detected" in log.message 
                                 or "cannot move" in log.message.lower()
                                 or (log.level == "warning" and "rejected" in log.message.lower()))
            
            # CRITICAL: Only abort if truly stuck (STUCK_LIMIT rejections) - allow rover to find alternative paths
            if rejection_count >= STUCK_LIMIT or stuck_count >= STUCK_LIMIT:
                mission_state_manager.add_log(
                    mission_id,
                    MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.SUPERVISOR,
                        message=f"Rover unable to find alternative path after {rejection_count} attempts. Aborting mission and returning to base (0,0).",
                        level="warning"
                    )
                )
                return "abort"

        # CRITICAL FIX: Do NOT abort immediately on obstacle detection
        # Instead, reject the move and let rover find alternative path
        current_action = state.get("current_action", {})
        validation = current_action.get("validation", {})
        reason = validation.get("reason", "").lower()
        
        # If obstacle detected, reject the move (don't abort) - rover will try alternative path
        if "obstacle" in reason:
            if mission_id:
                # CRITICAL FIX: Add the rejected position to obstacles list so rover knows about it
                next_position_data = current_action.get("next_position")
                if next_position_data:
                    if isinstance(next_position_data, dict):
                        rejected_pos = RoverPosition(x=next_position_data.get("x", 0), y=next_position_data.get("y", 0))
                    elif isinstance(next_position_data, RoverPosition):
                        rejected_pos = next_position_data
                    else:
                        rejected_pos = None
                    
                    if rejected_pos:
                        # Add rejected position to obstacles list
                        mission_state_manager.add_obstacle(mission_id, rejected_pos)
                        # Update obstacles in state so rover knows about it
                        mission = mission_state_manager.get_mission(mission_id)
                        if mission:
                            state["obstacles"] = mission.obstacles
                
                mission_state_manager.add_log(
                    mission_id,
                    MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.SAFETY,
                        message=f"Obstacle detected at proposed position. Finding alternative path to reach destination.",
                        level="warning"
                    )
                )
            # Return "rejected" instead of "abort" - this allows rover to try alternative paths
            return "rejected"

        return "rejected"
    
    async def _update_position_node(self, state: MissionGraphState) -> Dict[str, Any]:
        """Update rover position and fetch NASA images"""
        mission_id = state["mission_id"]
        current_action = state.get("current_action", {})
        steps = state.get("steps", [])
        current_step_index = state.get("current_step_index", 0)
        rover_position = state.get("rover_position", RoverPosition(x=0, y=0))
        
        # Get obstacles from mission state manager
        mission = mission_state_manager.get_mission(mission_id) if mission_id else None
        if mission:
            obstacles = mission.obstacles
        else:
            obstacles = state.get("obstacles", [])
        
        # Get phase machine state
        if self.app:
            route_q = self.app.state.route_q
        else:
            route_q = None
        
        # CRITICAL: Get navigation state for loop detection
        nav_state = mission_state_manager.get_navigation_state(mission_id)
        tick_count = nav_state.get("tick_count", 0) + 1
        stuck_count = nav_state.get("stuck_count", 0)
        prev_pos = nav_state.get("prev_pos")
        current_step_visited = nav_state.get("current_step_visited", set())
        
        # CRITICAL: Hard timebox check
        if tick_count >= MAX_TICKS_PER_MISSION:
            print(f"⏱️  Timebox exceeded ({tick_count} ticks). Aborting mission gracefully.")
            mission_state_manager.add_log(
                mission_id,
                MissionLog(
                    mission_id=mission_id,
                    agent_type=AgentType.SUPERVISOR,
                    message=f"⏱️ Timebox exceeded ({tick_count} ticks). Mission aborted.",
                    level="error"
                )
            )
            mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
            return {
                "rover_position": rover_position,
                "execution_complete": True,
                "error": f"Timebox exceeded ({tick_count} ticks)"
            }
        
        new_position = rover_position
        next_position_data = current_action.get("next_position")
        step_completed = False
        
        # CRITICAL: Check deterministic exit criteria BEFORE processing movement
        if current_step_index < len(steps):
            current_step = steps[current_step_index]
            if current_step.target_position:
                target_pos = (current_step.target_position.x, current_step.target_position.y)
                curr_pos = (rover_position.x, rover_position.y)
                
                # Exit criterion a): At target
                if is_at_target(curr_pos, target_pos, tol=0.5):  # tol=0.5 for grid (allow 1 cell tolerance)
                    print(f"🎯 Target reached: ({rover_position.x}, {rover_position.y}) == ({current_step.target_position.x}, {current_step.target_position.y})")
                    mission_state_manager.add_log(
                        mission_id,
                        MissionLog(
                            mission_id=mission_id,
                            agent_type=AgentType.ROVER,
                            message=f"🎯 Target reached at ({rover_position.x}, {rover_position.y})",
                            level="success"
                        )
                    )
                    mission_state_manager.update_step(mission_id, current_step.step_number, completed=True)
                    mission_state_manager.reset_navigation_state(mission_id)  # Reset for next step
                    return {
                        "rover_position": rover_position,
                        "current_action": {},
                        "logs": []
                    }
        
        # Exit criterion b): Steps exhausted
        if current_step_index >= len(steps):
            print(f"✅ Steps exhausted (index {current_step_index} >= {len(steps)})")
            mission_state_manager.add_log(
                mission_id,
                MissionLog(
                    mission_id=mission_id,
                    agent_type=AgentType.SUPERVISOR,
                    message="✅ Steps exhausted",
                    level="info"
                )
            )
            return {
                "rover_position": rover_position,
                "execution_complete": True,
                "logs": []
            }
        
        # Check if we need to move or if we're already at target
        if next_position_data:
            # Handle both dict and RoverPosition object
            if isinstance(next_position_data, dict):
                new_position = RoverPosition(x=next_position_data.get("x", rover_position.x), y=next_position_data.get("y", rover_position.y))
            elif isinstance(next_position_data, RoverPosition):
                new_position = next_position_data
            else:
                new_position = rover_position
            
            # CRITICAL: Stall/no-progress guard - check if position changed
            pos_changed = (new_position.x != rover_position.x or new_position.y != rover_position.y)
            new_pos_tuple = (new_position.x, new_position.y)
            
            # CRITICAL: Check if we're stuck (same position or revisiting)
            if not pos_changed or new_pos_tuple in current_step_visited:
                # No progress - increment stuck count
                stuck_count += 1
                print(f"⚠️  No progress detected (stuck_count={stuck_count}): position unchanged or revisiting {new_pos_tuple}")
                
                # Exit criterion c): Stuck limit hit
                if stuck_count >= STUCK_LIMIT:
                    print(f"🧯 Stuck limit hit ({stuck_count} consecutive no-progress iterations). Aborting mission.")
                    mission_state_manager.add_log(
                        mission_id,
                        MissionLog(
                            mission_id=mission_id,
                            agent_type=AgentType.SUPERVISOR,
                            message=f"🧯 Stuck limit hit ({stuck_count} consecutive no-progress iterations). Mission aborted.",
                            level="error"
                        )
                    )
                    mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
                    # Log structured summary
                    if current_step_index < len(steps):
                        current_step = steps[current_step_index]
                        if current_step.target_position:
                            target_pos = (current_step.target_position.x, current_step.target_position.y)
                            distance = calculate_distance(new_pos_tuple, target_pos)
                            print(f"📊 Exit summary: reason=stuck, ticks={tick_count}, final_pos={new_pos_tuple}, target={target_pos}, distance={distance}")
                    return {
                        "rover_position": new_position,
                        "execution_complete": True,
                        "error": f"Stuck limit hit ({stuck_count} iterations)"
                    }
            else:
                # Progress made - reset stuck count
                stuck_count = 0
            
            # Always add new position to visited set
            current_step_visited.add(new_pos_tuple)
            
            # CRITICAL FIX: Guard against infinite loop - if position hasn't changed, check if we're at target
            if not pos_changed:
                # Position hasn't changed - check if we're at target
                if current_step_index < len(steps):
                    current_step = steps[current_step_index]
                    if current_step.target_position:
                        at_target = (new_position.x == current_step.target_position.x and new_position.y == current_step.target_position.y)
                        if at_target:
                            # We're at target and position hasn't changed - mark step complete and return
                            print(f"✅ Position unchanged but at target ({new_position.x}, {new_position.y}). Marking step complete.")
                            mission_state_manager.update_step(mission_id, current_step.step_number, completed=True)
                            # Update state and return to prevent infinite loop
                            return {
                                "rover_position": new_position,
                                "current_action": {},
                                "logs": []
                            }
                        else:
                            # Position unchanged and NOT at target - check if target is blocked
                            target_pos = current_step.target_position
                            obstacle_positions = {(o.x, o.y) for o in obstacles} if obstacles else set()
                            if (target_pos.x, target_pos.y) in obstacle_positions:
                                # Target is blocked - abort and return to base
                                print(f"❌ CRITICAL: Position unchanged and target ({target_pos.x}, {target_pos.y}) is blocked. Aborting mission and returning to base.")
                                mission_state_manager.add_log(
                                    mission_id,
                                    MissionLog(
                                        mission_id=mission_id,
                                        agent_type=AgentType.SUPERVISOR,
                                        message=f"Target position ({target_pos.x}, {target_pos.y}) is blocked by an obstacle. Mission aborted. Returning to base (0,0).",
                                        level="error"
                                    )
                                )
                                mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
                                # Switch to return phase
                                if self.app:
                                    self.app.state.phase = Phase.RETURN_TO_BASE
                                    rover_pos = (rover_position.x, rover_position.y)
                                    obstacle_coords = [(o.x, o.y) for o in obstacles] if obstacles else []
                                    planned = astar(10, 10, rover_pos, (0, 0), obstacle_coords, safety=0, diag=True, turn_penalty=0.10)
                                    if planned:
                                        route_q.load(planned)
                                        return {
                                            "rover_position": new_position,
                                            "current_action": {
                                                "next_position": RoverPosition(x=planned[0][0], y=planned[0][1]) if planned else rover_position,
                                                "action": "return",
                                                "request_image": False,
                                                "findings": "",
                                                "reasoning": "Target blocked. Returning to base.",
                                                "status": "success"
                                            },
                                            "logs": []
                                        }
                                return {
                                    "rover_position": new_position,
                                    "execution_complete": True,
                                    "error": "Target position blocked by obstacle"
                                }
                            # Position unchanged and NOT at target - this is a stuck state
                            print(f"⚠️  WARNING: Position unchanged at ({new_position.x}, {new_position.y}) but target is ({current_step.target_position.x}, {current_step.target_position.y}). Preventing infinite loop.")
                            # Don't schedule another navigation step - return with current state
                            return {
                                "rover_position": new_position,
                                "current_action": {},
                                "logs": []
                            }
            else:
                # Position changed - update it
                mission_state_manager.update_rover_position(mission_id, new_position)
                
                # CRITICAL: Update navigation state with new position and reset stuck count
                mission_state_manager.update_navigation_state(
                    mission_id,
                    tick_count=tick_count,
                    stuck_count=stuck_count,
                    prev_pos=new_pos_tuple,
                    current_step_visited=current_step_visited
                )
                
                # CRITICAL: Structured logging
                if current_step_index < len(steps):
                    current_step = steps[current_step_index]
                    if current_step.target_position:
                        target_pos = (current_step.target_position.x, current_step.target_position.y)
                        distance = calculate_distance(new_pos_tuple, target_pos)
                        log_data = {
                            "tick": tick_count,
                            "pos": [new_position.x, new_position.y],
                            "target": [current_step.target_position.x, current_step.target_position.y],
                            "step": current_step_index + 1,
                            "stuck": stuck_count,
                            "distance": distance
                        }
                        print(f"📊 Navigation tick: {json.dumps(log_data)}")
            
            # Fetch NASA image if requested - use next photo from pool for variety
            if current_action.get("request_image"):
                try:
                    # Get next photo from rotational pool - ensures different photo each step
                    photo = nasa_client.get_next_photo_from_pool()
                    if photo:
                        image_url = photo.get("img_src", "")
                        if image_url:
                            mission_state_manager.add_nasa_image(mission_id, image_url)

                            # Update step with image
                            if current_step_index < len(steps):
                                mission_state_manager.update_step(
                                    mission_id,
                                    steps[current_step_index].step_number,
                                    nasa_image_url=image_url
                                )
                except Exception as e:
                    print(f"Error fetching NASA image: {e}")
        else:
            # No next position means we're executing an action at current position
            # This happens for scan/collect actions or when already at target
            new_position = rover_position
        
        # Initialize updated_steps
        updated_steps = list(steps)  # Create a copy of steps list
        
        # Check if current step target is reached
        if current_step_index < len(steps):
            current_step = steps[current_step_index]
            target_position = current_step.target_position
            
            # Determine if step is complete:
            # 1. If no target position (scan/collect actions), complete after one execution
            # 2. If target position exists, complete ONLY when rover reaches it
            # 3. For "return" action, check if we're at (0, 0)
            if target_position is None:
                # No target - actions like scan/collect at current position
                # Complete if we executed the action (moved or stayed)
                step_completed = True
            elif current_step.action == "return":
                # Return action - complete when at base (0, 0)
                step_completed = (new_position.x == 0 and new_position.y == 0)
                # Debug: log return step completion status
                if step_completed:
                    print(f"✅ Return step {current_step.step_number} completed: Rover at ({new_position.x}, {new_position.y})")
                    # CRITICAL FIX: Mark step complete and update state BEFORE emitting logs
                    mission_state_manager.update_step(mission_id, current_step.step_number, completed=True)
                else:
                    # CRITICAL FIX: Only log if position actually changed to avoid spam
                    if new_position.x != rover_position.x or new_position.y != rover_position.y:
                        print(f"❌ Return step {current_step.step_number} NOT complete: Rover at ({new_position.x}, {new_position.y}), Target is (0, 0). Continuing...")
                        # Log warning if return step is not complete
                        mission_state_manager.add_log(
                            mission_id,
                            MissionLog(
                                mission_id=mission_id,
                                agent_type=AgentType.ROVER,
                                message=f"Return step not complete: Rover at ({new_position.x}, {new_position.y}), Target is (0, 0). Continuing navigation...",
                                level="info"
                            )
                        )
            else:
                # CRITICAL FIX: Movement/explore action - complete ONLY when at target
                # Do NOT mark complete if rover is not at the exact target coordinates
                step_completed = (new_position.x == target_position.x and new_position.y == target_position.y)
                
                # Debug: log step completion status
                if step_completed:
                    print(f"✅ Step {current_step.step_number} completed: Rover at ({new_position.x}, {new_position.y}), Target was ({target_position.x}, {target_position.y})")
                else:
                    print(f"❌ Step {current_step.step_number} NOT complete: Rover at ({new_position.x}, {new_position.y}), Target is ({target_position.x}, {target_position.y})")
                    
                # CRITICAL: If step is not complete, ensure we don't mark it as complete
                if not step_completed:
                    # Log warning if we're trying to complete a step that's not at target
                    mission_state_manager.add_log(
                        mission_id,
                        MissionLog(
                            mission_id=mission_id,
                            agent_type=AgentType.ROVER,
                            message=f"Step {current_step.step_number} not complete: Rover at ({new_position.x}, {new_position.y}), Target is ({target_position.x}, {target_position.y}). Continuing navigation...",
                            level="info"
                        )
                    )
            if step_completed:
                # Update step in mission state manager
                mission_state_manager.update_step(mission_id, steps[current_step_index].step_number, completed=True)
                # Create updated step object
                updated_step = MissionStep(
                    step_number=current_step.step_number,
                    action=current_step.action,
                    target_position=current_step.target_position,
                    description=current_step.description,
                    completed=True,
                    nasa_image_url=current_step.nasa_image_url
                )
                updated_steps[current_step_index] = updated_step
                new_step_index = current_step_index + 1
                # Update current step display (1-indexed)
                mission_state_manager.set_current_step(mission_id, new_step_index + 1)
                
                # Check if we have findings from the action (collect, scan, explore)
                findings = current_action.get("findings", "")
                if findings:
                    # Store collected data
                    collected_data = {
                        "step_number": current_step.step_number,
                        "action": current_step.action,
                        "position": {"x": new_position.x, "y": new_position.y},
                        "findings": findings,
                        "timestamp": datetime.now().isoformat()
                    }
                    mission_state_manager.add_collected_data(mission_id, collected_data)
                    
                    log = MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.ROVER,
                        message=f"Step {current_step.step_number} completed: {findings}",
                        level="success"
                    )
                else:
                    log = MissionLog(
                        mission_id=mission_id,
                        agent_type=AgentType.ROVER,
                        message=f"Step {current_step.step_number} completed: Rover reached position ({new_position.x}, {new_position.y})",
                        level="success"
                    )
            else:
                # Step not complete yet, stay on same step
                new_step_index = current_step_index
                # Update current step display (1-indexed)
                mission_state_manager.set_current_step(mission_id, current_step_index + 1)
                distance_x = abs(target_position.x - new_position.x) if target_position else 0
                distance_y = abs(target_position.y - new_position.y) if target_position else 0
                
                log = MissionLog(
                    mission_id=mission_id,
                    agent_type=AgentType.ROVER,
                    message=f"Rover moved to position ({new_position.x}, {new_position.y}). Target: ({target_position.x if target_position else 'N/A'}, {target_position.y if target_position else 'N/A'}). Distance: ({distance_x}, {distance_y})",
                    level="info"
                )
        else:
            # No steps or index out of bounds
            new_step_index = current_step_index
            updated_steps = steps
            log = MissionLog(
                mission_id=mission_id,
                agent_type=AgentType.ROVER,
                message=f"Rover at position ({new_position.x}, {new_position.y})" if next_position_data else "Action completed",
                level="success"
            )
        
        mission_state_manager.add_log(mission_id, log)
        
        # CRITICAL FIX: Check if we've completed all steps
        # Only mark complete if ALL steps are actually completed AND rover is at all target positions
        execution_complete = False
        if updated_steps:
            all_completed = all(step.completed for step in updated_steps)
            
            # CRITICAL: Validate that rover actually reached all target positions
            # Check if any move/explore step is marked complete but rover isn't at target
            for i, step in enumerate(updated_steps):
                if step.completed and step.target_position:
                    # Get current rover position from mission state
                    mission = mission_state_manager.get_mission(mission_id)
                    if mission:
                        rover_pos = mission.rover_position
                        if isinstance(rover_pos, dict):
                            rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                        
                        # If step is marked complete but rover isn't at target, it's a false completion
                        if step.action != "return" and (rover_pos.x != step.target_position.x or rover_pos.y != step.target_position.y):
                            print(f"❌ FALSE COMPLETION DETECTED: Step {step.step_number} marked complete but rover at ({rover_pos.x}, {rover_pos.y}), target is ({step.target_position.x}, {step.target_position.y})")
                            # Unmark as complete
                            step.completed = False
                            mission_state_manager.update_step(mission_id, step.step_number, completed=False)
                            all_completed = False
            
            # CRITICAL: Only mark complete if ALL steps are completed
            execution_complete = all_completed
            
            if not all_completed and new_step_index >= len(updated_steps):
                # Log warning if we're trying to complete without finishing all steps
                print(f"⚠️  Warning: Step index ({new_step_index}) exceeds steps length ({len(updated_steps)}), but not all steps are completed. Continuing...")
                execution_complete = False  # Force continue
        
        return {
            "rover_position": new_position,
            "current_step_index": new_step_index,
            "execution_complete": execution_complete,
            "steps": updated_steps,  # Return updated steps so graph state reflects completion
            "logs": [log]
        }
    
    def _should_continue(self, state: MissionGraphState) -> Literal["continue", "complete", "abort"]:
        """Determine if mission should continue with deterministic exit criteria"""
        if state.get("error"):
            return "abort"

        if state.get("execution_complete"):
            return "complete"

        mission_id = state.get("mission_id")
        current_step_index = state.get("current_step_index", 0)
        
        # CRITICAL: Check navigation state for loop detection
        nav_state = mission_state_manager.get_navigation_state(mission_id)
        tick_count = nav_state.get("tick_count", 0)
        stuck_count = nav_state.get("stuck_count", 0)
        
        # Exit criterion: Hard timebox
        if tick_count >= MAX_TICKS_PER_MISSION:
            print(f"⏱️  Timebox exceeded in _should_continue ({tick_count} ticks)")
            return "abort"
        
        # Exit criterion: Stuck limit
        if stuck_count >= STUCK_LIMIT:
            print(f"🧯 Stuck limit hit in _should_continue ({stuck_count} iterations)")
            return "abort"

        # Get steps from mission state manager (source of truth) instead of graph state
        mission = mission_state_manager.get_mission(mission_id) if mission_id else None
        if mission:
            steps = mission.steps
        else:
            steps = state.get("steps", [])

        # CRITICAL FIX: Check if all steps are completed, including return-to-base step
        if steps:
            all_completed = all(step.completed for step in steps)
            
            # CRITICAL: Verify return step is completed (rover is at base)
            if all_completed:
                # Check if there's a return step and if rover is at base
                return_steps = [s for s in steps if s.action == "return"]
                if return_steps:
                    # Get rover position
                    rover_pos = mission.rover_position if mission else state.get("rover_position", RoverPosition(x=0, y=0))
                    if isinstance(rover_pos, dict):
                        rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                    
                    # If rover is not at base (0,0), mission is not complete
                    if rover_pos.x != 0 or rover_pos.y != 0:
                        print(f"⚠️  Mission not complete: Return step marked complete but rover at ({rover_pos.x}, {rover_pos.y}), not at base (0, 0)")
                        return "continue"  # Force continue until rover reaches base
                
                return "complete"

        # CRITICAL FIX: Don't complete just because index exceeds length
        # Only complete if ALL steps are actually completed AND rover is at base
        if current_step_index >= len(steps) and len(steps) > 0:
            # Check if all steps are actually completed
            all_completed = all(step.completed for step in steps)
            if not all_completed:
                print(f"⚠️  Warning: Step index ({current_step_index}) exceeds steps length ({len(steps)}), but not all steps completed. Continuing...")
                return "continue"  # Force continue, don't complete
            
            # CRITICAL FIX: Check if return step is completed (rover is at base)
            return_steps = [s for s in steps if s.action == "return"]
            if return_steps:
                rover_pos = mission.rover_position if mission else state.get("rover_position", RoverPosition(x=0, y=0))
                if isinstance(rover_pos, dict):
                    rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                
                # If rover is not at base (0,0), continue executing return step
                if rover_pos.x != 0 or rover_pos.y != 0:
                    print(f"⚠️  Step index exceeds length but rover at ({rover_pos.x}, {rover_pos.y}), not at base (0, 0). Continuing return step...")
                    # Find return step and continue executing it
                    for i, step in enumerate(steps):
                        if step.action == "return":
                            # Unmark return step if marked complete but rover not at base
                            if step.completed:
                                print(f"⚠️  Return step marked complete but rover not at base. Unmarking and continuing...")
                                mission_state_manager.update_step(mission_id, step.step_number, completed=False)
                            return {
                                "current_step_index": i,
                                "logs": []
                            }
                    return "continue"  # Force continue until rover reaches base
            
            return "complete"

        # Safety: prevent infinite loops
        # If we have no steps, complete
        if len(steps) == 0:
            return "complete"

        # CRITICAL: Check if current step is incomplete - must continue until target is reached
        if current_step_index < len(steps):
            current_step = steps[current_step_index]
            if not current_step.completed:
                # Step is not complete - must continue
                # Get current rover position to check if we're at target
                rover_pos = mission.rover_position if mission else state.get("rover_position", RoverPosition(x=0, y=0))
                if isinstance(rover_pos, dict):
                    rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                
                # CRITICAL FIX: For return step, check if rover is at base (0, 0)
                if current_step.action == "return":
                    target_pos = (0, 0)
                    curr_pos = (rover_pos.x, rover_pos.y)
                    if not is_at_target(curr_pos, target_pos, tol=0.5):
                        print(f"DEBUG: Return step not complete. Rover at ({rover_pos.x}, {rover_pos.y}), Target is (0, 0). Continuing...")
                        return "continue"
                elif current_step.target_position:
                    target_pos = (current_step.target_position.x, current_step.target_position.y)
                    curr_pos = (rover_pos.x, rover_pos.y)
                    if not is_at_target(curr_pos, target_pos, tol=0.5):
                        # Not at target yet - must continue
                        print(f"DEBUG: Step {current_step.step_number} not complete. Rover at ({rover_pos.x}, {rover_pos.y}), Target is ({current_step.target_position.x}, {current_step.target_position.y}). Continuing...")
                        return "continue"

        # CRITICAL FIX: Check for obstacle-blocked missions - only abort if truly stuck
        # Allow rover to find alternative paths instead of aborting immediately
        if mission:
            total_logs = len(mission.logs)
            recent_logs = mission.logs[-30:] if len(mission.logs) > 30 else mission.logs

            # Count warning-level obstacle blocks (but allow alternative pathfinding)
            obstacle_warnings = sum(1 for log in recent_logs if log.level == "warning" and ("adjacent to" in log.message.lower() or "at risk" in log.message.lower()))

            # CRITICAL FIX: Only abort if truly stuck (20+ warnings) - increased threshold to allow pathfinding
            if obstacle_warnings >= 20:
                # Insert return-to-base step if not already there
                has_return_step = any(step.action == "return" for step in steps)
                if not has_return_step and len(steps) > 0:
                    # Mark all incomplete steps as having obstacle interruption
                    for step in steps:
                        if not step.completed:
                            mission_state_manager.add_log(
                                mission_id,
                                MissionLog(
                                    mission_id=mission_id,
                                    agent_type=AgentType.SUPERVISOR,
                                    message=f"Step {step.step_number} blocked by obstacle. Initiating emergency return to base (0,0).",
                                    level="warning"
                                )
                            )
                return "abort"

        # Additional safety: if we've been on the same step for too long without progress
        if current_step_index < len(steps):
            current_step = steps[current_step_index]
            # If step is already completed, we should have advanced - this is a safety check
            if current_step.completed:
                # Step is marked complete but we're still on it - allow one more iteration to advance
                pass

        return "continue"
    
    async def _emergency_return_node(self, state: MissionGraphState) -> Dict[str, Any]:
        """Emergency return to base when mission is aborted"""
        mission_id = state["mission_id"]
        rover_position = state.get("rover_position", RoverPosition(x=0, y=0))

        # Update mission status to aborted
        mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)

        log = MissionLog(
            mission_id=mission_id,
            agent_type=AgentType.SUPERVISOR,
            message=f"Emergency return initiated from position ({rover_position.x}, {rover_position.y}) back to base (0,0)",
            level="warning"
        )
        mission_state_manager.add_log(mission_id, log)

        # Move rover back to (0,0) step by step
        current_pos = rover_position
        while current_pos.x != 0 or current_pos.y != 0:
            # Move towards (0,0)
            new_x = current_pos.x - 1 if current_pos.x > 0 else current_pos.x + 1 if current_pos.x < 0 else 0
            new_y = current_pos.y - 1 if current_pos.y > 0 else current_pos.y + 1 if current_pos.y < 0 else 0

            new_pos = RoverPosition(x=new_x, y=new_y)
            mission_state_manager.update_rover_position(mission_id, new_pos)

            log = MissionLog(
                mission_id=mission_id,
                agent_type=AgentType.ROVER,
                message=f"Returning to base: Rover moved to ({new_pos.x}, {new_pos.y})",
                level="info"
            )
            mission_state_manager.add_log(mission_id, log)

            current_pos = new_pos

        # Final log
        log = MissionLog(
            mission_id=mission_id,
            agent_type=AgentType.SUPERVISOR,
            message="Rover safely returned to base (0,0). Mission aborted due to obstacle detection. Mission incomplete.",
            level="warning"
        )
        mission_state_manager.add_log(mission_id, log)

        return {
            "status": MissionStatus.ABORTED,
            "rover_position": RoverPosition(x=0, y=0),
            "logs": [log],
            "execution_complete": True
        }

    async def _reporter_node(self, state: MissionGraphState) -> Dict[str, Any]:
        """Reporter agent node"""
        mission_id = state["mission_id"]

        mission_state_manager.update_agent_status(mission_id, AgentType.REPORTER, AgentStatus.REPORTING)

        # Get final mission status from mission state manager (source of truth)
        mission = mission_state_manager.get_mission(mission_id)
        final_status = mission.status if mission else state.get("status", MissionStatus.COMPLETE)
        
        # CRITICAL FIX: Only check for ACTUAL obstacle blocking, not just mentions
        # Only set to ABORTED if there's clear evidence of obstacle blocking the mission
        if mission and mission.logs:
            recent_logs = mission.logs[-15:]  # Check last 15 logs
            # CRITICAL: Only detect actual obstacle blocking, not just mentions
            # Look for specific patterns that indicate obstacle blocking:
            obstacle_blocking = any(
                ("obstacle detected at position" in log.message.lower() and log.level == "warning") or
                ("aborted due to obstacle" in log.message.lower()) or
                ("blocked by obstacle" in log.message.lower()) or
                ("unable to find alternative path" in log.message.lower()) or
                ("too many safety rejections" in log.message.lower())
                for log in recent_logs
            )
            if obstacle_blocking and final_status != MissionStatus.ABORTED:
                print(f"⚠️  Mission aborted due to actual obstacle blocking detected in logs")
                final_status = MissionStatus.ABORTED
                mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
        
        # CRITICAL: Validate mission actually completed - check if rover reached all targets AND returned to base
        # BUT: Skip validation if mission was aborted (obstacle detected)
        if final_status != MissionStatus.ABORTED and mission and mission.steps:
            all_steps_completed = all(step.completed for step in mission.steps)
            if not all_steps_completed:
                print(f"❌ FALSE COMPLETION: Mission marked complete but not all steps completed!")
                final_status = MissionStatus.ERROR
                mission_state_manager.update_mission_status(mission_id, MissionStatus.ERROR)
            else:
                # Double-check: verify rover is at target positions for all move/explore steps
                rover_pos = mission.rover_position
                if isinstance(rover_pos, dict):
                    rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                
                for step in mission.steps:
                    if step.target_position and step.action in ["move", "explore"]:
                        if rover_pos.x != step.target_position.x or rover_pos.y != step.target_position.y:
                            print(f"❌ FALSE COMPLETION: Step {step.step_number} target ({step.target_position.x}, {step.target_position.y}) not reached! Rover at ({rover_pos.x}, {rover_pos.y})")
                            final_status = MissionStatus.ERROR
                            mission_state_manager.update_mission_status(mission_id, MissionStatus.ERROR)
                            break
                
                # CRITICAL FIX: Verify rover is at base (0,0) for mission completion
                # Check if there's a return step - if so, rover must be at base
                return_steps = [s for s in mission.steps if s.action == "return"]
                if return_steps:
                    if rover_pos.x != 0 or rover_pos.y != 0:
                        print(f"❌ FALSE COMPLETION: Return step completed but rover at ({rover_pos.x}, {rover_pos.y}), not at base (0, 0)!")
                        final_status = MissionStatus.ERROR
                        mission_state_manager.update_mission_status(mission_id, MissionStatus.ERROR)

        # CRITICAL FIX: Only set to COMPLETE if not ABORTED or ERROR AND rover is at base
        if final_status != MissionStatus.ABORTED and final_status != MissionStatus.ERROR:
            # Final check: rover must be at base (0,0) for mission completion
            if mission:
                rover_pos = mission.rover_position
                if isinstance(rover_pos, dict):
                    rover_pos = RoverPosition(x=rover_pos.get("x", 0), y=rover_pos.get("y", 0))
                
                # Check if there's a return step - if so, rover must be at base
                return_steps = [s for s in mission.steps if s.action == "return"]
                if return_steps:
                    if rover_pos.x != 0 or rover_pos.y != 0:
                        print(f"⚠️  Mission cannot complete: Rover at ({rover_pos.x}, {rover_pos.y}), must return to base (0, 0)")
                        final_status = MissionStatus.ERROR
                        mission_state_manager.update_mission_status(mission_id, MissionStatus.ERROR)
                    else:
                        mission_state_manager.update_mission_status(mission_id, MissionStatus.COMPLETE)
                else:
                    # No return step - mission can complete without returning to base
                    mission_state_manager.update_mission_status(mission_id, MissionStatus.COMPLETE)
            else:
                mission_state_manager.update_mission_status(mission_id, MissionStatus.COMPLETE)
        else:
            # Ensure status is set correctly in state for reporter
            state["status"] = final_status

        # Update state with final status before generating report
        state["status"] = final_status
        report_data = await self.reporter.generate_report(state)

        log = MissionLog(
            mission_id=mission_id,
            agent_type=AgentType.REPORTER,
            message=f"Mission report generated: {report_data.get('outcome', 'Unknown')}",
            level="success" if final_status == MissionStatus.COMPLETE else "warning"
        )
        mission_state_manager.add_log(mission_id, log)
        mission_state_manager.update_agent_status(mission_id, AgentType.REPORTER, AgentStatus.COMPLETE)

        return {
            "status": final_status,
            "logs": [log]
        }
    
    async def execute_mission(self, mission_id: str, initial_state: Dict[str, Any], broadcast_callback=None) -> Dict[str, Any]:
        """Execute a mission using the LangGraph with optional broadcast callback"""
        try:
            # Create initial graph state
            graph_state: MissionGraphState = {
                "mission_id": mission_id,
                "goal": initial_state["goal"],
                "status": MissionStatus.PENDING,
                "steps": [],
                "current_step_index": 0,
                "rover_position": RoverPosition(x=0, y=0),
                "obstacles": initial_state.get("obstacles", []),
                "goal_positions": [],
                "logs": [],
                "agent_states": {},
                "nasa_images": [],
                "weather_data": None,
                "current_action": None,
                "safety_approved": None,
                "execution_complete": False,
                "error": None
            }
            
            # Stream execution and broadcast updates
            # Set recursion limit based on expected mission complexity
            # Each step might take up to 18 moves (diagonal across 10x10 grid)
            # With 8 steps max, that's ~144 moves, plus planning/validation overhead
            # Set to 500 to handle obstacle-blocked scenarios with retries
            config = {"recursion_limit": 500}
            
            final_state = None
            try:
                async for state_update in self.graph.astream(graph_state, config=config):
                    # state_update is a dict with node names as keys, values are state updates
                    # Merge all state updates into the graph state
                    if state_update:
                        for node_name, node_state in state_update.items():
                            if isinstance(node_state, dict):
                                # Update graph state with node state
                                graph_state.update(node_state)
                                
                                # Ensure steps list is properly updated (LangGraph should handle this, but be explicit)
                                if "steps" in node_state:
                                    graph_state["steps"] = node_state["steps"]
                        
                        # Use the merged state as final state
                        final_state = graph_state.copy()
                        
                        # Broadcast updates if callback provided
                        if broadcast_callback:
                            # Get current mission state for broadcasting
                            mission = mission_state_manager.get_mission(mission_id)
                            if mission:
                                await broadcast_callback({
                                    "type": "update",
                                    "mission_id": mission_id,
                                    "data": {
                                        "rover_position": {"x": mission.rover_position.x, "y": mission.rover_position.y},
                                        "current_step": mission.current_step,
                                        "total_steps": len(mission.steps),
                                        "status": mission.status.value,
                                        "agent_states": {k.value: v.value for k, v in mission.agent_states.items()},
                                        "logs": [{
                                            "mission_id": log.mission_id,
                                            "timestamp": log.timestamp.isoformat(),
                                            "agent_type": log.agent_type.value,
                                            "message": log.message,
                                            "level": log.level
                                        } for log in mission.logs[-10:]]  # Last 10 logs for better visibility
                                    }
                                })
            except Exception as stream_error:
                import traceback
                traceback.print_exc()
                print(f"Error in streaming execution: {stream_error}")
                # Fall back to regular invocation with config
                try:
                    final_state = await self.graph.ainvoke(graph_state, config=config)
                except Exception as invoke_error:
                    print(f"Error in regular invocation: {invoke_error}")
                    raise
            
            # If no streaming happened, run normally
            if final_state is None:
                final_state = await self.graph.ainvoke(graph_state, config=config)
            
            return final_state
            
        except Exception as e:
            mission_state_manager.update_mission_status(mission_id, MissionStatus.ERROR)
            log = MissionLog(
                mission_id=mission_id,
                agent_type=AgentType.SUPERVISOR,
                message=f"Mission execution error: {str(e)}",
                level="error"
            )
            mission_state_manager.add_log(mission_id, log)
            raise

