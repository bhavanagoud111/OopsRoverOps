from typing import Dict, Any
from datetime import datetime
from app.agents.state import MissionGraphState
from app.models.schemas import AgentType, MissionReport, MissionStatus
from app.services.nasa_client import nasa_client
import asyncio

class ReporterAgent:
    """Agent that collects mission data and formats reports with NASA data"""

    def __init__(self):
        self.agent_type = AgentType.REPORTER

    async def generate_report(self, state: MissionGraphState) -> Dict[str, Any]:
        """Generate comprehensive mission report from state with NASA data"""
        completed_steps = sum(1 for step in state.get("steps", []) if step.completed)
        total_steps = len(state.get("steps", []))
        rover_position = state.get("rover_position", {"x": 0, "y": 0})

        # Handle both RoverPosition object and dict
        if hasattr(rover_position, 'x'):
            rover_pos_dict = {"x": rover_position.x, "y": rover_position.y}
        elif isinstance(rover_position, dict):
            rover_pos_dict = rover_position
        else:
            rover_pos_dict = {"x": 0, "y": 0}

        # Calculate duration
        start_time = datetime.now()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() if start_time else None

        # Generate summary
        # CRITICAL FIX: Get mission status from mission state manager if available
        mission_id = state.get("mission_id")
        mission_status = state.get("status", MissionStatus.COMPLETE)
        
        # Check mission state manager for actual status
        if mission_id:
            from app.services.mission_state import mission_state_manager
            mission = mission_state_manager.get_mission(mission_id)
            if mission:
                mission_status = mission.status
        
        # CRITICAL FIX: Only check for ACTUAL obstacle blocking, not just mentions of "obstacle"
        # Only set to ABORTED if there's clear evidence of obstacle blocking the mission
        if mission_status != MissionStatus.ABORTED and mission_id:
            from app.services.mission_state import mission_state_manager
            mission = mission_state_manager.get_mission(mission_id)
            if mission and mission.logs:
                recent_logs = mission.logs[-15:]  # Check last 15 logs
                # CRITICAL: Only detect actual obstacle blocking, not just mentions
                # Look for specific patterns that indicate obstacle blocking:
                # - "Obstacle detected at position" (safety rejection)
                # - "aborted due to obstacle" (explicit abort)
                # - "blocked by obstacle" (explicit blocking)
                # - "unable to find alternative path" (truly stuck)
                obstacle_blocking = any(
                    ("obstacle detected at position" in log.message.lower() and log.level == "warning") or
                    ("aborted due to obstacle" in log.message.lower()) or
                    ("blocked by obstacle" in log.message.lower()) or
                    ("unable to find alternative path" in log.message.lower()) or
                    ("too many safety rejections" in log.message.lower())
                    for log in recent_logs
                )
                # Only set to ABORTED if there's clear evidence of obstacle blocking
                if obstacle_blocking:
                    mission_status = MissionStatus.ABORTED
                    print(f"⚠️  Reporter: Detected actual obstacle blocking in logs, setting status to ABORTED")

        # CRITICAL FIX: Get rover position from mission state manager (source of truth) if available
        actual_rover_pos = rover_pos_dict
        if mission_id:
            from app.services.mission_state import mission_state_manager
            mission = mission_state_manager.get_mission(mission_id)
            if mission and mission.rover_position:
                if hasattr(mission.rover_position, 'x'):
                    actual_rover_pos = {"x": mission.rover_position.x, "y": mission.rover_position.y}
                elif isinstance(mission.rover_position, dict):
                    actual_rover_pos = mission.rover_position
        
        at_base = (actual_rover_pos.get("x", 0) == 0 and actual_rover_pos.get("y", 0) == 0)
        
        # CRITICAL FIX: Generate summary and outcome based on actual agent logs and mission state
        # Check agent logs for actual mission execution details
        agent_summary_parts = []
        if mission_id:
            from app.services.mission_state import mission_state_manager
            mission = mission_state_manager.get_mission(mission_id)
            if mission and mission.logs:
                # Look for key agent messages to build accurate summary
                planner_logs = [log for log in mission.logs if log.agent_type == AgentType.PLANNER]
                rover_logs = [log for log in mission.logs if log.agent_type == AgentType.ROVER]
                safety_logs = [log for log in mission.logs if log.agent_type == AgentType.SAFETY]
                reporter_logs = [log for log in mission.logs if log.agent_type == AgentType.REPORTER]
                
                # Use actual agent messages to build summary
                if planner_logs:
                    planning_msg = planner_logs[-1].message if planner_logs else ""
                    if "Generated" in planning_msg or "steps" in planning_msg.lower():
                        agent_summary_parts.append(planning_msg)
                
                # Check for obstacle detection from safety agent
                obstacle_detected = any(
                    "obstacle" in log.message.lower() and log.level == "warning"
                    for log in safety_logs
                )
                
                # Check for completion from reporter
                if reporter_logs:
                    completion_msg = reporter_logs[-1].message if reporter_logs else ""
                    if "Mission report generated" in completion_msg:
                        agent_summary_parts.append(completion_msg)
        
        # Build summary from agent outputs or fallback to step count
        if agent_summary_parts:
            summary = f"Mission '{state.get('goal', 'Unknown')}': {'. '.join(agent_summary_parts[-2:])}"
        else:
            summary = f"Mission '{state.get('goal', 'Unknown')}' completed {completed_steps}/{total_steps} steps."
        
        # Generate outcome based on actual mission status and agent logs
        if mission_status == MissionStatus.ABORTED:
            # CRITICAL FIX: Check if rover actually returned to base before saying it did
            if at_base:
                # Rover is at base - use full message
                outcome = "Mission incomplete. Rover safely returned to base (0,0). Mission aborted due to obstacle detection. Mission incomplete."
            else:
                # Rover is NOT at base - don't say it returned
                outcome = f"Mission incomplete. Mission aborted due to obstacle detection. Rover at position ({actual_rover_pos.get('x', 0)}, {actual_rover_pos.get('y', 0)}). Mission incomplete."
        elif mission_status == MissionStatus.COMPLETE and state.get("execution_complete"):
            # CRITICAL FIX: Mission only complete if rover is at base (0,0)
            if at_base:
                outcome = "SUCCESS: Mission completed successfully"
            else:
                # Rover is not at base - mission is incomplete
                outcome = f"Mission incomplete. Mission execution finished but rover did not return to base. Rover at position ({actual_rover_pos.get('x', 0)}, {actual_rover_pos.get('y', 0)}). Mission incomplete."
        elif state.get("error"):
            outcome = f"ERROR: {state.get('error')}"
        else:
            # CRITICAL FIX: Check if rover is at base before saying mission is complete
            if at_base:
                outcome = "COMPLETED: Mission execution finished"
            else:
                outcome = f"Mission incomplete. Mission execution finished but rover did not return to base. Rover at position ({actual_rover_pos.get('x', 0)}, {actual_rover_pos.get('y', 0)}). Mission incomplete."

        # Fetch random mission photos for variety
        mission_photos = nasa_client.get_random_mission_photos(count=3)

        # Fetch APOD for report
        apod_data = {}
        try:
            apod_data = await nasa_client.get_apod()
            # Ensure apod_data is a dict
            if not isinstance(apod_data, dict):
                apod_data = nasa_client._get_mock_apod()
        except Exception as e:
            print(f"Error fetching APOD: {e}")
            apod_data = nasa_client._get_mock_apod()

        # Build step details with any associated images
        step_details = []
        for step in state.get("steps", []):
            step_info = {
                "step_number": step.step_number,
                "action": step.action,
                "description": step.description,
                "completed": step.completed,
                "target_position": {
                    "x": step.target_position.x,
                    "y": step.target_position.y
                } if step.target_position else None,
                "image_url": step.nasa_image_url if hasattr(step, 'nasa_image_url') else None
            }
            step_details.append(step_info)

        # Get collected data from mission state
        collected_data = state.get("collected_data", [])
        mission_id = state.get("mission_id")
        if mission_id:
            from app.services.mission_state import mission_state_manager
            mission = mission_state_manager.get_mission(mission_id)
            if mission and mission.collected_data:
                collected_data = mission.collected_data

        # Determine mission_status based on actual mission status
        if mission_status == MissionStatus.ABORTED:
            mission_status_str = "aborted"
        elif mission_status == MissionStatus.COMPLETE:
            mission_status_str = "complete"
        else:
            # Default to complete for successful pipeline
            mission_status_str = "complete"
        
        report_data = {
            "summary": summary,
            "outcome": outcome,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "duration_seconds": duration,
            "status": "success",
            "mission_status": mission_status_str,  # CRITICAL: Add mission_status field for frontend
            "rover_final_position": rover_pos_dict,
            "collected_data": collected_data,  # Include collected samples and findings
            "mission_photos": [
                {
                    "id": p.get("id"),
                    "url": p.get("img_src"),
                    "camera": p.get("camera", {}).get("name") if isinstance(p.get("camera"), dict) else p.get("camera"),
                    "sol": p.get("sol")
                }
                for p in mission_photos
            ],
            "apod": {
                "title": apod_data.get("title", "Astronomy Picture of the Day"),
                "date": apod_data.get("date", datetime.now().strftime("%Y-%m-%d")),
                "explanation": apod_data.get("explanation", ""),
                "image_url": apod_data.get("url", apod_data.get("hdurl", "")),
                "copyright": apod_data.get("copyright", "NASA")
            },
            "step_details": step_details,
            "timestamp": datetime.now().isoformat()
        }
        
        # CRITICAL: Save report to database
        if mission_id:
            try:
                from app.database.connection import get_db_context
                from app.database.repository import MissionRepository
                
                with get_db_context() as db:
                    repo = MissionRepository(db)
                    repo.save_report(mission_id, report_data)
                    print(f"✅ Report saved to database for mission {mission_id}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to save report to database: {e}")
                # Continue without database save (non-blocking)
        
        return report_data

