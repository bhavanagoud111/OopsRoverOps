import json
from typing import List, Dict, Any
from app.agents.base import BaseAgent
from app.models.schemas import AgentType, AgentStatus, MissionStep, RoverPosition

class PlannerAgent(BaseAgent):
    """Agent that breaks down natural language missions into structured steps"""
    
    def __init__(self):
        system_prompt = """You are a mission planner for a Mars rover. Your job is to break down high-level mission goals into specific, actionable steps.

The rover operates on a 10x10 grid (coordinates 0-9 for both x and y). The rover starts at position (0, 0).

CRITICAL: Extract coordinates from the mission goal text. Look for patterns like:
- "(5,9)" or "(5, 9)" or "5,9" or "5, 9"
- "coordinates (x, y)" or "position (x, y)"
- "go to (x, y)" or "move to (x, y)"
- "x=5, y=9" or "x:5, y:9"

If coordinates are specified in the goal, you MUST use those exact coordinates in your plan.

IMPORTANT: Understand the FULL mission goal, not just navigation:
- If the goal mentions "collect rock sample", "get sample", "collect sample" - you MUST include a "collect" action step
- If the goal mentions "scan", "analyze", "examine" - you MUST include a "scan" action step
- If the goal mentions "explore", "investigate" - you MUST include an "explore" action step
- The mission goal describes WHAT to do, not just WHERE to go

Available actions:
- move: Move the rover to a specific position (x, y)
- explore: Explore a specific area/position to gather information
- return: Return to the starting position (0, 0)
- scan: Scan the current area for obstacles, terrain features, or scientific data
- collect: Collect samples (rocks, soil, etc.) at the current position

When planning:
1. FIRST: Extract any coordinates mentioned in the mission goal (e.g., if goal says "move to (5,9)", extract x=5, y=9)
2. SECOND: Identify ALL actions mentioned in the goal (collect, scan, explore, etc.)
3. Break down the mission into 3-8 sequential steps that accomplish ALL parts of the goal
4. Each step should have a clear action and target position if applicable
5. If the goal mentions collecting samples, you MUST include a "collect" step at the target location
6. If the goal mentions scanning/analyzing, you MUST include a "scan" step
7. Consider obstacles and path optimization - plan a safe route
8. Always include a final "return to base" step unless the mission explicitly says otherwise
9. Use the EXACT coordinates extracted from the goal, not default values

Respond with a JSON array of steps, each with:
- step_number: sequential number starting from 1
- action: one of the available actions
- target_position: {{"x": number, "y": number}} if applicable (null for actions like "scan" or "collect" at current position)
- description: brief description of what this step accomplishes

Example response format:
{{
  "steps": [
    {{
      "step_number": 1,
      "action": "move",
      "target_position": {{"x": 5, "y": 9}},
      "description": "Move to target coordinates (5, 9)"
    }},
    {{
      "step_number": 2,
      "action": "explore",
      "target_position": {{"x": 5, "y": 9}},
      "description": "Explore the area and scan for features"
    }},
    {{
      "step_number": 3,
      "action": "return",
      "target_position": {{"x": 0, "y": 0}},
      "description": "Return to base"
    }}
  ]
}}"""
        
        super().__init__(AgentType.PLANNER, system_prompt, temperature=0.3)
    
    async def plan_mission(self, goal: str) -> List[MissionStep]:
        """Generate mission plan from natural language goal"""
        self.set_status(AgentStatus.PLANNING)

        # First, extract coordinates from goal to validate LLM response
        goal_coords = self._extract_coordinates_from_goal(goal)

        input_text = f"Create a mission plan for: {goal}"

        result = await self.process(input_text)

        if result["status"] == "error":
            # Fallback to simple plan
            return self._create_fallback_plan(goal)

        try:
            # Parse LLM response - it might be JSON or text with JSON
            response_text = result["response"]

            # Try to extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                plan_data = json.loads(json_text)
            else:
                # Try to parse the whole response as JSON
                plan_data = json.loads(response_text)

            steps_data = plan_data.get("steps", [])

            mission_steps = []
            
            # CRITICAL FIX: If goal has coordinates, ALWAYS create a move step FIRST
            # Don't trust LLM - force the first step to be a move to extracted coordinates
            if goal_coords and (goal_coords["x"] is not None and goal_coords["y"] is not None):
                print(f"🎯 FORCING first step to target ({goal_coords['x']}, {goal_coords['y']}) from goal coordinates")
                
                # Filter out return steps from the beginning
                filtered_steps = []
                for step in steps_data:
                    action = step.get("action", "").lower()
                    if action != "return":
                        filtered_steps.append(step)
                    elif len(filtered_steps) > 0:
                        # Return step after other steps - keep it
                        filtered_steps.append(step)
                steps_data = filtered_steps
                
                # ALWAYS insert a move step at the beginning with extracted coordinates
                # This ensures the rover ALWAYS goes to the correct target first
                forced_move_step = {
                    "step_number": 1,
                    "action": "move",
                    "target_position": {"x": goal_coords["x"], "y": goal_coords["y"]},
                    "description": f"Move to target coordinates ({goal_coords['x']}, {goal_coords['y']}) from mission goal"
                }
                
                # Renumber all existing steps
                for i, step in enumerate(steps_data):
                    step["step_number"] = i + 2
                
                # Insert forced move step at the beginning
                steps_data.insert(0, forced_move_step)
                print(f"✅ INSERTED forced move step to ({goal_coords['x']}, {goal_coords['y']}) as step 1")

            for step_data in steps_data:
                target_pos = None
                if step_data.get("target_position"):
                    pos_data = step_data["target_position"]
                    target_x = pos_data["x"]
                    target_y = pos_data["y"]
                    
                    # For step 1, ALWAYS use extracted coordinates if available
                    if step_data.get("step_number") == 1 and goal_coords and (goal_coords["x"] is not None and goal_coords["y"] is not None):
                        action = step_data.get("action", "").lower()
                        if action in ["move", "explore"]:
                            # FORCE step 1 to use extracted coordinates
                            target_x = goal_coords["x"]
                            target_y = goal_coords["y"]
                            step_data["description"] = f"Move to target coordinates ({target_x}, {target_y}) from mission goal"
                            print(f"✅ FORCED step 1 target to ({target_x}, {target_y})")

                    target_pos = RoverPosition(x=target_x, y=target_y)

                step = MissionStep(
                    step_number=step_data["step_number"],
                    action=step_data["action"],
                    target_position=target_pos,
                    description=step_data.get("description", ""),
                    completed=False
                )
                mission_steps.append(step)

            # CRITICAL FIX: Always ensure return-to-base step is at the end
            # Check if last step is a return step
            if mission_steps and len(mission_steps) > 0:
                last_step = mission_steps[-1]
                if last_step.action != "return":
                    # Add return step if not present
                    return_step = MissionStep(
                        step_number=len(mission_steps) + 1,
                        action="return",
                        target_position=RoverPosition(x=0, y=0),
                        description="Return to base (0, 0)",
                        completed=False
                    )
                    mission_steps.append(return_step)
                    print(f"✅ Added return-to-base step as step {return_step.step_number}")
            elif len(mission_steps) == 0:
                # No steps at all - add return step
                return_step = MissionStep(
                    step_number=1,
                    action="return",
                    target_position=RoverPosition(x=0, y=0),
                    description="Return to base (0, 0)",
                    completed=False
                )
                mission_steps.append(return_step)

            return mission_steps
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing planner response: {e}")
            print(f"Response was: {result.get('response', '')}")
            return self._create_fallback_plan(goal)
    
    def _extract_coordinates_from_goal(self, goal: str) -> dict:
        """Extract coordinates from goal text using regex patterns"""
        import re
        
        coord_patterns = [
            r'\((\d+)\s*,\s*(\d+)\)',  # (5,9) or (5, 9)
            r'at\s+(\d+)\s*,\s*(\d+)',  # at 5,9 or at 5, 9
            r'at\s+(\d+)\s+(\d+)',      # at 5 9
            r'(\d+)\s*,\s*(\d+)',      # 5,9 or 5, 9
            r'x\s*[=:]\s*(\d+)\s*[,;]\s*y\s*[=:]\s*(\d+)',  # x=5, y=9
            r'x\s*:\s*(\d+)\s*[,;]\s*y\s*:\s*(\d+)',       # x:5, y:9
            r'coordinates?\s+(\d+)\s*,\s*(\d+)',  # coordinate 5,9 or coordinates 5,9
            r'position\s+(\d+)\s*,\s*(\d+)',     # position 5,9
        ]
        
        target_x, target_y = None, None
        
        for pattern in coord_patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                try:
                    target_x = int(match.group(1))
                    target_y = int(match.group(2))
                    # Validate coordinates are within bounds
                    if 0 <= target_x <= 9 and 0 <= target_y <= 9:
                        break
                except (ValueError, IndexError):
                    continue
        
        return {"x": target_x, "y": target_y}
    
    def _create_fallback_plan(self, goal: str) -> List[MissionStep]:
        """Create a simple fallback plan if LLM parsing fails - extracts coordinates from goal"""
        # Use the coordinate extraction method
        goal_coords = self._extract_coordinates_from_goal(goal)
        
        target_x = goal_coords["x"]
        target_y = goal_coords["y"]
        
        # If no coordinates found, default to (5, 5) but log warning
        if target_x is None or target_y is None:
            print(f"Warning: Could not extract coordinates from goal '{goal}', using default (5, 5)")
            target_x, target_y = 5, 5
        
        return [
            MissionStep(
                step_number=1,
                action="move",
                target_position=RoverPosition(x=target_x, y=target_y),
                description=f"Move to target coordinates ({target_x}, {target_y}) from goal: {goal}",
                completed=False
            ),
            MissionStep(
                step_number=2,
                action="explore",
                target_position=RoverPosition(x=target_x, y=target_y),
                description=f"Explore the target area at ({target_x}, {target_y})",
                completed=False
            ),
            MissionStep(
                step_number=3,
                action="return",
                target_position=RoverPosition(x=0, y=0),
                description="Return to base",
                completed=False
            )
        ]

