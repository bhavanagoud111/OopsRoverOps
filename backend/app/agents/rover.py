from typing import Dict, Any, Optional
import json
from app.agents.base import BaseAgent
from app.models.schemas import AgentType, AgentStatus, MissionStep, RoverPosition

class RoverAgent(BaseAgent):
    """Agent that executes mission steps and determines movement actions"""
    
    def __init__(self):
        system_prompt = """You are a Mars rover execution agent. Your job is to execute mission steps by determining the next position to move to and actions to perform, while avoiding obstacles.

The rover operates on a 10x10 grid. You need to:
1. Calculate the next position based on the current step's target
2. AVOID OBSTACLES - plan a path that goes around obstacles, not through them
3. Consider the current position and plan movement towards the target
4. Use intelligent pathfinding - find the shortest safe path to the target
5. Request NASA images for interesting locations
6. Execute actions like explore, scan, collect and GENERATE DATA about findings

CRITICAL PATHFINDING RULES:
- NEVER move to a position that has an obstacle
- If the direct path to target is blocked, find an alternative route around obstacles
- Move one step at a time towards the target, avoiding obstacles
- Prefer diagonal moves when they get you closer to target and are safe
- If multiple safe paths exist, choose the shortest one
- **CRITICAL**: If you are NOT at the target coordinates yet, you MUST move towards them. Do NOT stay at current position unless you are already at the target.

When executing a step:
- If the step is "move" or "explore" AND has a target position: Calculate dx = target_x - current_x, dy = target_y - current_y. Move in the direction that reduces distance to target. Example: Current (2,3), Target (5,6) → move to (3,4) or (2,4) or (3,3) - NOT (2,3)!
- If the step is "scan" or "collect" AND you are already at target: STAY at current position and generate findings
- If the step is "scan" or "collect" AND you are NOT at target: Move towards target first, then execute action
- If the step is "return", move towards (0, 0) avoiding obstacles

Current rover capabilities:
- Can move one grid position at a time (up, down, left, right, or diagonally)
- Can take photos at each position
- Can scan for obstacles and terrain features - GENERATES scientific data
- Can collect samples (rocks, soil, minerals) - GENERATES sample data

CRITICAL: When executing "collect" or "scan" actions, you MUST generate realistic findings:
- For "collect": Describe what was collected (e.g., "Rock sample: Basalt, 2.5kg, contains iron oxide")
- For "scan": Describe what was found (e.g., "Terrain analysis: Sandy soil, low radiation, temperature -65°C")
- For "explore": Describe observations (e.g., "Area explored: Flat terrain, no obstacles, interesting geological formations")

Respond with a JSON object containing:
- next_position: {{"x": number, "y": number}} - the next position to move to (MUST avoid obstacles). If not at target yet, MUST be different from current position.
- action: what action to perform at this position
- request_image: boolean - whether to request a NASA image for this location
- findings: string - detailed findings/data when executing collect, scan, or explore actions (e.g., "Collected rock sample: Basalt composition, 2.3kg weight, contains iron oxide and silica")
- reasoning: brief explanation of your decision, especially how you avoided obstacles and moved towards target"""
        
        super().__init__(AgentType.ROVER, system_prompt, temperature=0.5)
    
    async def execute_step(
        self, 
        step: MissionStep, 
        current_position: RoverPosition,
        obstacles: list,
        mission_goal: str = None
    ) -> Dict[str, Any]:
        """Execute a mission step and determine next action"""
        self.set_status(AgentStatus.EXECUTING)
        
        obstacles_str = ", ".join([f"({o.x}, {o.y})" for o in obstacles]) if obstacles else "None"
        
        # Calculate distance to target for context
        if step.target_position:
            dx = step.target_position.x - current_position.x
            dy = step.target_position.y - current_position.y
            distance = abs(dx) + abs(dy)
            distance_info = f"Distance to target: {distance} steps (dx={dx}, dy={dy})"
        else:
            distance_info = "No target position (action at current location)"
        
        # Build context about the mission goal for better findings generation
        mission_context = f"\nMission Goal: {mission_goal}" if mission_goal else ""
        
        # Build explicit target instruction
        target_instruction = ""
        if step.target_position:
            target_instruction = f"""
TARGET COORDINATES: ({step.target_position.x}, {step.target_position.y})
YOU MUST MOVE TOWARDS THESE EXACT COORDINATES: ({step.target_position.x}, {step.target_position.y})
DO NOT choose random coordinates. The target is ({step.target_position.x}, {step.target_position.y}).
"""
        
        input_text = f"""Execute this mission step:
Step {step.step_number}: {step.action}
Description: {step.description}
Target position: ({step.target_position.x if step.target_position else 'None'}, {step.target_position.y if step.target_position else 'None'})
Current rover position: ({current_position.x}, {current_position.y})
{distance_info}
Known obstacles: {obstacles_str}
{mission_context}
{target_instruction}

CRITICAL NAVIGATION RULES:
1. If the step action is "move" or "explore" AND there is a target position, you MUST move towards that EXACT target
2. The target coordinates are ({step.target_position.x if step.target_position else 'N/A'}, {step.target_position.y if step.target_position else 'N/A'}) - use these exact numbers
3. Calculate the next position that gets you CLOSER to the target coordinates
4. If you are NOT at the target yet, you MUST move (do not stay at current position)
5. Move one step at a time towards the target: calculate dx = target_x - current_x, dy = target_y - current_y, then move in the direction that reduces distance
6. AVOID all obstacles listed above
7. Stay within grid bounds (0-9 for both x and y)
8. For "collect" or "scan" actions: only stay at current position if you are ALREADY at the target

CALCULATION EXAMPLE:
- Current: (2, 3), Target: (5, 6)
- dx = 5 - 2 = 3, dy = 6 - 3 = 3
- Next position should be (3, 4) or (2, 4) or (3, 3) - moving towards (5, 6)
- DO NOT return (2, 3) - that's staying in place!
- DO NOT choose (1, 1) or any other random coordinates - the target is (5, 6)!

IMPORTANT: When executing "collect", "scan", or "explore" actions, generate findings that are RELEVANT to the mission goal. For example:
- If the mission is about collecting rock samples, describe what rock samples were found
- If the mission is about scanning for water, describe water-related findings
- If the mission is about exploring terrain, describe terrain features
- Make findings specific to what the mission is trying to accomplish"""
        
        result = await self.process(input_text)
        
        if result["status"] == "error":
            # Fallback: move towards target or execute simple action with obstacle avoidance
            return self._create_fallback_action(step, current_position, obstacles)
        
        try:
            response_text = result["response"]
            
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                action_data = json.loads(json_text)
            else:
                action_data = json.loads(response_text)
            
            # Parse next position
            next_pos_data = action_data.get("next_position", {})
            next_position = RoverPosition(
                x=next_pos_data.get("x", current_position.x),
                y=next_pos_data.get("y", current_position.y)
            )
            
            # CRITICAL FIX: Check if proposed position is an obstacle BEFORE validating movement
            obstacle_positions = {(o.x, o.y) for o in obstacles} if obstacles else set()
            if (next_position.x, next_position.y) in obstacle_positions:
                print(f"⚠️  CRITICAL: LLM proposed obstacle position ({next_position.x}, {next_position.y}). Using fallback pathfinding to avoid obstacle.")
                # Immediately use fallback to find alternative path
                fallback_result = self._create_fallback_action(step, current_position, obstacles)
                next_position = fallback_result["next_position"]
                print(f"✅ Fallback pathfinding selected: ({next_position.x}, {next_position.y})")
            
            # CRITICAL: Validate that we're moving towards target if not already there
            if step.target_position:
                at_target = (current_position.x == step.target_position.x and current_position.y == step.target_position.y)
                not_moving = (next_position.x == current_position.x and next_position.y == current_position.y)
                
                # CRITICAL FIX: Guard against infinite loop - if at target and position unchanged, return completed
                if at_target and not_moving:
                    print(f"✅ Rover at target ({current_position.x}, {current_position.y}) and position unchanged. Returning completed status.")
                    return {
                        "next_position": current_position,  # Stay at current position
                        "action": action_data.get("action", step.action),
                        "request_image": action_data.get("request_image", True),
                        "findings": action_data.get("findings", ""),
                        "reasoning": "Rover reached target position",
                        "status": "completed"  # CRITICAL: Return completed status to LangGraph
                    }
                
                # Calculate if we're moving in the right direction
                if not at_target:
                    current_dist = abs(step.target_position.x - current_position.x) + abs(step.target_position.y - current_position.y)
                    next_dist = abs(step.target_position.x - next_position.x) + abs(step.target_position.y - next_position.y)
                    moving_away = next_dist > current_dist
                    
                    # CRITICAL FIX: If not moving, or moving away from target, ALWAYS use fallback
                    if not_moving or moving_away or (next_position.x != step.target_position.x and next_position.y != step.target_position.y and next_dist >= current_dist):
                        print(f"⚠️  Warning: LLM returned position ({next_position.x}, {next_position.y}) when target is ({step.target_position.x}, {step.target_position.y}). Current: ({current_position.x}, {current_position.y}). Forcing movement towards target.")
                        # Use fallback logic to ensure movement towards target
                        fallback_result = self._create_fallback_action(step, current_position, obstacles)
                        # CRITICAL FIX: ALWAYS use fallback position if it's different from current
                        # If fallback also returns same position, it means all paths are blocked (shouldn't happen for return step)
                        if fallback_result["next_position"].x != current_position.x or fallback_result["next_position"].y != current_position.y:
                            next_position = fallback_result["next_position"]
                            print(f"✅ Using fallback position: ({next_position.x}, {next_position.y})")
                        else:
                            # Fallback also returned same position - this should not happen, but log it
                            print(f"❌ ERROR: Fallback also returned same position ({current_position.x}, {current_position.y}). This should not happen!")
                            # Force a move anyway - try to move towards target manually
                            dx = step.target_position.x - current_position.x
                            dy = step.target_position.y - current_position.y
                            # Try to move one step in the direction of target
                            if dx != 0:
                                next_x = current_position.x + (1 if dx > 0 else -1)
                                if 0 <= next_x <= 9:
                                    next_position = RoverPosition(x=next_x, y=current_position.y)
                                    print(f"🔄 Forced move: ({current_position.x}, {current_position.y}) -> ({next_position.x}, {next_position.y})")
                            elif dy != 0:
                                next_y = current_position.y + (1 if dy > 0 else -1)
                                if 0 <= next_y <= 9:
                                    next_position = RoverPosition(x=current_position.x, y=next_y)
                                    print(f"🔄 Forced move: ({current_position.x}, {current_position.y}) -> ({next_position.x}, {next_position.y})")
            
            return {
                "next_position": next_position,
                "action": action_data.get("action", step.action),
                "request_image": action_data.get("request_image", True),
                "findings": action_data.get("findings", ""),  # Extract findings from LLM response
                "reasoning": action_data.get("reasoning", ""),
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error parsing rover response: {e}")
            return self._create_fallback_action(step, current_position, obstacles)
    
    def _create_fallback_action(self, step: MissionStep, current_position: RoverPosition, obstacles: list = None) -> Dict[str, Any]:
        """Create fallback action if LLM parsing fails - with obstacle avoidance"""
        import json
        
        obstacles = obstacles or []
        obstacle_positions = {(o.x, o.y) for o in obstacles}
        
        # Simple logic: move towards target or execute at current position
        if step.target_position:
            # Check if we're already at target
            if current_position.x == step.target_position.x and current_position.y == step.target_position.y:
                # Already at target, stay here and mark for completion
                next_position = current_position
                request_image = True
            else:
                # CRITICAL: Move one step towards target, avoiding obstacles
                # Calculate direction to target
                dx = step.target_position.x - current_position.x
                dy = step.target_position.y - current_position.y
                
                # Determine next position - move towards target
                next_x = current_position.x
                next_y = current_position.y
                
                # CRITICAL FIX: Try all 8 directions, sorted by distance to target
                # This ensures we find the best alternative path when direct path is blocked
                all_directions = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),  # Orthogonal
                    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
                ]
                
                # Calculate distance scores for all valid moves
                direction_scores = []
                for dir_x, dir_y in all_directions:
                    new_x = current_position.x + dir_x
                    new_y = current_position.y + dir_y
                    if 0 <= new_x <= 9 and 0 <= new_y <= 9:
                        if (new_x, new_y) not in obstacle_positions:
                            # Calculate distance to target
                            new_dx = abs(step.target_position.x - new_x)
                            new_dy = abs(step.target_position.y - new_y)
                            new_dist = new_dx + new_dy
                            current_dist = abs(step.target_position.x - current_position.x) + abs(step.target_position.y - current_position.y)
                            
                            # Prefer moves that get closer to target (allow slight detour)
                            if new_dist <= current_dist + 2:  # Allow 2-step detour
                                direction_scores.append((new_dist, new_x, new_y))
                
                # Sort by distance (closest first)
                direction_scores.sort(key=lambda x: x[0])
                
                # Find first valid candidate
                next_position = None
                if direction_scores:
                    _, best_x, best_y = direction_scores[0]
                    next_position = RoverPosition(x=best_x, y=best_y)
                    print(f"🔄 Fallback pathfinding: ({current_position.x}, {current_position.y}) -> ({best_x}, {best_y}) to reach target ({step.target_position.x}, {step.target_position.y})")
                else:
                    # If no valid moves found (all blocked), try any safe move (even if it increases distance slightly)
                    # This allows the rover to escape from being completely surrounded
                    for dir_x, dir_y in all_directions:
                        new_x = current_position.x + dir_x
                        new_y = current_position.y + dir_y
                        if 0 <= new_x <= 9 and 0 <= new_y <= 9:
                            if (new_x, new_y) not in obstacle_positions:
                                next_position = RoverPosition(x=new_x, y=new_y)
                                print(f"🔄 Emergency pathfinding: ({current_position.x}, {current_position.y}) -> ({new_x}, {new_y}) (all optimal paths blocked)")
                                break
                
                # If still blocked, stay in place (shouldn't happen often)
                if next_position is None:
                    next_position = current_position
                    print(f"⚠️  WARNING: All paths blocked from ({current_position.x}, {current_position.y}) to target ({step.target_position.x}, {step.target_position.y})")
                
                # Request image only when we reach the target
                request_image = (next_position.x == step.target_position.x and next_position.y == step.target_position.y)
        else:
            # Stay at current position for scan/collect actions without target
            next_position = current_position
            request_image = True
        
        # For collect/scan/explore actions, findings should be generated by LLM
        # Fallback doesn't generate findings - let LLM handle it
        return {
            "next_position": next_position,
            "action": step.action,
            "request_image": request_image,
            "findings": "",  # Findings should come from LLM, not hardcoded
            "reasoning": f"Fallback pathfinding: moving towards ({step.target_position.x if step.target_position else 'N/A'},{step.target_position.y if step.target_position else 'N/A'}) from ({current_position.x},{current_position.y}), avoiding {len(obstacles)} obstacles",
            "status": "success"
        }

