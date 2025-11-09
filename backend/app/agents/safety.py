from typing import Dict, Any
from app.agents.base import BaseAgent
from app.models.schemas import AgentType, AgentStatus, RoverPosition

class SafetyAgent(BaseAgent):
    """Agent that validates rover moves and blocks unsafe actions"""
    
    def __init__(self):
        system_prompt = """You are a safety validation agent for a Mars rover. Your job is to validate every rover movement and action to ensure safety.

You must check:
1. Position validity: Is the position within the 10x10 grid bounds (0-9)?
2. Obstacle detection: Will the rover hit an obstacle?
3. Path safety: Is the path clear and safe?
4. Terrain hazards: Are there any terrain issues?
5. Mission constraints: Does this action violate any safety protocols?

Rules:
- Rover cannot move outside the grid (0-9 for both x and y)
- Rover cannot move to positions with obstacles
- Rover should avoid dangerous terrain
- Abrupt direction changes might indicate issues

Respond with a JSON object:
{{
  "approved": true/false,
  "reason": "explanation of approval or rejection",
  "alternative_position": {{"x": number, "y": number}} or null if approved,
  "risk_level": "low"/"medium"/"high"
}}"""
        
        super().__init__(AgentType.SAFETY, system_prompt, temperature=0.2)
    
    async def validate_move(
        self,
        current_position: RoverPosition,
        proposed_position: RoverPosition,
        obstacles: list,
        weather_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Validate a proposed rover move"""
        self.set_status(AgentStatus.VALIDATING)
        
        obstacles_str = ", ".join([f"({o.x}, {o.y})" for o in obstacles]) if obstacles else "None"
        weather_str = f"Weather conditions: {weather_data}" if weather_data else "No weather data available"
        
        input_text = f"""Validate this rover movement:
Current position: ({current_position.x}, {current_position.y})
Proposed position: ({proposed_position.x}, {proposed_position.y})
Known obstacles: {obstacles_str}
{weather_str}

Check if this move is safe and valid."""
        
        result = await self.process(input_text)
        
        # First do basic validation
        basic_validation = self._basic_validation(proposed_position, obstacles)
        
        if not basic_validation["approved"]:
            return basic_validation
        
        # If basic validation passes, check LLM response
        if result["status"] == "error":
            return basic_validation
        
        try:
            response_text = result["response"]
            
            # Extract JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                import json
                safety_data = json.loads(json_text)
            else:
                import json
                safety_data = json.loads(response_text)
            
            approved = safety_data.get("approved", True)
            reason = safety_data.get("reason", "Validated by safety agent")
            
            # Override with basic validation if LLM approves unsafe move
            if approved and not basic_validation["approved"]:
                approved = False
                reason = basic_validation["reason"]
            
            alt_pos = None
            if safety_data.get("alternative_position"):
                alt_data = safety_data["alternative_position"]
                alt_pos = RoverPosition(x=alt_data["x"], y=alt_data["y"])
            
            return {
                "approved": approved,
                "reason": reason,
                "alternative_position": alt_pos,
                "risk_level": safety_data.get("risk_level", "low"),
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error parsing safety response: {e}")
            # Return basic validation if LLM parsing fails
            return basic_validation
    
    def _basic_validation(
        self,
        proposed_position: RoverPosition,
        obstacles: list
    ) -> Dict[str, Any]:
        """Basic validation without LLM (faster, more reliable for hard constraints)"""
        # Check bounds
        if proposed_position.x < 0 or proposed_position.x > 9:
            return {
                "approved": False,
                "reason": f"Position out of bounds: x={proposed_position.x} (must be 0-9)",
                "alternative_position": None,
                "risk_level": "high",
                "status": "success"
            }
        
        if proposed_position.y < 0 or proposed_position.y > 9:
            return {
                "approved": False,
                "reason": f"Position out of bounds: y={proposed_position.y} (must be 0-9)",
                "alternative_position": None,
                "risk_level": "high",
                "status": "success"
            }
        
        # Check obstacles
        for obstacle in obstacles:
            if obstacle.x == proposed_position.x and obstacle.y == proposed_position.y:
                return {
                    "approved": False,
                    "reason": f"Obstacle detected at position ({proposed_position.x}, {proposed_position.y})",
                    "alternative_position": None,
                    "risk_level": "high",
                    "status": "success"
                }
        
        # All checks passed
        return {
            "approved": True,
            "reason": "Move validated: position is safe and within bounds",
            "alternative_position": None,
            "risk_level": "low",
            "status": "success"
        }

