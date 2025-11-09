# Debug Guide - Coordinate Override Issue

## Problem
Rover always goes to (5,5) instead of goal coordinates (e.g., (5,9)).

## Quick Debug Steps

### 1. Check Coordinate Extraction
Add debug logging to `planner.py` line 74:
```python
goal_coords = self._extract_coordinates_from_goal(goal)
print(f"DEBUG: Extracted coordinates from goal '{goal}': {goal_coords}")
```

### 2. Check LLM Response
Add debug logging after line 86:
```python
response_text = result["response"]
print(f"DEBUG: LLM response: {response_text[:500]}")  # First 500 chars
```

### 3. Check Override Trigger
The override should print at line 119:
```python
print(f"Overriding LLM coordinates ({target_x}, {target_y}) with goal coordinates ({goal_coords['x']}, {goal_coords['y']}) for {action} step")
```

### 4. Check Final Step Coordinates
Add debug logging after line 138:
```python
mission_steps.append(step)
print(f"DEBUG: Step {step.step_number} - Action: {step.action}, Target: ({step.target_position.x if step.target_position else None}, {step.target_position.y if step.target_position else None})")
```

## Expected Output

For goal "Move to (5,9) and return":
```
DEBUG: Extracted coordinates from goal 'Move to (5,9) and return': {'x': 5, 'y': 9}
DEBUG: LLM response: {...}
Overriding LLM coordinates (5, 5) with goal coordinates (5, 9) for move step
DEBUG: Step 1 - Action: move, Target: (5, 9)
```

## If Override Not Triggering

Check:
1. Is `goal_coords` populated? (Should have x=5, y=9)
2. Is `first_move_overridden` False initially?
3. Is `action` in ["move", "explore"]?
4. Does step have `target_position`?

## Potential Issues

1. **Regex not matching**: Test `_extract_coordinates_from_goal()` with various formats
2. **LLM response format**: LLM might return different JSON structure
3. **Step order**: First step might not be move/explore
4. **Override condition**: Logic might not be triggering

## Test Coordinate Extraction

```python
# Test in Python shell
from app.agents.planner import PlannerAgent
planner = PlannerAgent()
coords = planner._extract_coordinates_from_goal("Move to (5,9) and return")
print(coords)  # Should print {'x': 5, 'y': 9}
```

## Alternative Fix

If override isn't working, force coordinates BEFORE calling LLM:

```python
# In plan_mission(), before calling LLM:
goal_coords = self._extract_coordinates_from_goal(goal)
if goal_coords["x"] is not None and goal_coords["y"] is not None:
    # Modify input to explicitly include coordinates
    input_text = f"Create a mission plan for: {goal}. The target coordinates are ({goal_coords['x']}, {goal_coords['y']}). Use these exact coordinates in your plan."
```

