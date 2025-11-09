# RoverOps Development Rules & Philosophy

Based on extensive code analysis and the ultrathink principle of solving the *real* problem, not just the surface issue.

## Core Principles

### 1. **Simplicity Over Cleverness**
- The coordinate extraction logic works because it's straightforward
- Regex patterns are clear and testable
- Override logic is explicit, not implicit

### 2. **Fail Gracefully**
- Always have fallbacks for when LLM fails
- Fallback logic should be as smart as the primary path
- Use extracted data from user input as ground truth

### 3. **Make Implicit Things Explicit**
- Use clear variable names: `goal_coords`, `first_move_overridden`, not `g` or `override`
- Comment critical sections explaining the "why"
- Log important decisions (extraction success, override triggered, fallback used)

### 4. **Test Your Assumptions**
- Don't assume the LLM will return perfect JSON
- Don't assume coordinates will be in standard format
- Test edge cases: missing coords, malformed input, LLM errors

## Architectural Patterns

### Pattern 1: Two-Layer Defense for LLM Reliability
```python
# Layer 1: Extract from source (user input)
extracted_value = extract_from_source(user_input)

# Layer 2: Call LLM for reasoning
llm_response = call_llm(user_input)

# Layer 3: Use extracted value to validate/override LLM if needed
if extracted_value and llm_value != extracted_value:
    use extracted_value  # Ground truth from user
```

This prevents LLM hallucinations from breaking functionality.

### Pattern 2: Fallback with Same Intelligence
```python
def primary_path():
    try:
        # Complex LLM-based logic
        return parse_llm_response()
    except:
        # Fallback should use same extraction logic, not hardcoded defaults
        return fallback_using_extraction()
```

Don't create two separate code paths with different logic.

### Pattern 3: Extracted Data as Ground Truth
```python
# Good:
user_goal = "Move to (5,9) and return"
extracted_coords = extract_coords(user_goal)  # (5,9) - ground truth
llm_plan = generate_plan(user_goal)  # May return different coords
if llm_plan.coords != extracted_coords:
    llm_plan.coords = extracted_coords  # Use ground truth

# Bad:
llm_plan = generate_plan(user_goal)  # Hope it returns right coords
if llm_plan.coords == (5,5):  # Assume (5,5) means error
    llm_plan.coords = (0,0)  # This is arbitrary
```

## Code Quality Standards

### 1. Coordinate Handling
```python
# Good: Use Pydantic models
target = RoverPosition(x=5, y=9)
print(f"Target: {target.x}, {target.y}")

# Bad: Use raw dicts/tuples
target = {"x": 5, "y": 9}
coords = (5, 9)
```

### 2. Status Tracking
```python
# Good: Use enums
if agent_status == AgentStatus.EXECUTING:

# Bad: Use strings
if agent_status == "executing":  # Case-sensitive, not validated
```

### 3. Error Handling
```python
# Good: Specific exceptions, fallback to known-good state
try:
    steps = parse_json(response)
except json.JSONDecodeError:
    steps = create_fallback_plan()  # Same extraction logic
    log.warning("LLM returned invalid JSON, using fallback")

# Bad: Generic exceptions, silently fail
try:
    steps = parse_json(response)
except:
    steps = []  # Empty? This breaks everything
```

## Testing Requirements

### For Each Agent:
1. **Happy Path Test** - Standard input, LLM succeeds
2. **Fallback Test** - LLM fails or returns invalid JSON
3. **Edge Case Test** - Malformed input, boundary values
4. **Roundtrip Test** - Input → Processing → Output integrity

### For Coordinate Logic Specifically:
```python
# Test extraction
assert extract_coords("(5,9)") == {"x": 5, "y": 9}
assert extract_coords("x=5,y=9") == {"x": 5, "y": 9}
assert extract_coords("move 5 9") == {"x": None, "y": None}

# Test override
llm_coords = {"x": 3, "y": 3}
goal_coords = {"x": 5, "y": 9}
assert override(llm_coords, goal_coords) == {"x": 5, "y": 9}

# Test fallback
assert fallback("Move to (8,2)") == {"x": 8, "y": 2}
assert fallback("Explore area") == {"x": 5, "y": 5}  # Default
```

## Performance Considerations

### 1. LLM Calls
- Cache mission plans when same goal appears multiple times
- Don't re-plan if only rover position changes
- Use temperature=0.3 for deterministic planning (not randomized)

### 2. Coordinate Operations
- Use simple math for pathfinding, not LLM reasoning for every step
- Extract coordinates once, reuse throughout mission
- Validate coordinates once at planning stage, not at execution

## Documentation Standards

### For Each Function:
```python
def plan_mission(self, goal: str) -> List[MissionStep]:
    """Generate mission plan from natural language goal

    Args:
        goal: Natural language goal like "Move to (5,9) and return"

    Returns:
        List of mission steps with extracted coordinates

    Notes:
        - Extracts coordinates from goal text as ground truth
        - Overrides LLM coordinates with extracted values for first move step
        - Falls back to extracting coordinates if LLM fails
    """
```

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Assuming LLM is Always Right
```python
# Bad
llm_coords = parse_llm(response)
step.target = llm_coords  # What if LLM hallucinated?
```

### ❌ Pitfall 2: Hardcoded Fallback Values
```python
# Bad
if not step.target_position:
    step.target_position = (5, 5)  # Why 5,5? Why not 0,0?
```

### ❌ Pitfall 3: Silent Failures
```python
# Bad
try:
    steps = parse_llm(response)
except:
    return []  # Now everything downstream breaks

# Good
except:
    return create_fallback_plan()  # Explicit fallback path
```

### ❌ Pitfall 4: Multiple Code Paths for Same Logic
```python
# Bad - now you maintain two coordinate extraction methods
def plan_mission():
    coords = extract_coords(goal)  # Primary

def create_fallback_plan():
    coords = extract_coords_legacy(goal)  # Different implementation!

# Good - same extraction, used everywhere
def plan_mission():
    coords = self._extract_coordinates_from_goal(goal)

def create_fallback_plan():
    coords = self._extract_coordinates_from_goal(goal)  # Reuse
```

## Code Review Checklist

Before merging any changes:

- [ ] All agents have fallback paths with same logic quality as primary path
- [ ] Coordinate extraction is centralized (DRY principle)
- [ ] Ground truth is extracted from user input, not derived from LLM
- [ ] No hardcoded default values (explain why they exist)
- [ ] Error messages are specific (e.g., "LLM returned 500" not "Error")
- [ ] All mission steps are validated before execution
- [ ] Tests cover happy path + fallback + edge cases
- [ ] Logging enables debugging without code changes

## Resources

- **Coordinate Extraction:** `app/agents/planner.py:_extract_coordinates_from_goal()`
- **Override Logic:** `app/agents/planner.py:plan_mission()` lines 112-124
- **Fallback Pattern:** `app/agents/planner.py:_create_fallback_plan()`

## Questions for Design Decisions

Before implementing new features, ask:

1. **What's the ground truth?** (User input, database, LLM reasoning?)
2. **What if LLM fails?** (Is fallback possible?)
3. **Can this be simpler?** (Do we really need LLM for this?)
4. **How will we debug this?** (What logs help troubleshooting?)
5. **What are the edge cases?** (Test them before shipping)

---

**Last Updated:** November 8, 2025
**Philosophy:** "Simplicity is the ultimate sophistication" - Leonardo da Vinci
