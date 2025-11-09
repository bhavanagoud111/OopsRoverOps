# Agent Functionality Documentation

## Overview

The Rover Ops system uses **AI-powered agents** that actually call LLMs via OpenRouter to make intelligent decisions. Each agent has a specific role and can fall back to hardcoded logic if the LLM fails.

## Agent Architecture

### 1. BaseAgent (`app/agents/base.py`)

**Purpose:** Base class for all AI agents that handles LLM initialization and communication.

**How it works:**
- Initializes connection to OpenRouter API using your API key
- Uses the model specified in `OPENROUTER_MODEL` env variable (default: `openrouter/polaris-alpha`)
- Creates prompt templates with system instructions
- Calls the LLM asynchronously via `ainvoke()`
- Returns structured response: `{"status": "success/error", "response": "...", "error": "..."}`

**Key Features:**
- Error handling: If LLM call fails, returns error status instead of crashing
- Async support: All LLM calls are async for performance
- Configurable: Model and temperature can be customized per agent

### 2. PlannerAgent (`app/agents/planner.py`)

**Purpose:** Converts natural language mission goals into structured mission steps.

**How it works:**
1. Receives a natural language goal (e.g., "Explore area (3, 4) and collect samples")
2. Calls LLM with a detailed system prompt explaining:
   - Grid constraints (10x10, coordinates 0-9)
   - Available actions (move, explore, scan, collect, return)
   - Output format (JSON with steps array)
3. Parses LLM response to extract:
   - Step numbers
   - Actions
   - Target positions (x, y coordinates)
   - Descriptions
4. **Fallback:** If LLM fails or parsing fails, generates a simple 3-step plan:
   - Move to (5, 5)
   - Explore
   - Return to base

**Example LLM Response:**
```json
{
  "steps": [
    {
      "step_number": 1,
      "action": "move",
      "target_position": {"x": 3, "y": 4},
      "description": "Move to exploration zone"
    },
    {
      "step_number": 2,
      "action": "scan",
      "target_position": null,
      "description": "Scan for obstacles"
    }
  ]
}
```

**Status:** ✅ **WORKING** - LLM generates intelligent, goal-specific plans

### 3. RoverAgent (`app/agents/rover.py`)

**Purpose:** Determines the next movement/action for each mission step.

**How it works:**
1. Receives:
   - Current mission step
   - Current rover position
   - Known obstacles
2. Calls LLM to determine:
   - Next position to move to
   - Action to perform
   - Whether to request NASA image
   - Reasoning for the decision
3. Parses LLM response for `next_position`, `action`, `request_image`, `reasoning`
4. **Fallback:** If LLM fails, uses simple pathfinding:
   - Moves one step towards target (Manhattan distance)
   - Prioritizes larger axis distance
   - Supports diagonal movement
   - Ensures bounds (0-9)

**Example LLM Response:**
```json
{
  "next_position": {"x": 1, "y": 1},
  "action": "move",
  "request_image": false,
  "reasoning": "Moving diagonally towards target (3, 4) to optimize path"
}
```

**Status:** ✅ **WORKING** - LLM generates intelligent movement decisions (fallback also works)

### 4. SafetyAgent (`app/agents/safety.py`)

**Purpose:** Validates rover moves for safety (bounds, obstacles, hazards).

**How it works:**
1. Receives:
   - Current position
   - Proposed position
   - Known obstacles
   - Weather data (optional)
2. **Two-tier validation:**
   - **Basic validation (FAST):** Checks bounds (0-9) and obstacles without LLM
   - **LLM validation (OPTIONAL):** Calls LLM for additional safety checks if basic passes
3. Returns: `{"approved": true/false, "reason": "...", "risk_level": "low/medium/high"}`
4. **Priority:** Basic validation always wins (if basic rejects, LLM is not called)

**Why two-tier?**
- Basic validation is **fast and reliable** for hard constraints
- LLM validation adds **intelligence** for edge cases and terrain hazards
- Ensures **safety first** - can't be overridden by LLM

**Status:** ✅ **WORKING** - Uses fast basic validation (LLM optional for advanced scenarios)

### 5. ReporterAgent (`app/agents/reporter.py`)

**Purpose:** Generates mission summary reports (does NOT use LLM).

**How it works:**
- Collects mission state data
- Formats summary with:
  - Completed steps count
  - Mission outcome
  - Duration (if available)
  - Status
- Returns structured report dictionary

**Status:** ✅ **WORKING** - Simple data formatting (no LLM needed)

## Agent Orchestration (LangGraph)

The `MissionSupervisor` (`app/agents/supervisor.py`) orchestrates all agents using LangGraph:

```
1. Planner Agent → Generates mission steps
2. Fetch NASA Data → Gets weather/images
3. Rover Agent → Determines next move
4. Safety Agent → Validates move
5. Update Position → Moves rover, fetches images
6. (Loop back to Rover Agent for next step)
7. Reporter Agent → Generates final report
```

**Key Features:**
- State management via `MissionGraphState`
- Conditional edges (continue/complete/abort)
- Recursion limit: 300 (handles long missions)
- Real-time updates via WebSocket

## Configuration

### Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-v1-...  # Your OpenRouter API key
OPENROUTER_MODEL=openrouter/polaris-alpha  # Model to use (free)
NASA_API_KEY=...  # NASA API key for images
```

### Available Models (Free)

- `openrouter/polaris-alpha` - **Recommended** (free, reliable)
- `meta-llama/llama-3.2-3b-instruct:free` - Alternative free option
- `google/gemini-2.0-flash-exp:free` - Google's free model (may have rate limits)

## Testing Agents

Run the test script to verify agent functionality:

```bash
cd backend
source venv/bin/activate
python test_agents_functionality.py
```

This will:
1. Test BaseAgent LLM calls
2. Test PlannerAgent mission planning
3. Test RoverAgent movement decisions
4. Test SafetyAgent validation
5. Show whether agents are using LLM or fallback logic

## Current Status

✅ **All agents are FUNCTIONAL and using LLM!**

- **PlannerAgent:** ✅ Generating intelligent, goal-specific mission plans
- **RoverAgent:** ✅ Making smart movement decisions (with fallback)
- **SafetyAgent:** ✅ Validating moves with fast basic checks
- **ReporterAgent:** ✅ Formatting mission reports

## Important Notes

1. **Fallback Logic:** All agents have fallback mechanisms to ensure the system works even if LLM fails
2. **Error Handling:** LLM errors are caught and logged, but don't crash the system
3. **Performance:** Basic validation in SafetyAgent is fast (no LLM call needed for simple checks)
4. **Cost:** Using free models (`openrouter/polaris-alpha`) means no cost per request
5. **Rate Limits:** Free models may have rate limits; the system handles this gracefully

## How to Verify Agents Are Working

1. **Check logs:** Look for "Using LLM-generated plan" vs "Using fallback plan"
2. **Mission quality:** LLM-generated plans are more intelligent and goal-specific
3. **Rover movement:** LLM decisions are more strategic than simple pathfinding
4. **Test script:** Run `test_agents_functionality.py` to see detailed output

## Troubleshooting

**Agents using fallback instead of LLM?**
1. Check `OPENROUTER_API_KEY` is set correctly
2. Check `OPENROUTER_MODEL` is a valid model name
3. Check network connectivity to OpenRouter API
4. Check logs for specific error messages

**LLM calls failing?**
1. Verify API key is valid
2. Check model name is correct (use `openrouter/polaris-alpha`)
3. Check rate limits (free models may have limits)
4. Check internet connectivity

**Agents working but not intelligent?**
- Ensure model is set to `openrouter/polaris-alpha` (not a fallback model)
- Check system prompts are correct (they define agent behavior)
- Verify LLM responses are being parsed correctly

