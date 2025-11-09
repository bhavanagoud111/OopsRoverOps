# Rover Ops Project - Complete Context & Current State

## Project Overview

**Rover Ops** is an AI-orchestrated Mars rover mission control simulator that uses multiple AI agents (powered by GPT-4o via OpenRouter) to plan, execute, monitor, and report on simulated Mars rover missions using real NASA data.

### Key Features
- Natural language mission input (e.g., "Move to (5,9) and return")
- Multi-agent orchestration using LangGraph
- Real-time visualization on a 10x10 grid
- NASA API integration (Mars rover photos, weather data)
- WebSocket-based real-time updates
- PDF mission reports

## Architecture

### Tech Stack

**Backend:**
- Python 3.12
- FastAPI (REST API + WebSocket)
- LangGraph (multi-agent orchestration)
- LangChain + OpenAI (via OpenRouter)
- Pydantic (data validation)
- Uvicorn (ASGI server)

**Frontend:**
- React 19 + TypeScript
- Vite
- Tailwind CSS
- WebSocket client
- jsPDF (report generation)

### Project Structure

```
RoverOps-Project/
├── backend/
│   ├── app/
│   │   ├── agents/          # AI agents
│   │   │   ├── base.py      # Base agent class with LLM initialization
│   │   │   ├── planner.py   # Mission planning agent (CRITICAL - has coordinate extraction bug)
│   │   │   ├── rover.py     # Rover execution agent (pathfinding with obstacle avoidance)
│   │   │   ├── safety.py    # Safety validation agent
│   │   │   ├── reporter.py  # Mission reporting agent
│   │   │   ├── supervisor.py # LangGraph orchestrator
│   │   │   └── state.py     # LangGraph state definitions
│   │   ├── models/
│   │   │   └── schemas.py   # Pydantic models (MissionStep, MissionState, etc.)
│   │   ├── services/
│   │   │   ├── mission_state.py # Mission state manager (in-memory storage)
│   │   │   └── nasa_client.py   # NASA API client
│   │   └── main.py          # FastAPI application entry point
│   ├── .env                  # API keys (OpenRouter, NASA)
│   └── requirements.txt      # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── components/      # React components
    │   │   ├── Dashboard.tsx
    │   │   ├── MissionInput.tsx
    │   │   ├── MissionCanvas.tsx
    │   │   ├── MissionLogs.tsx
    │   │   └── MissionReport.tsx
    │   ├── services/
    │   │   ├── api.ts        # REST API client
    │   │   ├── websocket.ts  # WebSocket client
    │   │   └── pdfGenerator.ts
    │   └── types/
    │       └── mission.ts    # TypeScript types
    └── package.json
```

## How It Works

### Mission Flow

1. **User Input**: User enters mission goal in frontend (e.g., "Move to (5,9) and return")
2. **API Request**: Frontend sends POST to `/api/mission/start` with goal
3. **Mission Creation**: Backend creates mission with unique ID
4. **Planning Phase**:
   - `PlannerAgent` receives goal
   - Calls GPT-4o via OpenRouter to break down goal into steps
   - **CRITICAL BUG**: Should extract coordinates from goal but LLM returns (5,5) instead
   - Returns list of `MissionStep` objects
5. **Execution Phase** (LangGraph workflow):
   - `fetch_nasa_data`: Gets Mars weather
   - `rover`: Executes steps, determines next position (with obstacle avoidance)
   - `safety`: Validates each move
   - `update_position`: Updates rover position, fetches NASA images
   - Loop until all steps complete
   - `reporter`: Generates final report
6. **Real-time Updates**: WebSocket broadcasts progress to frontend
7. **Visualization**: Frontend displays rover movement on 10x10 grid

### Agent System

#### 1. PlannerAgent (`app/agents/planner.py`)
- **Purpose**: Converts natural language goals to structured mission steps
- **Input**: Goal string (e.g., "Move to (5,9) and return")
- **Output**: List of `MissionStep` objects
- **Current Issue**: 
  - Has coordinate extraction logic but override isn't working
  - LLM returns (5,5) instead of extracting (5,9) from goal
  - Override logic exists but may not be triggering correctly

#### 2. RoverAgent (`app/agents/rover.py`)
- **Purpose**: Executes mission steps, determines next position
- **Features**: 
  - Obstacle-aware pathfinding
  - LLM reasoning for best path
  - Fallback pathfinding if LLM fails
- **Status**: Working correctly

#### 3. SafetyAgent (`app/agents/safety.py`)
- **Purpose**: Validates rover moves
- **Checks**: Bounds, obstacles, terrain safety
- **Status**: Working correctly

#### 4. ReporterAgent (`app/agents/reporter.py`)
- **Purpose**: Generates mission reports
- **Status**: Working correctly

#### 5. MissionSupervisor (`app/agents/supervisor.py`)
- **Purpose**: Orchestrates all agents using LangGraph
- **Workflow**: planner → fetch_nasa_data → rover → safety → update_position → (loop) → reporter
- **Status**: Working correctly

## Current Problem

### Issue: Rover Always Goes to (5,5) Instead of Goal Coordinates

**Symptoms:**
- User enters "Move to (5,9) and return"
- Rover goes to (5,5) instead of (5,9)
- Same path every time regardless of goal

**Root Cause:**
The `PlannerAgent` has coordinate extraction and override logic, but it's not working correctly. The LLM (GPT-4o) is returning (5,5) as a default, and the override isn't being applied properly.

**Attempted Fixes:**
1. ✅ Added coordinate extraction regex patterns
2. ✅ Added override logic to replace LLM coordinates with goal coordinates
3. ✅ Added description update to reflect correct coordinates
4. ❌ Still not working - override may not be triggering

**Code Location:**
- `backend/app/agents/planner.py` lines 69-140
- Key methods:
  - `plan_mission()`: Main planning method
  - `_extract_coordinates_from_goal()`: Extracts (x,y) from goal text
  - `_create_fallback_plan()`: Fallback if LLM parsing fails

**Debug Steps Needed:**
1. Check if `_extract_coordinates_from_goal()` is finding coordinates correctly
2. Verify override logic is being triggered (check for print statements)
3. Check if LLM response parsing is working
4. Verify `target_position` is being set correctly in `MissionStep`

## API Endpoints

### REST Endpoints
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/mission/start` - Start new mission
  - Request: `{"goal": "Move to (5,9) and return"}`
  - Response: `{"mission_id": "...", "status": "started", "message": "..."}`
- `GET /api/mission/{mission_id}` - Get mission status

### WebSocket
- `WS /ws/mission/{mission_id}` - Real-time mission updates
  - Message types: `status`, `update`, `complete`, `error`, `log`, `position`

## Environment Variables

**Backend `.env` file:**
```
OPENROUTER_API_KEY=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
OPENROUTER_MODEL=openai/gpt-4o
NASA_API_KEY=oYAHwEYGqv2gIGEdNZwe4ax1neNTA2FdTd3O2z92
BACKEND_PORT=8000
```

## Running the Project

### Backend
```bash
cd backend
source venv/bin/activate
PYTHONPATH=/path/to/backend python -m app.main
# Server runs on http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5174
```

## Key Data Models

### MissionStep
```python
class MissionStep(BaseModel):
    step_number: int
    action: str  # "move", "explore", "return", "scan", "collect"
    target_position: Optional[RoverPosition]  # (x, y) coordinates
    description: str
    completed: bool
    nasa_image_url: Optional[str]
```

### MissionState
```python
class MissionState(BaseModel):
    mission_id: str
    goal: str
    status: MissionStatus  # PENDING, PLANNING, EXECUTING, COMPLETE, ERROR
    current_step: int
    rover_position: RoverPosition  # Current position (x, y)
    obstacles: List[RoverPosition]  # Obstacle positions
    steps: List[MissionStep]  # Mission steps
    logs: List[MissionLog]  # Mission logs
    agent_states: Dict[AgentType, AgentStatus]  # Agent statuses
```

## Current State Summary

### ✅ Working
- Backend server running on port 8000
- Frontend running on port 5174
- API endpoints functional
- WebSocket connections working
- Agent orchestration (LangGraph) working
- Obstacle avoidance in rover agent
- NASA API integration
- Mission state management

### ❌ Not Working
- **Coordinate extraction/override in PlannerAgent**: Rover always goes to (5,5) instead of goal coordinates
- Override logic exists but may not be triggering
- Need to debug why coordinate override isn't working

### 🔧 Needs Investigation
1. Why is `_extract_coordinates_from_goal()` not working?
2. Is the override logic being triggered? (Check for print statements)
3. Is the LLM response being parsed correctly?
4. Are coordinates being set correctly in MissionStep?

## Next Steps for Fix

1. **Add debug logging** to `planner.py`:
   - Log extracted coordinates
   - Log LLM response
   - Log override trigger
   - Log final step coordinates

2. **Test coordinate extraction**:
   - Test `_extract_coordinates_from_goal()` with various formats
   - Verify regex patterns work

3. **Verify override logic**:
   - Check if `goal_coords` is populated
   - Check if `first_move_overridden` flag is working
   - Verify `target_position` is being set correctly

4. **Alternative approach**:
   - If override isn't working, consider forcing coordinate extraction BEFORE calling LLM
   - Or modify LLM prompt to be more explicit about using goal coordinates

## Important Files to Review

1. **`backend/app/agents/planner.py`** - Main issue location
   - Lines 69-140: `plan_mission()` method
   - Lines 137-162: `_extract_coordinates_from_goal()` method
   - Lines 164-200: `_create_fallback_plan()` method

2. **`backend/app/agents/supervisor.py`** - Orchestration
   - Lines 84-128: `_planner_node()` - Calls planner
   - Lines 161-217: `_rover_node()` - Executes steps

3. **`backend/app/services/mission_state.py`** - State management
   - Stores mission state in memory
   - Updates rover position, steps, logs

4. **`frontend/src/components/MissionInput.tsx`** - User input
5. **`frontend/src/services/api.ts`** - API calls
6. **`frontend/src/services/websocket.ts`** - WebSocket connection

## Testing

To test the fix:
1. Start backend: `cd backend && source venv/bin/activate && PYTHONPATH=. python -m app.main`
2. Start frontend: `cd frontend && npm run dev`
3. Open browser: `http://localhost:5174`
4. Enter goal: "Move to (5,9) and return"
5. Check logs for:
   - "Overriding LLM coordinates..." message
   - Target should be (5,9) not (5,5)
   - Rover should move to (5,9)

## Notes

- The project uses GPT-4o via OpenRouter for AI reasoning
- All agents have fallback logic if LLM fails
- Mission state is stored in memory (not persistent)
- Obstacles are randomly generated for each mission
- Grid is 10x10 (coordinates 0-9 for both x and y)
- Rover starts at (0,0) and should return to (0,0) at end

## Contact Points

- Backend entry: `backend/app/main.py`
- Frontend entry: `frontend/src/main.tsx`
- Key agent: `backend/app/agents/planner.py` (needs fix)
- State management: `backend/app/services/mission_state.py`

---

**Last Updated**: Current session
**Status**: Coordinate override not working - needs debugging

