from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
from typing import Dict, Set
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from pydantic import BaseModel

from app.models.schemas import StartMissionRequest, StartMissionResponse, MissionStatusResponse
from app.services.mission_state import mission_state_manager
from app.models.schemas import MissionStatus, WebSocketMessage, AgentType
from app.phases import Phase
from app.route_queue import RouteQueue
from app.agents.supervisor import MissionSupervisor
from app.database.connection import get_db, init_db, check_db_connection
from app.database.repository import MissionRepository
from fastapi import Depends

class ScheduleMissionRequest(BaseModel):
    goal: str
    scheduled_time: str  # ISO format datetime string

load_dotenv()

app = FastAPI(title="Rover Ops API", version="1.0.0")

# CRITICAL: Initialize phase machine and route queue state slots
from app.route_queue import RouteQueue
from app.phases import Phase

# Initialize state slots (will be set per mission)
app.state.phase = Phase.PLAN
app.state.route_q = RouteQueue()
app.state.base = (0, 0)  # Base is always (0, 0)

# Initialize global supervisor instance with app reference
supervisor = MissionSupervisor(app_instance=app)

# Initialize NASA client photo pool and database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize NASA client photo pool and database on server startup"""
    # Initialize database
    try:
        if check_db_connection():
            print("✅ Database connection successful")
            init_db()
            print("✅ Database tables initialized")
        else:
            print("⚠️  Warning: Database connection failed. Continuing without database persistence.")
    except Exception as e:
        print(f"⚠️  Warning: Database initialization failed: {e}. Continuing without database persistence.")
    
    # Initialize NASA client photo pool
    from app.services.nasa_client import nasa_client
    if not nasa_client.cached_photos_pool:
        # Try to build from API first, fallback if it fails
        try:
            await nasa_client._build_photo_pool()
        except Exception as e:
            print(f"Error building photo pool from API: {e}, using fallback")
            nasa_client._build_fallback_pool()
        print(f"NASA photo pool initialized with {len(nasa_client.cached_photos_pool)} images")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, mission_id: str):
        await websocket.accept()
        if mission_id not in self.active_connections:
            self.active_connections[mission_id] = set()
        self.active_connections[mission_id].add(websocket)

    def disconnect(self, websocket: WebSocket, mission_id: str):
        if mission_id in self.active_connections:
            self.active_connections[mission_id].discard(websocket)
            if not self.active_connections[mission_id]:
                del self.active_connections[mission_id]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict, mission_id: str):
        if mission_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[mission_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error sending message: {e}")
                    disconnected.add(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn, mission_id)

manager = ConnectionManager()

async def execute_mission_async(mission_id: str, goal: str):
    """Execute mission in background and broadcast updates via WebSocket"""
    try:
        # Get mission state
        mission = mission_state_manager.get_mission(mission_id)
        if not mission:
            await manager.broadcast({
                "type": "error",
                "mission_id": mission_id,
                "message": "Mission not found"
            }, mission_id)
            return
        
        # Initialize state for LangGraph
        obstacles = mission.obstacles
        
        initial_state = {
            "goal": goal,
            "obstacles": obstacles
        }
        
        # Broadcast mission start
        await manager.broadcast({
            "type": "status",
            "mission_id": mission_id,
            "status": "executing",
            "message": "Mission execution started"
        }, mission_id)
        
        # Define broadcast callback
        async def broadcast_update(message: dict):
            await manager.broadcast(message, mission_id)
        
        # Execute mission using LangGraph with streaming updates
        final_state = await supervisor.execute_mission(mission_id, initial_state, broadcast_callback=broadcast_update)
        
        # Broadcast completion
        await manager.broadcast({
            "type": "complete",
            "mission_id": mission_id,
            "status": "complete",
            "message": "Mission completed",
            "data": {
                "status": final_state.get("status", "complete").value if hasattr(final_state.get("status"), "value") else str(final_state.get("status", "complete")),
                "steps_completed": final_state.get("current_step_index", 0),
                "total_steps": len(final_state.get("steps", []))
            }
        }, mission_id)
        
        # CRITICAL: Send explicit mission_status frame after Mission complete event
        await manager.broadcast({
            "type": "mission_status",
            "mission_id": mission_id,
            "data": {
                "status": "complete"
            }
        }, mission_id)
        
    except Exception as e:
        print(f"Error executing mission {mission_id}: {e}")
        import traceback
        traceback.print_exc()
        await manager.broadcast({
            "type": "error",
            "mission_id": mission_id,
            "message": f"Mission execution error: {str(e)}"
        }, mission_id)
        mission_state_manager.update_mission_status(mission_id, MissionStatus.ERROR)

@app.get("/")
async def root():
    return {"message": "Rover Ops API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/mission/start", response_model=StartMissionResponse)
async def start_mission(request: StartMissionRequest, background_tasks: BackgroundTasks):
    """Start a new mission with a given goal"""
    mission_id = mission_state_manager.create_mission(request.goal)

    # CRITICAL: Reset phase machine and route queue for new mission
    app.state.phase = Phase.EXECUTE_TASK
    app.state.route_q = RouteQueue()  # Empty for now; return route will be set later
    app.state.base = (0, 0)  # Base is always (0, 0)

    # Start mission execution in background
    background_tasks.add_task(execute_mission_async, mission_id, request.goal)

    return StartMissionResponse(
        mission_id=mission_id,
        status="started",
        message=f"Mission started with goal: {request.goal}"
    )

@app.post("/api/mission/schedule")
async def schedule_mission(request: ScheduleMissionRequest, background_tasks: BackgroundTasks):
    """Schedule a mission to run at a specific time"""
    try:
        scheduled_time = datetime.fromisoformat(request.scheduled_time)
        now = datetime.now()

        if scheduled_time <= now:
            raise HTTPException(status_code=400, detail="Scheduled time must be in the future")

        mission_id = mission_state_manager.create_mission(request.goal)
        delay_seconds = (scheduled_time - now).total_seconds()

        # Schedule mission execution
        async def delayed_execution():
            await asyncio.sleep(delay_seconds)
            await execute_mission_async(mission_id, request.goal)

        background_tasks.add_task(delayed_execution)

        return {
            "mission_id": mission_id,
            "status": "scheduled",
            "scheduled_time": request.scheduled_time,
            "message": f"Mission scheduled for {request.scheduled_time}",
            "delay_seconds": delay_seconds
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {str(e)}")

@app.get("/api/mission/{mission_id}", response_model=MissionStatusResponse)
async def get_mission_status(mission_id: str):
    """Get current mission status"""
    mission = mission_state_manager.get_mission(mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    return MissionStatusResponse(
        mission_id=mission_id,
        status=mission.status,
        state=mission
    )

@app.get("/api/apod")
async def get_apod():
    """Get Astronomy Picture of the Day for mission background"""
    from app.services.nasa_client import nasa_client
    try:
        apod_data = await nasa_client.get_apod()
        return apod_data
    except Exception as e:
        print(f"Error fetching APOD: {e}")
        return nasa_client._get_mock_apod()

@app.get("/api/mission/{mission_id}/report")
async def get_mission_report(mission_id: str, db = Depends(get_db)):
    """Get detailed mission report - try database first, fallback to in-memory"""
    # Try to get report from database first
    try:
        repo = MissionRepository(db)
        db_report = repo.get_report(mission_id)
        if db_report:
            print(f"✅ Report retrieved from database for mission {mission_id}")
            return {
                "mission_id": mission_id,
                "goal": db_report.mission.goal if db_report.mission else "Unknown",
                "status": db_report.mission.status if db_report.mission else "unknown",
                "mission_status": db_report.mission_status,
                "summary": db_report.summary,
                "outcome": db_report.outcome,
                "completed_steps": db_report.completed_steps,
                "total_steps": db_report.total_steps,
                "duration_seconds": db_report.duration_seconds,
                "rover_final_position": {
                    "x": db_report.rover_final_x or 0,
                    "y": db_report.rover_final_y or 0
                },
                "collected_data": db_report.collected_data or [],
                "mission_photos": db_report.mission_photos or [],
                "apod": db_report.apod_data or {},
                "step_details": db_report.step_details or [],
                "timestamp": db_report.created_at.isoformat() if db_report.created_at else datetime.now().isoformat()
            }
    except Exception as e:
        print(f"⚠️  Warning: Failed to get report from database: {e}. Falling back to in-memory.")
    
    # Fallback to in-memory mission state
    mission = mission_state_manager.get_mission(mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    # Get diverse photos from pool for report
    from app.services.nasa_client import nasa_client
    mission_photos = nasa_client.get_random_photos_from_pool(count=3)

    # Get APOD
    apod_data = {}
    try:
        apod_data = await nasa_client.get_apod()
    except:
        apod_data = nasa_client._get_mock_apod()

    # CRITICAL: Determine mission_status from mission status
    if mission.status == MissionStatus.ABORTED:
        mission_status_str = "aborted"
    elif mission.status == MissionStatus.COMPLETE:
        mission_status_str = "complete"
    else:
        # Default to complete for successful pipeline
        mission_status_str = "complete"
    
    return {
        "mission_id": mission_id,
        "goal": mission.goal,
        "status": mission.status.value,
        "mission_status": mission_status_str,  # CRITICAL: Add mission_status field for frontend
        "rover_final_position": {"x": mission.rover_position.x, "y": mission.rover_position.y},
        "steps_completed": sum(1 for step in mission.steps if step.completed),
        "total_steps": len(mission.steps),
        "collected_data": mission.collected_data if hasattr(mission, 'collected_data') else [],  # Include collected data
        "mission_photos": [
            {
                "id": p.get("id"),
                "url": p.get("img_src"),
                "img_src": p.get("img_src"),  # Include both for compatibility
                "camera": p.get("camera", {}).get("name") if isinstance(p.get("camera"), dict) else p.get("camera"),
                "sol": p.get("sol")
            }
            for p in mission_photos
        ],
        "astronomy_picture_of_the_day": {
            "title": apod_data.get("title", "Astronomy Picture of the Day"),
            "date": apod_data.get("date", ""),
            "explanation": apod_data.get("explanation", ""),
            "image_url": apod_data.get("url", apod_data.get("hdurl", "")),
            "copyright": apod_data.get("copyright", "NASA")
        },
        "logs": [
            {
                "timestamp": log.timestamp.isoformat(),
                "agent": log.agent_type.value,
                "message": log.message,
                "level": log.level
            }
            for log in mission.logs
        ]
    }

@app.post("/mission/abort")
async def abort_mission(request: Request):
    """Abort mission and broadcast aborted status to all connected clients"""
    # Get mission_id from request body if provided, otherwise broadcast to all
    try:
        body = await request.json()
        mission_id = body.get("mission_id")
    except:
        mission_id = None
    
    # CRITICAL: Update mission status in mission state manager if mission_id provided
    if mission_id:
        mission_state_manager.update_mission_status(mission_id, MissionStatus.ABORTED)
    
    # Tell all connected clients the mission was aborted
    payload = {
        "type": "mission_status",
        "data": {
            "status": "aborted"
        }
    }
    if mission_id:
        payload["mission_id"] = mission_id
    
    dead = []
    if mission_id:
        # Broadcast to specific mission
        if mission_id in manager.active_connections:
            for ws in manager.active_connections[mission_id]:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
    else:
        # Broadcast to all missions
        for mission_connections in manager.active_connections.values():
            for ws in mission_connections:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
    
    # Clean up dead connections
    for d in dead:
        for mission_id_key, connections in manager.active_connections.items():
            if d in connections:
                connections.discard(d)
                if not connections:
                    del manager.active_connections[mission_id_key]
    
    return {"ok": True}

@app.websocket("/ws/mission/{mission_id}")
async def websocket_endpoint(websocket: WebSocket, mission_id: str):
    await manager.connect(websocket, mission_id)
    try:
        # Send current mission state on connection
        mission = mission_state_manager.get_mission(mission_id)
        if mission:
            await manager.send_personal_message({
                "type": "status",
                "mission_id": mission_id,
                "status": mission.status.value,
                "data": {
                    "rover_position": {"x": mission.rover_position.x, "y": mission.rover_position.y},
                    "current_step": mission.current_step,
                    "total_steps": len(mission.steps),
                    "agent_states": {k.value: v.value for k, v in mission.agent_states.items()}
                }
            }, websocket)
        
        # Keep connection alive and listen for messages
        while True:
            data = await websocket.receive_text()
            # Handle client messages if needed
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal_message({"type": "pong"}, websocket)
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket, mission_id)

if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

