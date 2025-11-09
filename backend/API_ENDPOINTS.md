# Rover Ops API - Endpoints Documentation

## Base URL
```
http://localhost:8000
```

## REST Endpoints

### 1. Root Endpoint
- **GET** `/`
- **Description**: API root endpoint
- **Response**:
```json
{
  "message": "Rover Ops API",
  "status": "running"
}
```

### 2. Health Check
- **GET** `/health`
- **Description**: Server health status
- **Response**:
```json
{
  "status": "healthy"
}
```

### 3. Start Mission
- **POST** `/api/mission/start`
- **Description**: Start a new mission with a goal
- **Request Body**:
```json
{
  "goal": "Explore area (3, 4) and collect samples"
}
```
- **Response**:
```json
{
  "mission_id": "uuid-string",
  "status": "started",
  "message": "Mission started with goal: ..."
}
```

### 4. Get Mission Status
- **GET** `/api/mission/{mission_id}`
- **Description**: Get current status of a mission
- **Path Parameters**:
  - `mission_id`: UUID of the mission
- **Response**:
```json
{
  "mission_id": "uuid-string",
  "status": "executing",
  "state": {
    "mission_id": "uuid-string",
    "goal": "...",
    "status": "executing",
    "current_step": 1,
    "rover_position": {"x": 0, "y": 0},
    "obstacles": [...],
    "steps": [...],
    "logs": [...],
    "agent_states": {...}
  }
}
```

## WebSocket Endpoints

### 5. Mission WebSocket
- **WebSocket** `/ws/mission/{mission_id}`
- **Description**: Real-time updates for mission execution
- **Path Parameters**:
  - `mission_id`: UUID of the mission
- **Message Types**:
  - `status`: Mission status update
  - `update`: Real-time mission progress
  - `complete`: Mission completion
  - `error`: Error notification
  - `log`: Agent log entry
  - `position`: Rover position update

- **Client Messages**:
  - `{"type": "ping"}`: Keep-alive ping
  - Server responds with `{"type": "pong"}`

## API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

Environment variables (`.env` file):
- `OPENROUTER_API_KEY`: OpenRouter API key for GPT-4o
- `OPENROUTER_MODEL`: Model name (default: `openai/gpt-4o`)
- `NASA_API_KEY`: NASA API key
- `BACKEND_PORT`: Server port (default: 8000)

## CORS

CORS is enabled for all origins (`*`) for development. Update in production.

