# Rover Ops - AI-Orchestrated Mission Control Simulator

A web-based mission control simulator that uses AI agents to plan, execute, monitor, and report on simulated Mars rover missions using real NASA data.

## Features

- **Natural Language Mission Input**: Enter mission goals in plain English
- **Multi-Agent Orchestration**: LangGraph-based coordination of Planner, Rover, Safety, and Reporter agents
- **Real-Time Visualization**: Live canvas showing rover movement on a 10x10 grid
- **NASA API Integration**: Real Mars rover photos and weather data
- **Live Mission Logs**: Real-time streaming of agent actions and decisions
- **Automated PDF Reports**: Generate mission reports with images and logs

## Tech Stack

### Backend
- FastAPI (Python)
- LangGraph for multi-agent orchestration
- LangChain + OpenAI for AI agents
- WebSockets for real-time updates
- NASA APIs (Mars Rover Photos, InSight Weather)

### Frontend
- React + TypeScript
- Vite
- Tailwind CSS
- jsPDF for report generation
- WebSocket client for real-time updates

## Setup

### Backend

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Add your API keys to `.env`:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/polaris-alpha
NASA_API_KEY=your_nasa_api_key_here
BACKEND_PORT=8000
```

6. Run the server:
```bash
python -m uvicorn app.main:app --reload
```

### Frontend

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create `.env` file (optional):
```
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

4. Run the development server:
```bash
npm run dev
```

## Usage

1. Start the backend server (port 8000)
2. Start the frontend development server (usually port 5173)
3. Open the app in your browser
4. Enter a mission goal (e.g., "Explore sector A and return to base")
5. Watch the agents plan, execute, and report on the mission in real-time

## Project Structure

```
Tigerhacks/
├── backend/
│   ├── app/
│   │   ├── agents/          # AI agents (Planner, Rover, Safety, Reporter)
│   │   ├── models/          # Pydantic models
│   │   ├── services/        # NASA client, mission state manager
│   │   └── main.py         # FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API and WebSocket services
│   │   └── types/          # TypeScript types
│   └── package.json
└── README.md
```

## API Keys

- **OpenRouter API Key**: Required for AI agents. Get one at https://openrouter.ai/ (free tier available)
- **NASA API Key**: Optional but recommended. Get one at https://api.nasa.gov/ (free)

## Quick Start

See `SETUP_INSTRUCTIONS.md` for detailed setup instructions.

## License

MIT

