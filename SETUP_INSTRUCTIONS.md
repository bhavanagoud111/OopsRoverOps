# Rover Ops - Setup Instructions

## Prerequisites

- Python 3.11+ (tested with Python 3.13)
- Node.js 18+ and npm
- OpenRouter API key (free tier available)
- NASA API key (free at https://api.nasa.gov/)

## Quick Start

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `backend` directory:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/polaris-alpha
NASA_API_KEY=your_nasa_api_key_here
BACKEND_PORT=8000
```

**Get API Keys:**
- OpenRouter: https://openrouter.ai/ (Sign up for free)
- NASA: https://api.nasa.gov/ (Generate API key for free)

### 3. Start Backend Server

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The backend will run on http://localhost:8000

### 4. Frontend Setup

```bash
cd frontend
npm install
```

### 5. Start Frontend Dev Server

```bash
cd frontend
npm run dev
```

The frontend will run on http://localhost:5173

## Usage

1. Open http://localhost:5173 in your browser
2. Enter a mission goal (e.g., "Explore area (3, 4) and collect samples")
3. Click "Start Mission"
4. Watch the AI agents plan and execute the mission in real-time!

## Project Structure

```
RoverOps/
├── backend/
│   ├── app/
│   │   ├── agents/          # AI agents (Planner, Rover, Safety, Reporter)
│   │   ├── models/          # Data models and schemas
│   │   ├── services/        # NASA client, mission state manager
│   │   └── main.py          # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables (create this)
│
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # API and WebSocket services
│   │   └── types/           # TypeScript types
│   ├── package.json         # Node.js dependencies
│   └── vite.config.ts       # Vite configuration
│
└── README.md
```

## Features

- **AI-Powered Mission Planning**: Natural language to structured mission steps
- **Multi-Agent System**: Planner, Rover, Safety, and Reporter agents
- **Real-Time Updates**: WebSocket-based live mission tracking
- **NASA Integration**: Real Mars rover photos and weather data
- **Visual Simulation**: 10x10 grid with rover movement visualization
- **Mission Reports**: PDF export of completed missions

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Change port in .env or use:
uvicorn app.main:app --reload --port 8001
```

**Missing dependencies:**
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

**LLM not working:**
- Check your `OPENROUTER_API_KEY` in `.env`
- Verify the model name (`openrouter/polaris-alpha` is free)
- Check internet connection

### Frontend Issues

**Port already in use:**
```bash
# Edit vite.config.ts or use:
npm run dev -- --port 5174
```

**Build errors:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**API connection errors:**
- Ensure backend is running on port 8000
- Check CORS settings in `backend/app/main.py`
- Verify WebSocket URL in `frontend/src/services/websocket.ts`

## Testing

### Test Agents

```bash
cd backend
source venv/bin/activate
python test_agents_functionality.py
```

This will verify that all AI agents are working correctly.

## Deployment

### Backend Deployment (e.g., Railway, Render, Fly.io)

1. Set environment variables in your hosting platform
2. Point to `backend/app/main:app` as the application
3. Ensure port is configurable (most platforms use `PORT` env var)

### Frontend Deployment (e.g., Vercel, Netlify)

1. Build the frontend: `cd frontend && npm run build`
2. Deploy the `dist` folder
3. Update API URL in production environment variables

## License

This project is provided as-is for educational and demonstration purposes.

## Support

For issues or questions, check the `AGENT_FUNCTIONALITY.md` file in the backend directory for detailed agent documentation.

