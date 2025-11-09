# 🚀 Start Rover Ops Project

## ✅ Configuration Complete

All API keys are configured and dependencies are installed.

## Start the Application

### Terminal 1: Backend Server
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Backend will be available at: **http://localhost:8000**

### Terminal 2: Frontend Server
```bash
cd frontend
npm run dev
```

Frontend will be available at: **http://localhost:5174**

## Verify Setup

1. **Backend Health Check**: Visit http://localhost:8000/health
2. **Frontend**: Open http://localhost:5174 in your browser
3. **Test Mission**: Enter a goal like "Move to (5,9) and return"

## Configuration Summary

✅ **Backend .env**:
- OPENROUTER_API_KEY: Configured
- NASA_API_KEY: Configured
- BACKEND_PORT: 8000

✅ **Frontend .env**:
- VITE_API_URL: http://localhost:8000
- VITE_WS_URL: ws://localhost:8000

✅ **Dependencies**: All installed

## Troubleshooting

**Backend won't start:**
- Check port 8000 is available: `lsof -i :8000`
- Verify venv is activated: `which python` should show venv path

**Frontend won't start:**
- Check port 5174 is available
- Try: `rm -rf node_modules && npm install`

**API connection errors:**
- Ensure backend is running first
- Check CORS settings in `backend/app/main.py`

