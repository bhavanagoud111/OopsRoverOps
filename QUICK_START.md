# Quick Start Guide

## ✅ Project Status: Ready for API Keys

The project is configured and ready. You just need to add your API keys.

## 🔑 Adding API Keys

### Option 1: Use the configure script (Recommended)
```bash
./configure.sh
```
Then enter your NASA and OpenRouter API keys when prompted.

### Option 2: Update backend/.env directly
Edit `backend/.env` and replace the placeholder values:
```bash
OPENROUTER_API_KEY=your_actual_openrouter_key_here
NASA_API_KEY=your_actual_nasa_key_here
```

### Option 3: Set environment variables and run configure
```bash
export NASA_API_KEY=your_nasa_key
export OPENROUTER_API_KEY=your_openrouter_key
./configure.sh
```

## 🚀 Starting the Project

### 1. Start Backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```
Backend will run on: http://localhost:8000

### 2. Start Frontend (in a new terminal)
```bash
cd frontend
npm run dev
```
Frontend will run on: http://localhost:5174

## 📋 What's Already Configured

✅ Backend dependencies installed
✅ Frontend dependencies installed  
✅ Backend .env file created (needs your API keys)
✅ Frontend .env file created
✅ All services configured

## 🔍 Verify Configuration

Check that your API keys are set:
```bash
cd backend
cat .env | grep API_KEY
```

## 📝 Notes

- **OpenRouter**: The project uses OpenRouter (not OpenRoute) for AI agents
- **NASA API**: Free API key available at https://api.nasa.gov/
- **OpenRouter**: Free tier available at https://openrouter.ai/

