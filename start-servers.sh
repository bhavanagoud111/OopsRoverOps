#!/bin/bash
# Start both backend and frontend servers

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Rover Ops Servers${NC}"
echo ""

# Kill existing servers
echo "Clearing existing servers..."
lsof -ti:8000 | xargs kill 2>/dev/null || true
lsof -ti:5174 | xargs kill 2>/dev/null || true
sleep 1

# Start Backend
echo -e "${GREEN}📦 Starting Backend...${NC}"
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
cd ..
echo "Backend PID: $BACKEND_PID"
echo "Backend logs: /tmp/backend.log"

# Start Frontend
echo -e "${GREEN}📦 Starting Frontend...${NC}"
cd frontend
npm run dev > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend PID: $FRONTEND_PID"
echo "Frontend logs: /tmp/frontend.log"

# Wait for servers to start
echo ""
echo "Waiting for servers to start..."
sleep 5

# Check status
echo ""
echo -e "${GREEN}✅ Server Status:${NC}"
if lsof -ti:8000 > /dev/null; then
    echo "✅ Backend: http://localhost:8000"
else
    echo "❌ Backend: Not running"
fi

if lsof -ti:5174 > /dev/null; then
    echo "✅ Frontend: http://localhost:5174"
else
    echo "❌ Frontend: Not running"
    echo "Check logs: tail -f /tmp/frontend.log"
fi

echo ""
echo "To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo "  or: lsof -ti:8000 | xargs kill && lsof -ti:5174 | xargs kill"
echo ""
echo "To view logs:"
echo "  Backend: tail -f /tmp/backend.log"
echo "  Frontend: tail -f /tmp/frontend.log"

