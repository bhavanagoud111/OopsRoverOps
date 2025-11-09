#!/bin/bash
# Rover Ops Project Setup Script

set -e

echo "🚀 Rover Ops Project Setup"
echo "=========================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if API keys are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo -e "${YELLOW}Usage: ./setup.sh <NASA_API_KEY> <OPENROUTER_API_KEY>${NC}"
    echo ""
    echo "Or set environment variables:"
    echo "  export NASA_API_KEY=your_nasa_key"
    echo "  export OPENROUTER_API_KEY=your_openrouter_key"
    echo "  ./setup.sh"
    echo ""
    exit 1
fi

NASA_KEY=${1:-$NASA_API_KEY}
OPENROUTER_KEY=${2:-$OPENROUTER_API_KEY}

if [ -z "$NASA_KEY" ] || [ -z "$OPENROUTER_KEY" ]; then
    echo -e "${RED}Error: Both NASA_API_KEY and OPENROUTER_API_KEY are required${NC}"
    exit 1
fi

# Backend Setup
echo -e "${GREEN}📦 Setting up backend...${NC}"
cd backend

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create/Update .env file
echo "Configuring backend .env file..."
cat > .env << EOF
# OpenRouter API Configuration
OPENROUTER_API_KEY=${OPENROUTER_KEY}
OPENROUTER_MODEL=openrouter/polaris-alpha

# NASA API Configuration
NASA_API_KEY=${NASA_KEY}

# Backend Server Configuration
BACKEND_PORT=8000
EOF

echo -e "${GREEN}✅ Backend configured${NC}"
deactivate
cd ..

# Frontend Setup
echo -e "${GREEN}📦 Setting up frontend...${NC}"
cd frontend

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
else
    echo "Node modules already installed"
fi

# Create .env file
echo "Configuring frontend .env file..."
cat > .env << EOF
# Frontend API Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
EOF

echo -e "${GREEN}✅ Frontend configured${NC}"
cd ..

echo ""
echo -e "${GREEN}✨ Setup complete!${NC}"
echo ""
echo "To start the project:"
echo "  1. Backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo "  2. Frontend: cd frontend && npm run dev"
echo ""

