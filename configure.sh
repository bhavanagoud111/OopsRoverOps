#!/bin/bash
# Quick configuration script for API keys

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🔑 Rover Ops API Key Configuration${NC}"
echo ""

# Get NASA API key
if [ -z "$NASA_API_KEY" ]; then
    read -p "Enter your NASA API key: " NASA_KEY
else
    NASA_KEY=$NASA_API_KEY
    echo "Using NASA_API_KEY from environment"
fi

# Get OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    read -p "Enter your OpenRouter API key: " OPENROUTER_KEY
else
    OPENROUTER_KEY=$OPENROUTER_API_KEY
    echo "Using OPENROUTER_API_KEY from environment"
fi

# Update backend .env
echo ""
echo "Updating backend/.env..."
cd backend
cat > .env << EOF
# OpenRouter API Configuration
OPENROUTER_API_KEY=${OPENROUTER_KEY}
OPENROUTER_MODEL=openrouter/polaris-alpha

# NASA API Configuration
NASA_API_KEY=${NASA_KEY}

# Backend Server Configuration
BACKEND_PORT=8000
EOF
cd ..

echo -e "${GREEN}✅ Configuration complete!${NC}"
echo ""
echo "Backend .env updated with your API keys."
echo ""

