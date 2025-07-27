#!/bin/bash

# CCGL Analytics System - Start MCP System
# Starts all MCP servers in the cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 CCGL Analytics MCP System Starter${NC}"
echo "========================================="

cd ccgl_mcp_servers

echo -e "${BLUE}🔧 Starting MCP orchestrator...${NC}"

python3 mcp_orchestrator.py --start --monitor &

ORCHESTRATOR_PID=$!

echo -e "${GREEN}✅ MCP system started with PID: $ORCHESTRATOR_PID${NC}"
echo -e "${BLUE}💡 Press Ctrl+C to stop all servers${NC}"

# Wait for interrupt
trap "kill $ORCHESTRATOR_PID" INT TERM

wait $ORCHESTRATOR_PID