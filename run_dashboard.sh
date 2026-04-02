#!/bin/bash
# Launches AgentScope dashboard locally with capstone-rag environment loaded
cd /Users/pegahzargarian/projects/MCP-1/workspace/AgenticScope

# Load AgentScope keys
set -a
source .env
# Load capstone-rag DB + model config (overwrites model with Haiku)
source /Users/pegahzargarian/projects/MCP-1/workspace/capstone-rag/.env
set +a

source .venv/bin/activate
python -m agentscope.dashboard.app
