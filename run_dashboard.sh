#!/bin/bash
# Launches AgentScope dashboard locally with capstone-rag environment loaded
cd /Users/pegahzargarian/projects/MCP-1/workspace/AgenticScope

# Load AgentScope keys
set -a
source .env
# Load capstone-rag DB + model config
source /Users/pegahzargarian/projects/MCP-1/workspace/capstone-rag/.env
set +a

# Give DeepEval more time per API call (default 88s is too short for Haiku under load)
export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=180

# Disable inter-judge variance by default (prevents OpenAI timeout)
export AGENTSCOPE_VARIANCE=0

export PYTHONPATH=/Users/pegahzargarian/projects/MCP-1/workspace/AgenticScope

source .venv/bin/activate
python -m agentscope.dashboard.app
