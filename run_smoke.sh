#!/bin/bash
cd /Users/pegahzargarian/projects/MCP-1/workspace/AgenticScope
set -a
source .env
set +a
python3 smoke_test.py > smoke_output.txt 2>&1
echo "exit_code:$?" >> smoke_output.txt
