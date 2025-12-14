#!/bin/bash
# Start MCP Server with SmolVLA Policy for AI Control

# Kill any existing MCP server
pkill -f mcp_server 2>/dev/null || true
sleep 2

# Activate conda environment
source /home/shrek/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# Configuration
export PYTHONPATH=/home/shrek/urdf-os/src
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so101_policy"
export SO101_PORT=/dev/ttyACM0
export SO101_ROBOT_ID=my_awesome_follower_arm
export SO101_CAMERA_SOURCES=/dev/video6,/dev/video2
export SO101_CAMERA_NAMES=image,image2
export SO101_POLICY_CAMERA_MAP='{"image":"front","image2":"wrist"}'

LOG_FILE=/tmp/mcp_server.log
PID_FILE=/tmp/mcp_server.pid

echo "Starting MCP Server with SmolVLA policy..."
echo "Configuration:"
echo "  Robot ID: $SO101_ROBOT_ID @ $SO101_PORT"
echo "  Cameras: image=/dev/video6, image2=/dev/video2"
echo "  Policy: $SMOLVLA_POLICY_ID"
echo ""

nohup /home/shrek/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_server   --transport sse   --port 8765 > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "✓ MCP Server started (PID: $(cat $PID_FILE))"
echo "Server URL: http://localhost:8765/sse"
echo "Log file: $LOG_FILE"
echo ""

# Wait and show initial logs
sleep 5
echo "--- Initial logs ---"
tail -30 "$LOG_FILE"
