#!/bin/bash
# SO101 VLA Demo Startup Script

# Configuration
# Follower on port 0, leader on port 1 (use auto-discovery by default)
export SO101_PORT=/dev/ttyACM0
export SO101_ROBOT_ID=my_awesome_follower_arm
export SO101_LEADER_PORT=/dev/ttyACM1
export SO101_LEADER_ID=my_awesome_leader_arm
export SO101_CAMERA_SOURCES=/dev/video6,/dev/video2
# Name cameras to match policy expectations: image (front), image2 (mount)
export SO101_CAMERA_NAMES=image,image2
export PYTHONPATH=/home/shrek/urdf-os/src

# Eval VLA settings
# Run only 1 episode since leader arm is broken (can't reset position between episodes)
export EVAL_NUM_EPISODES=1

# Use a unique dataset root for eval runs to avoid FileExistsError on repeated launches.
# If you want a fixed path, override EVAL_DATASET_ROOT before running this script.
: "${EVAL_DATASET_ROOT:=$(mktemp -d /tmp/so101_eval_XXXXXX)}"
export EVAL_DATASET_ROOT

# PID file to track the running process
PID_FILE=/tmp/so101_demo.pid
LOG_FILE=/tmp/so101_demo.log

# Start the demo
echo "Starting SO101 VLA Demo..."
echo "Configuration:"
echo "  Leader:   $SO101_LEADER_ID @ $SO101_LEADER_PORT"
echo "  Follower: $SO101_ROBOT_ID @ $SO101_PORT"
echo "  Cameras:  front=/dev/video6, mount=/dev/video2"
echo ""

nohup /home/shrek/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.demo_script > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "✓ Demo started (PID: $(cat $PID_FILE))"
echo ""
echo "Web UI: http://localhost:8000/static/index.html"
echo "Log file: $LOG_FILE"
echo ""
echo "To stop: ./stop_so101_demo.sh"

# Wait a moment and show initial logs
sleep 3
echo ""
echo "--- Recent logs ---"
tail -20 "$LOG_FILE"
