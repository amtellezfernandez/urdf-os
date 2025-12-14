#!/bin/bash
# SO101 VLA Demo Startup Script

# Configuration
# Follower on port 0, leader on port 1
export SO101_PORT=/dev/ttyACM0
export SO101_ROBOT_ID=my_awesome_follower_arm
export SO101_LEADER_PORT=/dev/ttyACM1
export SO101_LEADER_ID=my_awesome_leader_arm
export SO101_CAMERA_SOURCES=/dev/video0,/dev/video4
export SO101_CAMERA_NAMES=front,mount
export PYTHONPATH=/home/shrek/urdf-os/src

# PID file to track the running process
PID_FILE=/tmp/so101_demo.pid
LOG_FILE=/tmp/so101_demo.log

# Start the demo
echo "Starting SO101 VLA Demo..."
echo "Configuration:"
echo "  Leader:   $SO101_LEADER_ID @ $SO101_LEADER_PORT"
echo "  Follower: $SO101_ROBOT_ID @ $SO101_PORT"
echo "  Cameras:  front=/dev/video0, mount=/dev/video4"
echo ""

nohup python -m so101_vla_demo.demo_script > "$LOG_FILE" 2>&1 &
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
