#!/bin/bash
# SO101 VLA Demo Shutdown Script

PID_FILE=/tmp/so101_demo.pid
LOG_FILE=/tmp/so101_demo.log

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Demo may not be running."
    echo "Checking for running processes..."

    # Try to find any running demo processes
    PIDS=$(pgrep -f "so101_vla_demo.demo_script")
    if [ -z "$PIDS" ]; then
        echo "No SO101 demo processes found."
        exit 0
    else
        echo "Found running processes: $PIDS"
        echo "Stopping processes..."
        kill $PIDS 2>/dev/null
        sleep 2

        # Force kill if still running
        if pgrep -f "so101_vla_demo.demo_script" > /dev/null; then
            echo "Force stopping remaining processes..."
            pkill -9 -f "so101_vla_demo.demo_script"
        fi
        echo "✓ All processes stopped."
        exit 0
    fi
fi

PID=$(cat "$PID_FILE")
echo "Stopping SO101 VLA Demo (PID: $PID)..."

if ps -p $PID > /dev/null 2>&1; then
    kill $PID
    echo "Waiting for graceful shutdown..."
    sleep 2

    # Check if still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Process still running, force killing..."
        kill -9 $PID
    fi

    echo "✓ Demo stopped."
else
    echo "Process $PID not found (already stopped)."
fi

# Also stop any lerobot-teleoperate processes that might be running
TELEOP_PIDS=$(pgrep -f "lerobot-teleoperate")
if [ ! -z "$TELEOP_PIDS" ]; then
    echo "Stopping teleop processes: $TELEOP_PIDS"
    kill $TELEOP_PIDS 2>/dev/null
    sleep 1
    # Force kill if needed
    pkill -9 -f "lerobot-teleoperate" 2>/dev/null
fi

rm -f "$PID_FILE"
echo ""
echo "Log file preserved at: $LOG_FILE"
