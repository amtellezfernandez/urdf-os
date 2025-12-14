#!/bin/bash
# SO101 VLA Demo Restart Script

echo "Restarting SO101 VLA Demo..."
echo ""

# Stop the demo
./stop_so101_demo.sh

echo ""
echo "---"
echo ""

# Start the demo
./start_so101_demo.sh
