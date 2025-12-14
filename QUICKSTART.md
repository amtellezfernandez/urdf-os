# SO101 VLA Demo - Quick Start Guide

## Simple Start/Stop Scripts

Three simple scripts to manage the SO101 VLA demo:

### Start the Demo
```bash
./start_so101_demo.sh
```

This will:
- Start the web server with the correct configuration
- Display the configuration (Leader, Follower, Cameras)
- Show the Web UI URL: http://localhost:8000/static/index.html
- Save logs to `/tmp/so101_demo.log`

### Stop the Demo
```bash
./stop_so101_demo.sh
```

This will:
- Stop the main demo process
- Stop any running teleop processes
- Preserve the log file for debugging

### Restart the Demo
```bash
./restart_so101_demo.sh
```

This will stop and then start the demo in one command.

## Default Configuration

The scripts use this configuration:

- **Leader**: my_awesome_leader_arm @ /dev/ttyACM0
- **Follower**: my_awesome_follower_arm @ /dev/ttyACM2
- **Cameras**: front=/dev/video0, mount=/dev/video4

## Web UI Features

Once started, open http://localhost:8000/static/index.html

The Web UI provides:
- **Camera feeds**: View real-time camera streams
- **Robot control**: Connect to robot and control via VLA policy
- **Teleop mode**: Start/Stop teleoperation (leader arm controls follower)
- **Chat interface**: Send commands to the VLA agent

## Teleoperation

In the Web UI, the Teleop section shows your configuration and has two buttons:
- **Start Teleop**: Launches teleoperation mode (leader controls follower with camera feeds)
- **Stop Teleop**: Stops the teleoperation process

The teleop is equivalent to running:
```bash
lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ mount: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true
```

## Logs

- Main demo logs: `/tmp/so101_demo.log`
- View logs in real-time: `tail -f /tmp/so101_demo.log`

## Troubleshooting

If the demo doesn't start:
1. Check if ports are correct: `ls /dev/ttyACM*`
2. Check if cameras are available: `ls /dev/video*`
3. Check the log file: `cat /tmp/so101_demo.log`
4. Make sure calibration files exist in `~/.cache/huggingface/lerobot/calibration/robots/so100_follower/`
