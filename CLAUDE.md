# SO101 VLA Demo - Claude Code Guide

## Quick Start

### Environment Setup (CRITICAL)
Always activate the conda environment before running any scripts:
```bash
source /home/shrek/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export PYTHONPATH=/home/shrek/urdf-os/src
```

### Start MCP Server
```bash
./start_mcp_server.sh
# Or manually:
/home/shrek/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_server \
  --transport sse --host 0.0.0.0 --port 8765
```

### Run Eval Script
```bash
./run_eval_vla.sh
```

## Hardware Configuration

| Component | Device Path | Notes |
|-----------|-------------|-------|
| Follower Arm | `/dev/ttyACM0` | SO101 6-DOF arm |
| Leader Arm | `/dev/ttyACM1` | For teleoperation |
| Camera 1 (image) | `/dev/video6` | Front camera |
| Camera 2 (image2) | `/dev/video2` | Wrist camera |

### Environment Variables
```bash
SO101_PORT=/dev/ttyACM0
SO101_ROBOT_ID=my_awesome_follower_arm
SO101_CAMERA_SOURCES=/dev/video6,/dev/video2
SO101_CAMERA_NAMES=image,image2
SO101_POLICY_CAMERA_MAP='{"image":"front","image2":"wrist"}'
```

## Available Policies

| Policy | HuggingFace ID | Task |
|--------|----------------|------|
| SmolVLA | `Gurkinator/smolvla_so101_policy` | General manipulation |
| StackCups (XVLA) | `Gowshigan/stackcupsv5` | Stack cups task |

## MCP Server Tools

### Connection & Status
- `connect_robot()` - Connect to the SO101 arm
- `disconnect_robot()` - Disconnect safely
- `get_robot_status()` - Check connection state

### Camera
- `get_robot_camera_frame(camera_name)` - Get image from camera
  - camera_name: "image" (front) or "image2" (wrist)

### Policy Execution
- `warmup_policy(policy_type)` - Load policy into memory
  - policy_type: "smolvla" or "xvla"
- `enable_motion(enabled)` - Safety gate (must be True to move)
- `start_skill(policy_type, instruction, max_steps)` - Execute policy
- `get_skill_status(skill_id)` - Monitor execution
- `stop_skill(skill_id)` - Emergency stop

### Example Workflow
```python
connect_robot()
get_robot_camera_frame("image")  # Verify camera works
warmup_policy("xvla")
enable_motion(True)
skill_id = start_skill("xvla", instruction="stack the cups", max_steps=50)
# Monitor with get_skill_status(skill_id)
# Stop with stop_skill(skill_id) if needed
```

## Common Issues & Fixes

### 1. Tensor Shape Error
**Error**: `(b,c,h,w) expected, but got torch.Size([3, 480, 640])`
**Fix**: Applied in `mcp_server.py` lines 186-196 - adds batch dimension to tensors.

### 2. Motor Detection Failure
**Error**: `Missing motor IDs: - 3 (expected model: 777)`
**Cause**: FeetechMotorsBus broadcast_ping timing differs from direct scservo.
**Fix**: Power cycle the robot arm, or check USB connections.

### 3. Wrong LeRobot Installation
**Error**: `KeyError: 'xvla'` when running eval
**Cause**: Two lerobot installations exist:
- `/home/shrek/urdf-os/src/lerobot/` - Has XVLA (correct)
- `/home/shrek/lerobot/` - Missing XVLA (wrong)
**Fix**: Always set `PYTHONPATH=/home/shrek/urdf-os/src`

### 4. Permission Denied on Serial Port
**Error**: `Permission denied opening serial device '/dev/ttyACM0'`
**Fix**: `sudo usermod -a -G dialout $USER` then log out/in

## File Structure
```
/home/shrek/urdf-os/
├── so101_vla_demo/
│   ├── mcp_server.py      # MCP server with VLA tools
│   ├── robot_interface.py # Robot connection wrapper
│   └── config.py          # Configuration parsing
├── src/lerobot/           # LeRobot with XVLA support
├── run_eval_vla.sh        # Eval recording script
├── start_mcp_server.sh    # MCP server launcher
└── .mcp.json              # Claude Code MCP config
```

## Calibration
Robot ID must match calibration: `my_awesome_follower_arm`
Calibration files: `~/.cache/huggingface/lerobot/calibration/`
