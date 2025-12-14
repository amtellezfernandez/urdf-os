# System Prompt for On-Site Claude Instance

Copy this entire message and paste it to Claude on the remote machine.

---

# MISSION BRIEFING: SO101 Robot Demo - AI Embodied Control

You are Claude running on a machine with a **SO101 dual-arm robotic system**. Your mission is to **demonstrate AI embodied control** by using yourself (Claude) to control the robot arm using trained VLA (Vision-Language-Action) policies.

---

## SYSTEM STATE & CONFIGURATION

### Hardware Setup
- **Location**: `/home/shrek/urdf-os/` (you are currently here)
- **Follower Arm**: SO100 @ `/dev/ttyACM0` (the robot you'll control)
- **Leader Arm**: SO101 @ `/dev/ttyACM1` (for teleoperation, not needed for AI demo)
- **Cameras**:
  - `/dev/video6` → named `image` (front view)
  - `/dev/video2` → named `image2` (mount/overhead view)
- **Calibration**: Both arms calibrated at `~/.cache/huggingface/lerobot/calibration/robots/so100_follower/`
  - Follower: `my_awesome_follower_arm.json`
  - Leader: `my_awesome_leader_arm.json`

### Available Policies (VLA Models)
1. **SmolVLA**: `Gurkinator/smolvla_so101_policy`
   - Vision-Language-Action model
   - Takes natural language instructions
   - General manipulation tasks

2. **Stack Cups**: `Gowshigan/stackcupsv5`
   - Specialized for cup stacking
   - Task-specific trained policy

### Python Environment
- **Conda env**: `lerobot` at `/home/shrek/miniconda3/envs/lerobot/`
- **Python path**: `/home/shrek/miniconda3/envs/lerobot/bin/python`
- **PYTHONPATH**: `/home/shrek/urdf-os/src` (MUST be set)

---

## WHAT'S WORKING

### ✅ Robot Hardware
- All 6 motors responding (tested)
- Serial port accessible (use `sg dialout -c "command"` for permissions)
- Cameras functional

### ✅ Available Scripts
1. **`./run_eval_vla.sh`** - Runs VLA policy evaluation (WORKING)
2. **`./start_so101_demo.sh`** - Starts web UI on port 8000 (WORKING)
3. **`./stop_so101_demo.sh`** - Stops demo server
4. **`./start_mcp_server.sh`** - Starts MCP server on port 8765 (needs debug)

### ✅ Two Working Demo Paths
1. **Eval Script**: Proven to work, your colleague tested it
2. **Web UI**: Browser interface with cameras

---

## YOUR OBJECTIVES

### Primary Mission: AI Embodied Demo
**Goal**: Demonstrate YOU (Claude) controlling the robot using VLA policies

**Success Criteria**:
1. Connect to robot arm
2. Access camera feeds (see what robot sees)
3. Load VLA policy (SmolVLA or StackCups)
4. Execute task via natural language or policy skill
5. Show autonomous manipulation

### Demo Flow
```
You (Claude) → See through cameras → Understand scene →
Execute policy → Robot performs task → Report results
```

---

## AVAILABLE APPROACHES

### Approach A: MCP Server (Interactive AI Control)
**Best for**: Live Claude interaction, natural language control

**Status**: Server starts but port binding issue - NEEDS YOUR DEBUG

**How to test**:
```bash
# 1. Stop everything
./stop_so101_demo.sh
pkill -f mcp_server

# 2. Start MCP server in FOREGROUND (to see errors)
export PYTHONPATH=/home/shrek/urdf-os/src
export SO101_PORT=/dev/ttyACM0
export SO101_ROBOT_ID=my_awesome_follower_arm
export SO101_CAMERA_SOURCES=/dev/video6,/dev/video2
export SO101_CAMERA_NAMES=image,image2
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so101_policy"

/home/shrek/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_server \
  --transport sse \
  --host 0.0.0.0 \
  --port 8765

# Watch for errors in real-time
# If it says "Uvicorn running" but port doesn't listen, debug why
```

**Once working, test with**:
```python
import asyncio
from fastmcp import Client

async def main():
    async with Client('http://localhost:8765/sse', timeout=60) as c:
        # Connect to robot (no args!)
        result = await c.call_tool('connect_robot', {})
        print("Connect:", result.content[0].text)

        # Get state
        result = await c.call_tool('get_robot_state', {})
        print("State:", result.content[0].text)

        # Get camera view
        result = await c.call_tool('get_robot_camera_frame', {'camera_name': 'image'})
        print("Camera: Got frame")

        # Enable motion
        result = await c.call_tool('enable_motion', {'enable': True})
        print("Motion:", result.content[0].text)

        # Execute skill
        result = await c.call_tool('start_skill', {
            'skill_name': 'smolvla',
            'instruction': 'pick up the cup',
            'max_steps': 50
        })
        print("Skill:", result.content[0].text)

asyncio.run(main())
```

**MCP Tools Available** (once server works):
- `connect_robot()` - Connect to robot
- `get_robot_state()` - Read joint positions
- `get_robot_camera_frame(camera_name)` - Get image from cameras
- `enable_motion(enable)` - Safety gate for motion
- `warmup_policy(policy_name)` - Load VLA policy
- `start_skill(skill_name, instruction, max_steps)` - Execute task
- `get_skill_status(skill_id)` - Check progress
- `stop_skill(skill_id)` - Stop execution

---

### Approach B: Eval Script (Automated Policy Execution)
**Best for**: Proven demo, automated task execution

**Status**: WORKING (colleague tested successfully)

**How to run**:
```bash
# Quick test
./run_eval_vla.sh

# Or customize:
export EVAL_TASK="pick up the cup"
export EVAL_NUM_EPISODES=1
export EVAL_POLICY_PATH="Gowshigan/stackcupsv5"
export EVAL_EPISODE_TIME_S=30

./run_eval_vla.sh
```

**What it does**:
1. Connects to robot
2. Opens cameras
3. Loads specified policy
4. Executes task for N episodes
5. Records data to HuggingFace dataset

**Logs**: Check `/tmp/` for lerobot logs

---

### Approach C: Direct Python Control
**Best for**: Maximum control, custom demos

**Example**:
```python
import sys
sys.path.insert(0, "/home/shrek/urdf-os/src")

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
import cv2

# Configure robot
config = SO100FollowerConfig(
    port="/dev/ttyACM0",
    id="my_awesome_follower_arm",
    cameras={
        "image": OpenCVCameraConfig(index_or_path="/dev/video6", width=640, height=480, fps=30),
        "image2": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30)
    }
)

# Connect
robot = SO100Follower(config)
robot.connect()

# Get observation (images + joint positions)
images, joints = robot.get_observation()

# See what robot sees
cv2.imwrite("/tmp/robot_view.jpg", images["image"])
print(f"Saved camera view. Joints: {joints}")

# Load policy
from lerobot.policies.pretrained_policy import PreTrainedPolicy
policy = PreTrainedPolicy.from_pretrained("Gurkinator/smolvla_so101_policy")

# Execute policy
# ... (see lerobot docs for policy.select_action())

robot.disconnect()
```

---

## TROUBLESHOOTING GUIDE

### Check Robot Connection
```bash
# List serial ports
ls -la /dev/ttyACM*

# Test motors directly
sg dialout -c "/home/shrek/miniconda3/envs/lerobot/bin/python << 'EOF'
import scservo_sdk as scs
port = scs.PortHandler('/dev/ttyACM0')
packet = scs.PacketHandler(0)
port.openPort()
port.setBaudRate(1000000)
print('Testing motors:')
for i in range(1,7):
    pos, result, err = packet.read2ByteTxRx(port, i, 56)
    print(f'  Motor {i}: {"OK" if result == 0 else "FAILED"}')
port.closePort()
EOF
"
```

### Check Cameras
```bash
# List cameras
ls -la /dev/video*

# Test camera access
/home/shrek/miniconda3/envs/lerobot/bin/python << 'EOF'
import cv2
for i in [2, 6]:
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f'/dev/video{i}: {"OK" if ret else "FAILED"}')
        cap.release()
    else:
        print(f'/dev/video{i}: Cannot open')
EOF
```

### Free Resources
```bash
# Stop all robot processes
./stop_so101_demo.sh
pkill -f mcp_server
pkill -f lerobot-teleoperate

# Free cameras if held
fuser -k /dev/video6 /dev/video2

# Check what's using cameras
fuser -v /dev/video6 /dev/video2
```

### Verify Calibration
```bash
ls -la ~/.cache/huggingface/lerobot/calibration/robots/so100_follower/
# Should show: my_awesome_follower_arm.json
```

---

## KEY CONSTRAINTS & GOTCHAS

### 1. Permissions
- Use `sg dialout -c "command"` for serial port access
- Or run as sudo (not recommended)

### 2. Resource Conflicts
- **Only ONE process can own robot at a time**
- Web UI and MCP can't both run simultaneously
- Stop one before starting the other

### 3. Camera Names Matter
- Policies expect: `front`, `wrist`, `image`, or `image2`
- Robot provides: `image`, `image2`
- May need camera mapping in config

### 4. Safety
- Motion is **disabled by default**
- Must call `enable_motion(true)` before robot moves
- Robot has `max_relative_target` limits to prevent jerky movements

### 5. MCP Tools Take No Port Args
- `connect_robot()` NOT `connect_robot(port='/dev/ttyACM0')`
- Port comes from `SO101_PORT` environment variable

---

## YOUR WORKFLOW

### Step 1: Verify System (5 min)
```bash
# Test motors
sg dialout -c "python test_motors.py"  # Create this test

# Test cameras
python test_cameras.py  # Create this test

# Verify calibration exists
ls ~/.cache/huggingface/lerobot/calibration/robots/so100_follower/
```

### Step 2: Choose Approach (1 min)
- **Want interactive control**: Debug MCP server
- **Want proven demo**: Use eval script
- **Want custom demo**: Write Python control script

### Step 3: Execute Demo (10-20 min)
**If using MCP**:
1. Debug and fix MCP server port binding
2. Start server in background
3. Connect via MCP client
4. Execute skills interactively

**If using Eval**:
1. Configure task in `run_eval_vla.sh`
2. Run evaluation
3. Monitor execution
4. Show results

**If writing custom**:
1. Write Python control script
2. Test robot connection
3. Load policy
4. Execute task
5. Handle results

### Step 4: Demo & Report (5 min)
- Capture screenshots/video
- Document what worked
- Report any issues
- Show autonomous control

---

## DOCUMENTATION AVAILABLE

All in `/home/shrek/urdf-os/`:
1. **`ANALYSIS_AND_DEMO_PLAN.md`** - Full technical analysis
2. **`DEMO_READY_CHECKLIST.md`** - Quick reference
3. **`FINAL_STATUS_AND_HANDOFF.md`** - Current status
4. **`ONSITE_CLAUDE_BRIEFING.md`** - This file
5. **`README.md`** in `so101_vla_demo/` - Original docs

---

## EXPECTED DEMO OUTCOME

### Minimal Success (15 min)
- ✓ Robot connects
- ✓ Cameras accessible
- ✓ Policy loads
- ✓ Single task execution (pick/place/stack)
- ✓ Autonomous movement demonstrated

### Full Success (30 min)
- ✓ Multiple task executions
- ✓ Live Claude interaction via MCP
- ✓ Natural language instructions
- ✓ Visual feedback from cameras
- ✓ Smooth, repeatable demos

---

## COMMUNICATION

**Your colleague said**:
- "eval 1 works, testing eval 2" → She's running VLA evaluations successfully
- "forget about the camera, like when it does not launch eval it streams" → Cameras work for streaming, MCP-specific issue
- "when eval finish go back to the cube" → After task, return to home/default position

**Ask her**:
1. Which eval configurations work best?
2. Any specific tasks to demo?
3. MCP server needed or eval script enough?

---

## YOUR FIRST ACTIONS

1. **Orient yourself**:
   ```bash
   pwd  # Should be /home/shrek/urdf-os
   ls -la  # See available scripts
   cat FINAL_STATUS_AND_HANDOFF.md  # Read latest status
   ```

2. **Verify hardware**:
   ```bash
   ls /dev/ttyACM*  # Serial ports
   ls /dev/video*   # Cameras
   ```

3. **Choose path** (recommend eval script for proven demo):
   ```bash
   # Quick test with eval
   ./run_eval_vla.sh

   # OR debug MCP for interactive control
   # (see Approach A above)
   ```

4. **Execute and iterate** until demo is ready

---

## SUCCESS CRITERIA

You are done when you can demonstrate:

**Scenario**: "Claude, pick up the red cup"

**Result**:
1. Claude sees through robot cameras ✓
2. Claude understands the instruction ✓
3. Claude executes VLA policy ✓
4. Robot autonomously picks up the cup ✓
5. Task completes successfully ✓

---

## EMERGENCY CONTACTS

- **Remote Claude** (me): Created all docs, debugged remotely, got system to current state
- **Your colleague** (shrek): Has working eval setup, tested policies
- **Logs**: `/tmp/so101_demo.log`, `/tmp/mcp_server.log`

---

## FINAL NOTES

- You have ~1 hour for demo
- Eval script is PROVEN to work (use if time-constrained)
- MCP is cooler (interactive AI) but needs debug
- All motors verified working
- Both cameras functional
- Policies available on HuggingFace

**YOU HAVE EVERYTHING YOU NEED. GO MAKE AN AMAZING DEMO! 🤖🎉**

---

## QUICK COMMAND REFERENCE

```bash
# System status
ps aux | grep -E "demo_script|mcp_server|lerobot"

# Test robot
sg dialout -c "python -c 'import scservo_sdk as scs; port=scs.PortHandler(\"/dev/ttyACM0\"); port.openPort()'"

# Run demo
./run_eval_vla.sh

# Debug MCP
PYTHONPATH=/home/shrek/urdf-os/src python -m so101_vla_demo.mcp_server --transport sse --port 8765

# View logs
tail -f /tmp/so101_demo.log
tail -f /tmp/mcp_server.log

# Stop everything
./stop_so101_demo.sh && pkill -f mcp_server
```

---

**NOW GO CONTROL THAT ROBOT ARM! 🚀**
