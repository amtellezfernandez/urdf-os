# SO101 Demo Analysis & AI Embodiment Plan

## 1. WHAT YOUR COLLEAGUE CHANGED

### Major Migration: SO100 → SO101
- **Renamed entire demo**: `so100_vla_demo/` → `so101_vla_demo/`
- **Why**: SO101 has TWO arms (leader + follower) for teleoperation
- **SO100**: Single arm (what we worked on earlier)
- **SO101**: Dual-arm system for teleoperation + VLA training

### Hardware Configuration
**Before (your setup):**
- Single SO100 follower arm @ `/dev/ttyACM0`
- Cameras: video4, video6

**After (colleague's setup):**
- **Follower arm**: `my_awesome_follower_arm` @ `/dev/ttyACM0`
- **Leader arm**: `my_awesome_leader_arm` @ `/dev/ttyACM1` (for teleoperation)
- **Cameras**: `/dev/video6` (front), `/dev/video2` (mount)
- **Camera names**: `image`, `image2` (to match policy expectations)

### Key Changes
1. **Removed mock robot support** - Always uses real hardware
2. **Added teleoperation support** - Leader arm controls follower
3. **Created simple start/stop scripts**:
   - `start_so101_demo.sh`
   - `stop_so101_demo.sh`
   - `restart_so101_demo.sh`
4. **Added eval script**: `run_eval_vla.sh` for running trained policies
5. **Both arms are calibrated**:
   - `my_awesome_follower_arm.json`
   - `my_awesome_leader_arm.json`

---

## 2. WHAT WE PREVIOUSLY FOUND

### Our Working Setup (SO100)
- ✅ Robot calibration working
- ✅ Camera streaming (wrist + overhead)
- ✅ MCP server with tools
- ✅ Robot control via MCP client
- ✅ Motion enabled successfully
- ✅ Home skill executed
- ❌ SmolVLA skill failed (no policy configured)
- ❌ Motor 6 detection issues (timing problem during startup)

### Key Issues We Solved
1. **Serial port permissions** - `sg dialout` workaround
2. **Camera discovery** - Finding robot cameras vs webcam
3. **Calibration process** - Avoiding 2048 overflow bug
4. **Robot ID requirement** - Added `SO100_ROBOT_ID` env var
5. **Camera streaming** - Added `get_robot_camera_frame()` MCP tool

---

## 3. POLICIES THEY'RE TRYING TO USE

### Configured Policies (from checkpoints.example.env)

#### SmolVLA
```bash
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so101_policy"
```
- **Type**: Vision-Language-Action model
- **Purpose**: Natural language instruction → robot action
- **Example**: "pick up the red cup"
- **Status**: HuggingFace checkpoint available

#### XVLA (Optional)
```bash
# export XVLA_POLICY_ID="/absolute/path/to/xvla_checkpoint"
```
- **Type**: Alternative VLA model
- **Purpose**: Similar to SmolVLA
- **Status**: Path not configured (commented out)

#### Eval Policy (Stack Cups)
```bash
EVAL_POLICY_PATH="Gowshigan/stackcupsv5"
```
- **Type**: Task-specific policy
- **Task**: "Stack cups"
- **Purpose**: Demonstration/evaluation
- **Status**: HuggingFace checkpoint available

### Policy Camera Mapping
```bash
export SO101_POLICY_CAMERA_MAP='{"front":"overhead","wrist":"wrist"}'
```
- Policies expect cameras named: `front`, `wrist`
- Robot provides: `image`, `image2`
- Mapping required to align expectations

---

## 4. WHAT'S MISSING FOR AI DEMO

### Critical Missing Components

#### ❌ Policy Not Loaded in Demo Server
**Current**: Demo server (`demo_script.py`) is running but **NO policy is loaded**
- Environment variables not set
- `SMOLVLA_POLICY_ID` needs to be exported before starting demo
- Server logs show no policy warmup/loading

#### ❌ MCP Server Not Running
**Current**: Only web UI server is running
- **Web UI**: ✅ Running on port 8000
- **MCP Server**: ❌ Not started
- **Issue**: Can't use AI tools to control robot

#### ❌ Camera Name Mismatch
**Policy expects**: `front`, `wrist`
**Robot provides**: `image`, `image2`
**Solution**: Either:
1. Rename cameras to match policy
2. Use `SO101_POLICY_CAMERA_MAP` correctly

#### ❌ Python Path Issue
**Issue**: Start script uses `python` but only `python3` exists
- Causes demo to fail silently
- We fixed it by using full conda path

#### ⚠️ SSH Access Limitations
- Can't directly test robot from your home
- Need to coordinate with colleague on-site
- Port forwarding working but limited interaction

---

## 5. HOW TO HAVE GREAT AI EMBODIED DEMO

### Setup Steps (On Colleague's Machine via SSH)

#### Step 1: Stop Current Demo
```bash
ssh -p 15619 shrek@7.tcp.eu.ngrok.io
# Password: shrek

cd urdf-os
./stop_so101_demo.sh
```

#### Step 2: Configure Policy Environment
```bash
# Load the policy configuration
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so101_policy"
export SO101_PORT=/dev/ttyACM0
export SO101_ROBOT_ID=my_awesome_follower_arm
export SO101_CAMERA_SOURCES=/dev/video6,/dev/video2
export SO101_CAMERA_NAMES=image,image2
export SO101_POLICY_CAMERA_MAP='{"image":"front","image2":"wrist"}'
export PYTHONPATH=/home/shrek/urdf-os/src
```

#### Step 3: Start MCP Server with Policy
```bash
/home/shrek/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_server \
  --transport sse \
  --port 8765 > /tmp/mcp_server.log 2>&1 &
```

#### Step 4: Verify Policy Loaded
```bash
# Check logs for policy loading
tail -f /tmp/mcp_server.log
# Should see: "Loading SmolVLA policy..."
```

#### Step 5: Connect and Control via AI

**From your machine** (with SSH tunnel):
```bash
# Terminal 1: SSH tunnel for MCP
ssh -L 8765:localhost:8765 -p 15619 shrek@7.tcp.eu.ngrok.io

# Terminal 2: Use MCP client or let Claude control it
python mcp_client.py connect_robot /dev/ttyACM0
python mcp_client.py enable_motion true
python mcp_client.py get_robot_camera_frame image
python mcp_client.py start_skill smolvla "pick up the cup" 100
```

### AI Embodiment Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: AI SEES                                             │
│  get_robot_camera_frame("image")  → Front view              │
│  get_robot_camera_frame("image2") → Mount/overhead view     │
│  → AI understands scene                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: AI PLANS                                            │
│  AI analyzes visual input                                   │
│  Formulates instruction: "pick up the red cup"              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: AI ACTS                                             │
│  connect_robot("/dev/ttyACM0")                              │
│  enable_motion(true)                                        │
│  start_skill("smolvla", "pick up the red cup", max_steps)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: AI MONITORS                                         │
│  Loop:                                                      │
│    - get_skill_status(skill_id)                             │
│    - get_robot_camera_frame() for visual feedback           │
│    - Adjust if needed                                       │
│  Until: status == "completed" or "failed"                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: AI REPORTS                                          │
│  "I successfully picked up the red cup"                     │
│  Shows before/after images                                  │
└─────────────────────────────────────────────────────────────┘
```

### Demo Script Examples

#### Example 1: Object Manipulation
```python
# AI sees the scene
camera_views = {
    "front": get_robot_camera_frame("image"),
    "overhead": get_robot_camera_frame("image2")
}

# AI identifies objects
"I see a red cup on the table to the right, and a blue block to the left"

# AI executes skill
start_skill("smolvla", "pick up the red cup and place it on the blue block", 100)

# AI monitors progress
while True:
    status = get_skill_status(skill_id)
    if status["status"] == "completed":
        break
    sleep(1)

# AI confirms
"Task completed! The red cup is now on top of the blue block."
```

#### Example 2: Multi-Step Task
```python
tasks = [
    "pick up the red cup",
    "move it to the left side of the table",
    "release the cup",
    "return to home position"
]

for i, task in enumerate(tasks):
    print(f"Step {i+1}: {task}")

    # Get current view
    image = get_robot_camera_frame("image")

    # Execute
    skill_id = start_skill("smolvla", task, 50)

    # Wait for completion
    wait_for_skill(skill_id)

    print(f"✓ Completed: {task}")
```

---

## 6. IMMEDIATE ACTION PLAN (< 1 hour)

### Priority 1: Fix Python Path (5 min)
```bash
# Edit start_so101_demo.sh
sed -i 's|nohup python|nohup /home/shrek/miniconda3/envs/lerobot/bin/python|g' start_so101_demo.sh
```

### Priority 2: Start MCP Server with Policy (10 min)
```bash
# Create start_mcp.sh script
cat > start_mcp.sh << 'EOF'
#!/bin/bash
export PYTHONPATH=/home/shrek/urdf-os/src
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so101_policy"
export SO101_PORT=/dev/ttyACM0
export SO101_ROBOT_ID=my_awesome_follower_arm
export SO101_CAMERA_SOURCES=/dev/video6,/dev/video2
export SO101_CAMERA_NAMES=image,image2

/home/shrek/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_server \
  --transport sse --port 8765 > /tmp/mcp_server.log 2>&1 &

echo "MCP Server started (PID: $!)"
echo "Logs: tail -f /tmp/mcp_server.log"
EOF

chmod +x start_mcp.sh
./start_mcp.sh
```

### Priority 3: Test AI Control (20 min)
```bash
# Terminal 1: SSH tunnel
ssh -L 8765:localhost:8765 -p 15619 shrek@7.tcp.eu.ngrok.io

# Terminal 2: Test MCP client
python mcp_client.py list_tools
python mcp_client.py connect_robot /dev/ttyACM0
python mcp_client.py get_robot_state
python mcp_client.py get_robot_camera_frame image
python mcp_client.py enable_motion true
python mcp_client.py start_skill home
```

### Priority 4: Run Demo with AI (20 min)
Let Claude/AI:
1. See through cameras
2. Identify objects
3. Execute pick-and-place
4. Monitor progress
5. Report results

---

## 7. DEMO SCENARIO IDEAS

### Scenario A: "Stack the Cups"
- AI sees 3 cups on table
- Instructs: "stack the three cups"
- Uses SmolVLA or stackcupsv5 policy
- Shows live camera feed during execution

### Scenario B: "Sort by Color"
- Red cup on left, blue block on right
- AI: "organize objects by color grouping"
- Demonstrates multi-step planning

### Scenario C: "Fetch and Deliver"
- "Pick up the object nearest to you and move it to the far end of the table"
- Tests spatial reasoning + manipulation

---

## SUMMARY

### What Works
- ✅ Dual-arm SO101 system calibrated
- ✅ Web UI running with camera feeds
- ✅ Teleoperation ready (leader → follower)
- ✅ Policies available on HuggingFace

### What's Broken
- ❌ MCP server not running
- ❌ Policy not loaded in any server
- ❌ Python path issue in start script
- ❌ Camera name mapping needs fix

### Quick Fix (Next 30 min)
1. Fix python path in start script
2. Start MCP server with policy
3. Test AI control via SSH tunnel
4. Run simple demo: "pick up the cup"

### For Great Demo
- Use SmolVLA with natural language
- Show live camera feeds
- Let AI narrate what it sees and does
- Demonstrate multi-step task completion
