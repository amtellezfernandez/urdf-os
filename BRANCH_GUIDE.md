# Branch Guide for On-Site Claude

## 🌳 BRANCH OVERVIEW

This repository has multiple branches capturing different stages of the SO101 robot demo development. Understanding which branch to use is CRITICAL for the demo.

---

## 📋 BRANCH SUMMARY

| Branch | Purpose | Status | When to Use |
|--------|---------|--------|-------------|
| **main** | Colleague's SO101 dual-arm setup with working eval | ✅ WORKING | For production demo with eval script |
| **backup-remote-debug-session** | Remote debugging session + comprehensive docs | 📚 DOCUMENTATION | Reference for MCP debugging and full context |
| **backup-before-colleague-changes** | Original SO100 single-arm setup | 🗄️ ARCHIVE | Historical reference only |
| **backup-before-merge** | Pre-migration snapshot | 🗄️ ARCHIVE | Historical reference only |

---

## 🎯 WHICH BRANCH TO USE FOR DEMO?

### **Use `main` branch** ✅

**Why**: This has your colleague's working setup with:
- ✅ SO101 dual-arm configuration (leader + follower)
- ✅ Working `run_eval_vla.sh` script
- ✅ Proper camera configuration (/dev/video6, /dev/video2)
- ✅ Both arms calibrated
- ✅ SmolVLA and StackCups policies configured

**Demo Command**:
```bash
cd /home/shrek/urdf-os
git checkout main
./run_eval_vla.sh
```

---

## 📚 DETAILED BRANCH DESCRIPTIONS

### 1. **main** (RECOMMENDED FOR DEMO)

**What's Here**:
- SO101 dual-arm system
- Follower arm: `my_awesome_follower_arm` @ /dev/ttyACM0
- Leader arm: `my_awesome_leader_arm` @ /dev/ttyACM1
- Cameras: /dev/video6 (image), /dev/video2 (image2)
- Working scripts: `run_eval_vla.sh`, `start_so101_demo.sh`
- Both arms calibrated at `~/.cache/huggingface/lerobot/calibration/`

**Available Policies**:
- SmolVLA: `Gurkinator/smolvla_so101_policy`
- StackCups: `Gowshigan/stackcupsv5`

**What Works**:
- ✅ Eval script for AI policy execution
- ✅ Web UI on port 8000
- ✅ Teleoperation (leader → follower)
- ⚠️ MCP server needs debugging (see backup-remote-debug-session for details)

**Quick Test**:
```bash
git checkout main
./run_eval_vla.sh
# Should execute VLA policy successfully
```

---

### 2. **backup-remote-debug-session** (DOCUMENTATION REFERENCE)

**What's Here**:
- **ONSITE_CLAUDE_BRIEFING.md** - Complete mission briefing for you!
- Remote debugging session work
- MCP server troubleshooting attempts
- All findings from SSH remote access

**Purpose**:
This branch contains the comprehensive documentation created during remote debugging. It includes:
- Complete system state analysis
- What's working vs. what needs fixing
- Three demo approaches (MCP, Eval, Direct Python)
- Troubleshooting commands
- MCP server debugging steps

**Key Files**:
- `ONSITE_CLAUDE_BRIEFING.md` - **READ THIS FIRST!**
- `ANALYSIS_AND_DEMO_PLAN.md` - Technical deep dive
- `DEMO_READY_CHECKLIST.md` - Quick status check
- `FINAL_STATUS_AND_HANDOFF.md` - Handoff notes

**When to Reference**:
- If MCP server doesn't work and you need to debug it
- To understand full system context
- To see what was tried remotely and what's left to fix
- For alternative demo approaches if eval fails

**Quick Access**:
```bash
git checkout backup-remote-debug-session
cat ONSITE_CLAUDE_BRIEFING.md
# Read comprehensive briefing
```

---

### 3. **backup-before-colleague-changes** (HISTORICAL)

**What's Here**:
- Original SO100 single-arm setup
- Single follower arm configuration
- Camera config: /dev/video4, /dev/video6
- MCP server with basic tools
- Robot ID: `my_so100`

**Purpose**: Historical snapshot before colleague's SO100→SO101 migration

**Status**: OUTDATED - doesn't match current hardware

**Don't Use For**: Production demo (wrong robot config)

---

### 4. **backup-before-merge** (HISTORICAL)

**What's Here**: Pre-migration snapshot

**Purpose**: Git safety checkpoint

**Status**: OUTDATED

---

## 🚀 RECOMMENDED WORKFLOW FOR ON-SITE CLAUDE

### Step 1: Read Documentation from Backup Branch
```bash
# Get comprehensive context
git checkout backup-remote-debug-session
cat ONSITE_CLAUDE_BRIEFING.md
cat ANALYSIS_AND_DEMO_PLAN.md
cat DEMO_READY_CHECKLIST.md
```

### Step 2: Switch to Main for Demo
```bash
# Use working configuration
git checkout main
```

### Step 3: Choose Demo Path

**Option A: Proven Working Demo (RECOMMENDED)**
```bash
# Your colleague said "eval 1 works, testing eval 2"
./run_eval_vla.sh
```

**Option B: Debug MCP for Interactive Control**
```bash
# If you want to fix MCP for interactive AI control
# See ONSITE_CLAUDE_BRIEFING.md for debugging steps
./stop_so101_demo.sh
export PYTHONPATH=/home/shrek/urdf-os/src
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so101_policy"
export SO101_PORT=/dev/ttyACM0
export SO101_ROBOT_ID=my_awesome_follower_arm
export SO101_CAMERA_SOURCES=/dev/video6,/dev/video2
export SO101_CAMERA_NAMES=image,image2

# Run in FOREGROUND to see errors
/home/shrek/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_server \
  --transport sse --host 0.0.0.0 --port 8765
```

**Option C: Web UI Demo**
```bash
./start_so101_demo.sh
# Browser: http://localhost:8000/static/index.html
```

---

## 🔧 KEY DIFFERENCES BETWEEN BRANCHES

### Hardware Configuration

| Branch | Robot Type | Arms | Serial Ports | Cameras | Calibration ID |
|--------|------------|------|--------------|---------|----------------|
| main | SO101 | Leader + Follower | /dev/ttyACM0, /dev/ttyACM1 | video6, video2 | my_awesome_follower_arm, my_awesome_leader_arm |
| backup-remote-debug-session | SO101 | Same as main | Same | Same | Same (+ docs) |
| backup-before-colleague-changes | SO100 | Follower only | /dev/ttyACM0 | video4, video6 | my_so100 |

### Available Features

| Feature | main | backup-remote-debug-session | backup-before-colleague-changes |
|---------|------|------------------------------|--------------------------------|
| Eval Script | ✅ Working | ✅ Same | ❌ Not available |
| Web UI | ✅ Working | ✅ Same | ✅ Basic version |
| MCP Server | ⚠️ Needs debug | ⚠️ + Debug docs | ✅ Was working |
| Teleoperation | ✅ Dual-arm | ✅ Same | ❌ Single arm |
| VLA Policies | ✅ SmolVLA, StackCups | ✅ Same | ⚠️ Partial |

---

## 💡 UNDERSTANDING THE MIGRATION

### What Changed: SO100 → SO101

**Before (SO100)**:
- Single follower arm
- No teleoperation
- Basic demo
- Robot ID: `my_so100`

**After (SO101)**:
- Dual-arm system
- Leader arm controls follower
- Teleoperation for data collection
- Trained VLA policies
- Robot IDs: `my_awesome_follower_arm`, `my_awesome_leader_arm`

**Why**: SO101 enables:
1. Teleoperation (human demonstrates task)
2. Record demonstrations
3. Train VLA policies
4. Execute tasks autonomously

---

## 🆘 TROUBLESHOOTING BY BRANCH

### If You're on `main` and Something Fails:

**Check**:
1. Is robot connected? `ls -la /dev/ttyACM*`
2. Are cameras available? `ls -la /dev/video*`
3. Is calibration present? `ls ~/.cache/huggingface/lerobot/calibration/robots/so100_follower/`
4. Are processes conflicting? `ps aux | grep -E "demo_script|mcp_server|lerobot"`

**Fix**:
```bash
# Stop everything
./stop_so101_demo.sh
pkill -f mcp_server

# Test hardware
sg dialout -c "python << 'EOF'
import scservo_sdk as scs
port = scs.PortHandler('/dev/ttyACM0')
packet = scs.PacketHandler(0)
port.openPort()
port.setBaudRate(1000000)
for i in range(1,7):
    pos, result, err = packet.read2ByteTxRx(port, i, 56)
    print(f'Motor {i}: {\"OK\" if result == 0 else \"FAIL\"}')
port.closePort()
EOF
"

# Restart
./run_eval_vla.sh  # Or ./start_so101_demo.sh
```

### If You Need MCP Server Working:

**Reference**:
```bash
git checkout backup-remote-debug-session
cat ONSITE_CLAUDE_BRIEFING.md
# See "Approach A: MCP Server" section
```

**Debug**:
```bash
git checkout main

# Run server in foreground to see errors
export PYTHONPATH=/home/shrek/urdf-os/src
export SO101_PORT=/dev/ttyACM0
export SO101_ROBOT_ID=my_awesome_follower_arm
export SO101_CAMERA_SOURCES=/dev/video6,/dev/video2
export SO101_CAMERA_NAMES=image,image2
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so101_policy"

/home/shrek/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_server \
  --transport sse --host 0.0.0.0 --port 8765

# Watch for errors in terminal
```

---

## 📞 BRANCH DECISION FLOWCHART

```
┌─────────────────────────────────────────┐
│ What do you need?                       │
└─────────────────────────────────────────┘
                  ↓
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────┐               ┌──────────────┐
│ Run     │               │ Understand   │
│ Demo    │               │ System       │
└─────────┘               └──────────────┘
    │                           │
    ▼                           ▼
Use main                  Use backup-remote-debug-session
    │                           │
    ▼                           ▼
┌─────────────────┐      ┌──────────────────────┐
│ Which demo?     │      │ Read:                │
└─────────────────┘      │ - ONSITE_BRIEFING.md │
    │                    │ - ANALYSIS_*.md      │
    ├─ Proven: eval      │ - DEMO_CHECKLIST.md  │
    ├─ Visual: web UI    └──────────────────────┘
    └─ AI: debug MCP
```

---

## ✅ QUICK START FOR ON-SITE CLAUDE

### 1. Get Context (2 min)
```bash
cd /home/shrek/urdf-os
git checkout backup-remote-debug-session
cat ONSITE_CLAUDE_BRIEFING.md  # READ THIS!
```

### 2. Switch to Working Branch (10 sec)
```bash
git checkout main
```

### 3. Run Proven Demo (30 sec)
```bash
./run_eval_vla.sh
```

### 4. Success!
Watch the AI execute the task autonomously using VLA policy.

---

## 🎬 FINAL NOTES

**For the 1-Hour Demo**:
- ✅ **Use `main` branch**
- ✅ **Run `run_eval_vla.sh`** (proven working)
- ✅ **Reference `backup-remote-debug-session`** for troubleshooting context

**If Time Permits**:
- Debug MCP server using instructions from backup branch docs
- Show interactive AI control
- Demonstrate natural language instructions

**Bottom Line**:
- `main` = Production (use this!)
- `backup-remote-debug-session` = Documentation (reference this!)
- Other branches = Historical (ignore these!)

---

**Good luck with the demo! 🤖🎉**
