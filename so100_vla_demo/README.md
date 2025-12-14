# SO100 VLA Demo Scaffold

## Quick Start (Team Member Setup)

### 1. Clone and Install

```bash
git clone <repo-url>
cd urdf-os

# Create conda environment
conda create -n lerobot python=3.10 -y
conda activate lerobot

# Install lerobot from repo root
pip install -e .

# Install additional dependencies for MCP server
pip install fastmcp
```

### 2. Hardware Permissions (Linux)

```bash
# Add user to required groups
sudo usermod -a -G dialout,video $USER
# Log out and back in for changes to take effect
```

### 3. Run the Web UI (Camera Streaming)

```bash
# Find your cameras first
lerobot-find-cameras opencv

# Start server (adjust camera sources based on above)
USE_MOCK_ROBOT=true \
USE_REAL_CAMERAS=true \
SO100_CAMERA_SOURCES="/dev/video4,/dev/video6" \
SO100_CAMERA_NAMES="wrist,overhead" \
python -m so100_vla_demo.demo_script
```

Open http://localhost:8000/static/index.html → Click **Connect** to see cameras.

To use the **real robot arm** (not mock):
```bash
USE_MOCK_ROBOT=false \
SO100_PORT=/dev/ttyACM0 \
SO100_CAMERA_SOURCES="/dev/video4,/dev/video6" \
SO100_CAMERA_NAMES="wrist,overhead" \
python -m so100_vla_demo.demo_script
```

### 4. Run MCP Server (AI Controls Robot via Claude Code)

**Step 1:** Create `.mcp.json` in the repo root (per-developer; not committed).

We ship a template: copy `.mcp.example.json` → `.mcp.json`, then update paths for YOUR machine:
```bash
# Find your values:
which python                    # → e.g. /home/USER/miniconda3/envs/lerobot/bin/python
pwd                             # → e.g. /home/USER/urdf-os
lerobot-find-cameras opencv     # → e.g. /dev/video4, /dev/video6
ls /dev/ttyACM* /dev/ttyUSB*    # → e.g. /dev/ttyACM0
```

Then edit `.mcp.json`:
```json
{
  "mcpServers": {
    "so100-vla": {
      "command": "/home/USER/miniconda3/envs/lerobot/bin/python",
      "args": ["-m", "so100_vla_demo.mcp_server"],
      "cwd": "/home/USER/urdf-os",
      "env": {
        "PYTHONPATH": "/home/USER/urdf-os/src",
        "SO100_CAMERA_SOURCES": "/dev/video4,/dev/video6",
        "SO100_CAMERA_NAMES": "wrist,overhead",
        "SO100_PORT": "auto",
        "SMOLVLA_POLICY_ID": "Gurkinator/smolvla_so100_policy"
      }
    }
  }
}
```
Replace `/home/USER/...` with your actual paths.

**Step 2:** Restart Claude Code (it auto-loads MCP servers from `.mcp.json`)

**Step 3:** Claude Code can now control the robot:
```
# See cameras
list_cameras()
get_camera_frame("wrist")

# Connect robot
connect_robot("auto")  # or connect_robot("mock") for safe testing

# Run VLA skill with natural language
start_skill("smolvla", instruction="pick up the red cup")
get_skill_status("skill_xxx")
stop_skill("skill_xxx")  # interrupt if needed

# Optional: pre-load checkpoint before a live demo (avoids first-call download/load delays)
warmup_policy("smolvla")
```

Tip: you can also keep policy/checkpoint settings in a shell env file:
- `so100_vla_demo/checkpoints.example.env` (copy and edit)

### 5. Demo Flow (AI Agent Controlling Robot)

1. **Claude Code sees** → `get_camera_frame("wrist")`
2. **Claude Code thinks** → "I see a red cup on the table"
3. **Claude Code acts** → `start_skill("smolvla", instruction="pick up the red cup")`
4. **Claude Code monitors** → `get_skill_status(...)` + `get_camera_frame("wrist")`
5. **Claude Code stops/replans** → `stop_skill(...)` if needed

### 6. MCP Client Smoke Test (No Claude)

This repo includes a small client that calls the MCP tools (in-process, no sockets required):

```bash
python -m so100_vla_demo.mcp_client_demo
python -m so100_vla_demo.mcp_client_demo --policy Gurkinator/smolvla_so100_policy --steps 2
```

---

This folder contains a **self‑contained scaffold** for a LeRobot hackathon demo with the SO100 arm:

- Use LeRobot’s `SO100Follower` to talk directly to the arm (no DDS/ROS).
- Record your own demos (`lerobot-record`) for **search** and **grasp**.
- Train policies locally or on your AMD MI300X VM (Diffusion / SmolVLA / XVLA).
- Wrap them as **skills**: `search_object` and `grasp_object`.
- Orchestrate a **“search then grasp”** demo, driven by a VLM/LLM (Gemini, Claude, Qwen, or stub).
- Provide a small FastAPI/WebSocket **server** for a web UI: camera stream + chat.

The code here is deliberately lightweight and model‑agnostic: you bring your policies and LLM backend; this gives you the wiring.

---

## High‑Level Design & Reasoning

### Goals

1. **Use your own data** (teleop) and policies, not a black‑box SOTA config.
2. **Separate skills cleanly**:
   - `search_object` → move camera/arm to *see* the object (active perception).
   - `grasp_object` → run a visuomotor policy to *manipulate* once object is visible.
3. **Let the LLM plan** in terms of skills, not raw joint angles.
4. **Avoid unnecessary infrastructure**:
   - No MCP runtime, no DDS, no ROS required for SO100.
   - Just LeRobot, your policies, and a small VLM/LLM client.

### Why a learned `search_object()` skill

We explicitly **do not** want a hard‑coded “scan pattern”:

- A fixed scan is blind: it checks poses A, B, C regardless of context.
- A learned `search_object` policy from teleop demos can capture human‑like intuition:
  - If you see the edge of a table, you know to check *on* the table, not under it.
  - If you see a box, you naturally peek around it.
  - You move smoothly and efficiently rather than jerkily.

So the idea is:

- Record episodes where you **only search and center** the object (e.g. tennis ball) with the camera.
- Train a policy that maps **(image + robot state) → small end‑effector motion**.
- Wrap this as a `search_object` skill that runs until the object is visible, then stop.

Once search succeeds, a second policy (SmolVLA/XVLA/diffusion) takes over to grasp.

### Why not use existing MCP / DDS stack

For the LeRobot hackathon with SO100:

- LeRobot already gives a clean SO100 robot API and camera integration.
- DDS and your Unitree sim stack aren’t needed.
- MCP adds complexity that doesn’t buy much for a focused demo.

So we treat:

- **LLM side** as a simple pluggable client (Gemini/Claude/Qwen) via `LLMEngine`.
- **Robot side** as LeRobot’s `SO100Follower` class.
- **Skills** as plain Python functions you can call from the LLM or a fixed script.

---

## Components in This Folder

### Core Config & Robot Wrapper

- `config.py`
  - `SO100DemoConfig`:
    - `port`: SO100 serial port (`SO100_PORT` env or `/dev/ttyUSB0`).
    - `camera_indexes`: OpenCV camera sources for all cameras (from `SO100_CAMERA_SOURCES`, `SO100_CAMERA_INDEXES`, or legacy `SO100_CAMERA_INDEX`).
      - Example: `SO100_CAMERA_SOURCES="0,2,/dev/video4"`
    - `demo_fps`: control / streaming FPS (default 15).
    - `use_mock`: if `True` (or `USE_MOCK_ROBOT=true`), use a **mock robot** instead of real SO100.
    - `use_real_cameras`: when `use_mock=True`, stream from real cameras if `USE_REAL_CAMERAS=true` (otherwise synthetic).
    - `mock_video_path`, `mock_static_image_path`: optional sources for mock camera frames.
    - Optional `search_policy_path`, `grasp_policy_path`.
  - `to_robot_config()` → creates `SO100FollowerConfig` with a `wrist` OpenCV camera.

- `robot_interface.py`
  - `SO100RobotInterface`:
    - `connect()` / `disconnect()` around `SO100Follower`.
    - `get_observation() -> (image, joints_dict)`:
      - image: first camera frame (HxWxC, `np.uint8`).
      - joints: `{joint_name: position_float}`.
    - `send_joint_targets(joint_targets: dict[str, float])`:
      - wraps into `{f"{name}.pos": value}` and calls `robot.send_action(...)`.
  - `make_robot_interface(cfg: SO100DemoConfig)`:
    - Returns a real `SO100RobotInterface` when `cfg.use_mock` is `False`.
    - Returns a `MockRobotInterface` when `cfg.use_mock` is `True`.

- `mock_robot_interface.py`
  - `MockRobotInterface`:
    - Implements the same API as `SO100RobotInterface` (`connect`, `disconnect`, `get_observation`, `send_joint_targets`).
    - Can return:
      - synthetic scenes with a table and colored objects (red cup, blue block, green ball),
      - frames from a video file (`MOCK_VIDEO_PATH`),
      - or a static image (`MOCK_STATIC_IMAGE_PATH`).
    - Tracks an internal fake joint dictionary for display / debugging.
  - This is how you and the judges can run the full demo **without the real robot**.

### Policy Skills

#### Search Skill (`search_object`)

- `search_skill.py`
  - `SearchPolicySkill`:
    - `policy_path: Optional[Path]`, `preprocessor`, `postprocessor`, `policy`.
    - Uses `PreTrainedPolicy.from_pretrained(policy_path)` from LeRobot.
    - `step(image, joints) -> action_dict`:
      - Builds a minimal batch with `observation.image` (and later state).
      - Applies preprocessor → `policy.select_action(...)` → postprocessor.
      - Returns `{ACTION-prefixed keys: float}`.
    - `run_search_loop(robot, object_name, detect_fn, max_steps) -> (found, steps)`:
      - Loop:
        - `image, joints = robot.get_observation()`.
        - `detect_fn(image, object_name)` decides if the object is visible.
        - If visible → return `True, step`.
        - Else → call `step(...)`, convert action keys to joint names, and send via `robot.send_joint_targets`.
      - If `policy_path is None`, orchestrator can skip this phase.

#### Grasp Skill (`grasp_object`)

- `grasp_skill.py`
  - `GraspPolicySkill`:
    - Same pattern as `SearchPolicySkill` but for grasping.
    - `run_grasp_loop(robot, max_steps)`:
      - Repeatedly:
        - read `(image, joints)`,
        - call `step(...)`,
        - send joint targets.
      - You can later add success detection (e.g. gripper state, object height) and early stopping.

### Orchestrator

- `demo_orchestrator.py`
  - `SO100DemoOrchestrator`:
    - Holds:
      - `cfg: SO100DemoConfig`
      - `robot: SO100RobotInterface`
      - `search_skill: SearchPolicySkill`
      - `grasp_skill: GraspPolicySkill`
      - `detect_fn(frame, object_name) -> bool`
    - `run(object_name)`:
      - `robot.connect()`
      - If `search_skill.policy_path` is set:
        - `found, steps = search_skill.run_search_loop(...)`.
        - If not found → abort.
      - `grasp_skill.run_grasp_loop(...)`.
      - `robot.disconnect()`.

### CLI Demo Runner

- `run_demo.py`
  - CLI usage:
    ```bash
    python -m so100_vla_demo.run_demo \
      --object-name "tennis ball" \
      --search-policy-path /path/to/search_pretrained_model \
      --grasp-policy-path /path/to/grasp_pretrained_model \
      --so100-port /dev/ttyUSB0 \
      --camera-index 0
    ```
  - If `--search-policy-path` is omitted:
    - The orchestrator skips search and runs grasp directly.
  - If `--grasp-policy-path` is omitted:
    - The script warns that grasp won’t use a real policy (you’ll wire it later).
  - Uses a placeholder `simple_pixel_detector` that always returns `False` (so you don’t move the robot unexpectedly before wiring a real detector / VLM).

---

## LLM Abstraction (Gemini / Claude / Qwen / Stub)

### Config

- `llm_config.py`
  - `LLMConfig`:
    - `provider`: `"gemini"`, `"claude"`, `"qwen"` or `"stub"` (anything else → stub).
    - `model_name`: e.g. `"gemini-1.5-flash"`, `"claude-3-5-sonnet"`.
    - `api_key_env`: name of env var where API key is stored.
  - `LLMConfig.load(config_path=None)`:
    1. If `config_path` is a JSON file, load it.
    2. Else, read:
       - `LLM_PROVIDER`
       - `LLM_MODEL`
       - `LLM_API_KEY_ENV`
    3. Else, use defaults (`gemini-1.5-flash`, `GEMINI_API_KEY`).

- `llm_config.json`
  - Small JSON config file living next to the Python modules:
    ```json
    {
      "provider": "stub",
      "model_name": "gemini-1.5-flash",
      "api_key_env": "GEMINI_API_KEY"
    }
    ```
  - The server first tries to load this file. You can edit it to switch provider/model.
  - Put your secret in the environment variable specified by `api_key_env`, e.g.:
    ```bash
    export GEMINI_API_KEY=your_key_here
    ```
    (never commit your real API key to git).

### Engines

- `llm_engine.py`
  - `BaseLLMEngine.chat(messages, tools=None) -> dict`.
  - Implementations:
    - `GeminiEngine(LLMConfig)`
    - `ClaudeEngine(LLMConfig)`
    - `QwenEngine(LLMConfig)`
    - `StubEngine()` – returns a simple echo message:
      - `"[STUB LLM] I received: '...'. Configure a real LLM to get meaningful answers."`
  - `make_llm_engine(cfg: Optional[LLMConfig]) -> BaseLLMEngine`:
    - provider `"gemini"` → `GeminiEngine`.
    - provider `"claude"` → `ClaudeEngine`.
    - provider `"qwen"` → `QwenEngine`.
    - anything else → `StubEngine`.

All real engines currently raise `NotImplementedError` in `chat()`. This is deliberate: for debug you don’t need a real API yet, and you won’t accidentally leak keys. You’ll fill in the HTTP calls later.

---

## Debug Server & Client Interface

### FastAPI WebSocket Server

- `server.py`
  - Start it from the LeRobot repo root:
    ```bash
    uvicorn so100_vla_demo.server:app --host 0.0.0.0 --port 8000
    ```
  - WebSocket endpoint: `ws://localhost:8000/ws`.
  - Static frontend served from:
    - `http://localhost:8000/` (redirects to `http://localhost:8000/static/index.html`)
  - Uses:
    - `SO100DemoConfig` → `make_robot_interface(cfg)` → real or mock robot.
    - `LLMConfig.load("llm_config.json")` if present, or env vars.
    - `make_llm_engine(...)` for LLM (stub fallback).

#### Messages from client

- Start camera streaming:
  ```json
  {"type": "command", "action": "start_stream"}
  ```
  - Server:
    - Connects the robot interface lazily (real or mock).
    - Repeatedly:
      - `frame, joints = robot_interface.get_observation()`.
      - Builds a small JPEG thumbnail and sends it as base64.
      - Broadcasts:
        ```json
        {
          "type": "frame",
          "shape": [H, W, C],
          "image_b64": "...",     // base64-encoded JPEG
          "joints": {"joint_0": 0.1, ...}
        }
        ```

- Stop streaming:
  ```json
  {"type": "command", "action": "stop_stream"}
  ```

- Chat:
  ```json
  {"type": "chat", "text": "Hello, what do you see?"}
  ```
  - Server:
    - Calls `llm_engine.chat(...)`.
    - If the real engine raises `NotImplementedError`, falls back to `StubEngine`.
    - Returns:
      ```json
      {"type": "chat", "text": "... stub or real LLM reply ..."}
      ```

This is enough to debug:

- Camera connectivity (`shape` & `thumbnail_size`).
- WebSocket / UI pipeline.
- LLM request/response loop (via stub).
Later you can extend this further (see below).

---

## MCP Skills (No API Key Needed)

If you don’t have an LLM/VLM API key, you can still do **agentic control** by running an MCP server and letting
Claude Code (or another MCP client) call tools:

- camera tools: `list_cameras`, `get_camera_frame`
- robot tools: `connect_robot`, `get_robot_state`
- skill tools: `start_skill("smolvla"| "xvla", instruction="...")`, `stop_skill`, `get_skill_status`

### Adding a Policy Checkpoint (Easy / For Any Repo User)

You don’t need to edit code. A checkpoint can be either:
- a Hugging Face model repo id (recommended), or
- a local folder path.

**Requirement:** for language VLA skills we need a *LeRobot-exported* checkpoint that includes processor + stats
files (not only `config.json` and `model.safetensors`).

Minimum expected files:
- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- `policy_preprocessor_step_*_normalizer_processor.safetensors`
- `policy_postprocessor_step_*_unnormalizer_processor.safetensors`

#### Option A: Use environment variables (simplest)

Example (public SmolVLA checkpoint for SO100):

```bash
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so100_policy"
```

If your policy expects camera names like `front` but your robot uses `overhead`, set:

```bash
export SO100_POLICY_CAMERA_MAP='{"front":"overhead","wrist":"wrist"}'
```

#### Option B: Set it at runtime via MCP tool

Call the tool:

```text
set_policy(policy_name="smolvla", policy_id="Gurkinator/smolvla_so100_policy")
```

### Run MCP Server

**Option A: For Claude Code (stdio transport)**
```bash
python -m so100_vla_demo.mcp_server
```
Configure via `.mcp.json` (see Quick Start above).

**Option B: As HTTP server for any AI client (SSE transport)**
```bash
# Set environment first
export PYTHONPATH=/path/to/urdf-os/src
export SO100_CAMERA_SOURCES="/dev/video4,/dev/video6"
export SO100_CAMERA_NAMES="wrist,overhead"
export SO100_PORT="/dev/ttyACM0"
export SMOLVLA_POLICY_ID="Gurkinator/smolvla_so100_policy"

# Run as HTTP server
python -m so100_vla_demo.mcp_server --transport sse --host 0.0.0.0 --port 8765
```
Clients connect to: `http://localhost:8765/sse`

### Example MCP “See → Act” flow (from the client)

1) See:
- `list_cameras()`
- `get_camera_frame("wrist")`

2) Connect:
- `connect_robot("auto")` (or pass a specific port)

3) Act (language instruction):
- `start_skill("smolvla", instruction="pick up the blue block", max_steps=80)`
- `get_skill_status(skill_id)`
- `stop_skill(skill_id)` if you want to interrupt/replan

## How to Use This Scaffold (Step‑by‑Step)

### 1. Connect SO100 and test camera manually

Before running anything here:

1. Use a simple script (or `lerobot-record` with 1–2 episodes) to verify:
   - The SO100 arm connects on your chosen port.
   - The wrist camera index is correct and returns frames.

### 2. Run the debug server and test over WebSocket

From the root of the LeRobot repo:

```bash
uvicorn so100_vla_demo.server:app --host 0.0.0.0 --port 8000
```

Then connect with a WebSocket client (e.g. `websocat` or your own UI):

- Start stream:
  ```json
  {"type": "command", "action": "start_stream"}
  ```
- You should see `{"type":"frame", ...}` messages with image shapes.
- Send a chat:
  ```json
  {"type": "chat", "text": "Hello!"}
  ```
  and see a stub LLM reply.

At this stage, no policies or real LLM are needed; you’re only validating wiring.

### 3. Full mock demo in the browser (no hardware)

For teammates and judges without the SO100 arm, you can run a **fully self‑contained mock demo**:

```bash
python -m so100_vla_demo.demo_script
```

This will:

- Default to real hardware when it looks like SO100 is connected, otherwise fall back to `USE_MOCK_ROBOT=true`.
- You can force mock mode with `USE_MOCK_ROBOT=true`.
- Start the FastAPI server on `http://localhost:8000`.
- Serve the web UI at `http://localhost:8000/static/index.html`.

In the browser you can:

- Click **Connect** → **Start Stream** to see a synthetic scene with a table and colored objects.
- Watch:
  - Live camera frames (mock or real).
  - Status messages (idle/searching/grasping/done).
  - Reasoning logs describing what the agent is “thinking”.
- Chat with the LLM (stub by default) in the right panel.

Chat commands:

- `/stream on` / `/stream off`
- `/help`

Under the hood, in mock mode:

- The server runs a simple scripted `search_and_grasp` sequence:
  - Search phase: pans fake joints and logs reasoning.
  - Grasp phase: short scripted sequence with more logs.
  - Final status: `{"type": "status", "phase": "done"}`.

Once you have real policies and object detection, you can replace this scripted behavior with a proper `search_object` + `grasp_object` implementation while keeping the same UI.

### 4. Record your own datasets (search & grasp)

On the SO100 machine, use `lerobot-record` twice:

1. **Search dataset** (e.g. `my_so100_search_for_ball`):
   - Start with the ball hidden/partially occluded.
   - Teleoperate the arm/camera to **search until the ball is visible and centered**.
   - Each episode ends when you can clearly see the ball.

2. **Grasp dataset** (e.g. `my_so100_grasp_ball`):
   - Start from poses where the ball is visible.
   - Teleoperate the arm to grasp and lift the ball.

Set `--dataset.root` to a local directory (no Hub) and `--dataset.push_to_hub=false`. Then rsync the dataset folders to your AMD VM for training.

### 5. Train policies on AMD VM

On the MI300X machine (with ROCm‑enabled PyTorch):

1. **Search policy**:
   ```bash
   lerobot-train \
     --dataset.repo_id=my_so100_search_for_ball \
     --dataset.root=/path/to/datasets \
     --policy.type=diffusion \  # or act, simple model first
     --policy.push_to_hub=false \
     --steps=50000 \
     --batch_size=64
   ```
   - This learns `(image + state) → Δpose` for search.

2. **Grasp policy**:
   ```bash
   lerobot-train \
     --dataset.repo_id=my_so100_grasp_ball \
     --dataset.root=/path/to/datasets \
     --policy.type=smolvla \    # or xvla/diffusion, depending on your design
     --policy.push_to_hub=false \
     --steps=200000 \
     --batch_size=64
   ```

Each training run produces a `pretrained_model` directory containing `config.json` and `model.safetensors`. That is what `SearchPolicySkill` and `GraspPolicySkill` load.

### 6. Plug trained policies into the demo runner

Once you have policies, copy or mount the `pretrained_model` dirs onto the SO100 machine and run:

```bash
python -m so100_vla_demo.run_demo \
  --object-name "tennis ball" \
  --search-policy-path /path/to/search/pretrained_model \
  --grasp-policy-path /path/to/grasp/pretrained_model \
  --so100-port /dev/ttyUSB0 \
  --camera-sources "0"
```

The orchestrator will:

1. Connect SO100.
2. Run the search skill until your detection function says the ball is visible.
3. Run the grasp skill for a fixed horizon.
4. Disconnect.

Make sure to:

- Extend `SearchPolicySkill.step` and `GraspPolicySkill.step` so the observation keys (`observation.image`, `observation.state`, etc.) match what your policies expect.
- Replace `simple_pixel_detector` with a real detector (e.g. a small VLM call or a classical detector) that checks if the ball is visible in the current frame.

### 7. Connect a real LLM (Gemini / Qwen / Claude)

After everything works with the stub:

1. Implement `chat()` in `GeminiEngine`, `QwenEngine`, or `ClaudeEngine` using your API client.
2. Set LLM configuration:
   - Via env:
     ```bash
     export LLM_PROVIDER=gemini
     export LLM_MODEL=gemini-1.5-flash
     export LLM_API_KEY_ENV=GEMINI_API_KEY
     export GEMINI_API_KEY=...your_key...
     ```
3. Restart the server:
   ```bash
   uvicorn so100_vla_demo.server:app --host 0.0.0.0 --port 8000
   ```
4. Your web UI now:
   - Receives live frames (and later thumbnails/images).
   - Sends chat messages that go through a real LLM.
   - Can be extended to issue high‑level commands like “search for the tennis ball” → server maps that to a proper `search_and_grasp` implementation using your trained policies.

---

## What’s Missing / To‑Do for Final Demo

1. **Policy training and wiring**
   - Train real search and grasp policies on your datasets.
   - Wire them into `SearchPolicySkill.step` and `GraspPolicySkill.step` with correct feature keys and shapes.

2. **Real object detection**
   - Replace `simple_pixel_detector` with:
     - a small detector (color/shape based), or
     - a VLM call (send frame to Gemini/Qwen and parse “ball visible?”).

3. **LLM integration**
   - Implement real `chat()` for Gemini/Qwen/Claude in `llm_engine.py`.
   - Optionally use function‑calling to let the LLM invoke:
     - `search_object` (orchestrator’s search phase),
     - `grasp_object` (grasp phase),
     - low‑level moves if needed.

4. **UI/frontend**
   - Build a small web UI that:
     - Connects to `ws://server/ws`,
     - Shows frame thumbnails / live video (extend server to send JPEG bytes),
     - Has a chat panel,
     - Shows planner reasoning / skill calls.

5. **Safety & limits**
   - Define safe workspaces / joint limits for autonomous motion.
   - Add emergency stop in your teleop hardware or UI.
   - Add checks in `run_search_loop` and `run_grasp_loop` to avoid long uncontrolled motion.

6. **Competition polish**
   - Log trials (videos + text) for your demo.
   - Prepare a short script showing:
     1. Initial partial view with hidden ball.
     2. LLM decides to “search”.
     3. `search_object` skill executes until ball is visible.
     4. LLM decides to “grasp”.
     5. `grasp_object` skill grasps the ball.

This scaffold should give you a clear path from **ideas** (active perception + VLA manipulation) to a working hackathon demo using SO100, LeRobot, your AMD GPU, and whatever LLM backend you prefer.
