# SO100 VLA Demo – Full System Architecture

This document dives into the detailed architecture: modules, data flow, message formats, and how things should evolve once real policies and VLMs are plugged in.

---

## 1. Module Overview

### 1.1 Frontend (Browser)

**File:** `so100_vla_demo/static/index.html`

Responsibilities:

- WebSocket client for `/ws`:
  - Sends JSON messages for chat and commands.
  - Receives JSON messages with frames, status, reasoning, chat replies.
- UI layout:
  - Camera panel (`<img id="video">`).
  - Controls: connect/disconnect, start/stop stream, quick “find object” buttons.
  - Chat panel: user ↔ assistant messages.
  - Reasoning log: status + internal thoughts from the backend.
- Minimal state:
  - `isConnected`, current phase (`Idle`, `Streaming`, `Searching`, `Grasping`, `Done`).

---

### 1.2 Backend Server

**File:** `so100_vla_demo/server.py`

Key parts:

- `FastAPI` app:
  - Configures CORS for local development.
  - Mounts static files (`/static -> so100_vla_demo/static`).
- `ConnectionManager`:
  - Tracks active WebSocket connections.
  - Provides `broadcast_json(message)` for multicasting frames/status/reasoning.
- Global singletons:
  - `demo_cfg: SO100DemoConfig` (config.py).
  - `llm_cfg: LLMConfig` (llm_config.py / llm_config.json).
  - `llm_engine: BaseLLMEngine` (llm_engine.py).
  - `robot_interface: SO100RobotInterface | MockRobotInterface` (robot_interface.py / mock_robot_interface.py via `make_robot_interface`).
  - Async tasks:
    - `stream_task` – camera streaming loop.
    - `behavior_task` – `search_and_grasp` loop (mock mode now).

#### 1.2.1 Camera Streaming Loop

**Function:** `camera_stream_loop()`

- Connect robot (real or mock) lazily via `robot_interface.connect()`.
- Loop while `streaming` flag is `True`:
  - Call `frame, joints = robot_interface.get_observation()`.
  - Convert `frame` (HxWxC `uint8`) to a JPEG thumbnail.
  - Base64‑encode JPEG to string: `image_b64`.
  - Broadcast:
    ```json
    {
      "type": "frame",
      "shape": [H, W, C],
      "image_b64": "...",
      "joints": {"joint_0": 0.1, "joint_1": -0.2, "...": "..."}
    }
    ```
  - `await asyncio.sleep(1.0 / demo_cfg.demo_fps)`.

#### 1.2.2 WebSocket Endpoint

**Path:** `/ws`

Main loop:

1. Accept connection, add to `ConnectionManager`.
2. For each incoming message:
   - Parse JSON.
   - Switch on `data["type"]`:

**Case 1 – Chat**

- Request:
  ```json
  {"type": "chat", "text": "Hello, what do you see?"}
  ```
- Server:
  - Calls `await llm_engine.chat(messages=[{"role": "user", "content": text}], tools=None)`.
  - On `NotImplementedError`, falls back to `StubEngine`.
  - Reply:
    ```json
    {"type": "chat", "text": "...reply from LLM..."}
    ```

**Case 2 – Commands**

- General shape:
  ```json
  {"type": "command", "action": "...", ...}
  ```

Supported `action` values:

1. `"start_stream"`:
   - If not already streaming:
     - Set `streaming = True`.
     - Start `stream_task = asyncio.create_task(camera_stream_loop())`.
   - Reply:
     ```json
     {"type": "status", "text": "streaming_started", "phase": "streaming"}
     ```

2. `"stop_stream"`:
   - Set `streaming = False`.
   - Cancel `stream_task` if running.
   - Reply:
     ```json
     {"type": "status", "text": "streaming_stopped", "phase": "idle"}
     ```

3. `"search_and_grasp"`:
   - Request:
     ```json
     {"type": "command", "action": "search_and_grasp", "object": "red cup"}
     ```
   - Cancel any existing `behavior_task`.
   - **Mock mode** (current implementation):
     - Start `behavior_task = asyncio.create_task(_mock_search_and_grasp(object_name))`.
     - Reply:
       ```json
       {
         "type": "status",
         "text": "search_and_grasp started for 'red cup' (mock mode)",
         "phase": "searching"
       }
       ```
   - **Real mode** (future):
     - For now, returns:
       ```json
       {
         "type": "status",
         "text": "search_and_grasp is only implemented in mock mode for now.",
         "phase": "idle"
       }
       ```
     - Later, will call real `SearchPolicySkill` + `GraspPolicySkill`.

**Case 3 – Unknown Types/Actions**

- Reply with:
  ```json
  {"type": "error", "text": "Unknown message type: '...'"}
  ```
  or
  ```json
  {"type": "error", "text": "Unknown command action: '...'"}
  ```

---

## 2. Robot Abstraction

### 2.1 Configuration

**File:** `so100_vla_demo/config.py`

- `SO100DemoConfig` fields:
  - `port`: SO100 serial port (default `/dev/ttyUSB0`, or `SO100_PORT`).
  - `camera_indexes`: camera sources (OpenCV indices or `/dev/video*` paths) from `SO100_CAMERA_SOURCES` / `SO100_CAMERA_INDEXES` / legacy `SO100_CAMERA_INDEX`.
  - `demo_fps`: streaming/control FPS (default `15`).
  - `use_mock`: from `USE_MOCK_ROBOT` (`"1"/"true"/"yes"` → True).
  - `use_real_cameras`: when `use_mock=True`, stream from real cameras if `USE_REAL_CAMERAS=true`.
  - `mock_video_path`: optional video file path (`MOCK_VIDEO_PATH` env).
  - `mock_static_image_path`: optional static image path (`MOCK_STATIC_IMAGE_PATH` env).
  - `search_policy_path`, `grasp_policy_path`: optional paths to trained policies.
- `to_robot_config()`:
  - Builds a `SO100FollowerConfig` with one `OpenCVCameraConfig` per configured camera.

### 2.2 Real Robot Interface

**File:** `so100_vla_demo/robot_interface.py`

- `SO100RobotInterface`:
  - `config: SO100FollowerConfig`.
  - `connect()` → best-effort auto-detects serial port if needed and calls `SO100Follower.connect(calibrate=...)` (controlled by `SO100_CALIBRATE`).
  - `disconnect()` → calls `robot.disconnect()`.
  - `get_observation()`:
    - Calls `robot.get_observation()` (LeRobot).
    - Extracts all configured camera frames and joint positions (`{name: obs[f"{name}.pos"]}`).
  - `send_joint_targets(joint_targets)`:
    - Converts to `{f"{name}.pos": value}` and calls `robot.send_action(action)`.
- `make_robot_interface(cfg: SO100DemoConfig)`:
  - If `cfg.use_mock` is `True`, returns a `MockRobotInterface`.
  - Else, returns `SO100RobotInterface(cfg.to_robot_config())`.

### 2.3 Mock Robot Interface

**File:** `so100_vla_demo/mock_robot_interface.py`

- `MockRobotInterface`:
  - Holds:
    - `cfg: SO100DemoConfig`.
    - `joints: dict[str, float]` with some fake joint names.
    - Optional `_static_image`, `_video_cap`, `_frame_index`.
  - `connect()`:
    - Loads static image (if `mock_static_image_path`).
    - Opens video capture (if `mock_video_path` and OpenCV available).
  - `disconnect()`:
    - Releases any OpenCV resources.
  - `get_observation()`:
    - If video configured → returns next frame (looping).
    - Else if static image → returns that.
    - Else → returns a synthetic scene:
      - Dark background, table rectangle, colored shapes (red cup, blue block, green ball) that change over time to mimic “search”.
    - Returns `(frame, joints_copy)`.
  - `send_joint_targets(joint_targets)`:
    - Simply updates `self.joints` (no hardware).

---

## 3. Skills and Policies

### 3.1 Search Policy Skill

**File:** `so100_vla_demo/search_skill.py`

- `SearchPolicySkill`:
  - Attributes:
    - `policy_path: Optional[Path]`.
    - `preprocessor`, `postprocessor`: `PolicyProcessorPipeline | None`.
    - `policy: PreTrainedPolicy | None`.
  - `load()`:
    - Calls `PreTrainedPolicy.from_pretrained(policy_path)`.
  - `step(image, joints) -> Dict[str, float]`:
    - Builds a batch dict with keys like `observation.image`.
    - Applies `preprocessor`, then `policy.select_action(batch)`, then `postprocessor`.
    - Filters `ACTION-*` keys, converts to `{str: float}`.
  - `run_search_loop(robot, object_name, detect_fn, max_steps)`:
    - Loop:
      - `image, joints = robot.get_observation()`.
      - If `detect_fn(image, object_name)` → stop (object found).
      - Else:
        - `action_dict = step(image, joints)`.
        - Convert keys `"...pos"` to joint names and send via `robot.send_joint_targets`.
    - Returns `(found: bool, steps_taken: int)`.

### 3.2 Grasp Policy Skill

**File:** `so100_vla_demo/grasp_skill.py`

- `GraspPolicySkill`:
  - Similar to `SearchPolicySkill`, but focused on the manipulation phase.
  - `run_grasp_loop(robot, max_steps)`:
    - Loop:
      - `image, joints = robot.get_observation()`.
      - `action_dict = step(image, joints)`.
      - Convert and send joint targets.
    - Future extension: early stopping when grasp success is detected.

### 3.3 High‑Level Orchestrator

**File:** `so100_vla_demo/demo_orchestrator.py`

- `SO100DemoOrchestrator`:
  - Fields:
    - `cfg: SO100DemoConfig`.
    - `robot: SO100RobotInterface`.
    - `search_skill: SearchPolicySkill`.
    - `grasp_skill: GraspPolicySkill`.
    - `detect_fn(frame, object_name) -> bool`.
  - `run(object_name)`:
    - `robot.connect()`.
    - If `search_skill.policy_path` is set:
      - `found, steps = search_skill.run_search_loop(...)`.
      - If not found → log warning and return.
    - Run `grasp_skill.run_grasp_loop(...)`.
    - `robot.disconnect()`.

### 3.4 CLI Entry Point

**File:** `so100_vla_demo/run_demo.py`

- Parses arguments:
  - `--object-name`,
  - `--search-policy-path`,
  - `--grasp-policy-path`,
  - `--so100-port`,
  - `--camera-index`.
- Builds `SO100DemoConfig`, `SO100RobotInterface`, `SearchPolicySkill`, `GraspPolicySkill`.
- Uses a placeholder `simple_pixel_detector(frame, object_name)` (always `False`) – intentionally safe until a real detector is plugged in.
- Calls `SO100DemoOrchestrator.run(object_name)`.

---

## 4. LLM / VLM Integration

### 4.1 Config and Engines

**Files:**

- `so100_vla_demo/llm_config.py`
  - `LLMConfig` dataclass:
    - `provider`, `model_name`, `api_key_env`.
  - `LLMConfig.load(config_path=None)` reads JSON or env vars.
- `so100_vla_demo/llm_config.json`
  - Default config used by server.
- `so100_vla_demo/llm_engine.py`
  - `BaseLLMEngine` abstract class with `async chat(...)`.
  - Implementations:
    - `GeminiEngine`, `ClaudeEngine`, `QwenEngine`: structure only, `chat()` raises `NotImplementedError`.
    - `StubEngine`: local echo‑style engine for testing.
  - `make_llm_engine(cfg)` chooses engine based on `cfg.provider`.

### 4.2 Planned Skill‑Calling Pattern

In the current code, the WebSocket layer only supports:

- Free‑form chat (text ↔ text).
- Predefined command `search_and_grasp`.

Later, you can extend `chat` responses so the LLM:

- Returns function/tool calls such as:
  - `search_object(object_description: str)`.
  - `grasp_object(object_description: str)`.
- The server interprets these calls and invokes:
  - `SearchPolicySkill` / `GraspPolicySkill` on the real robot, or
  - Equivalent behavior in mock mode.

This keeps the architecture **hierarchical**:

- LLM decides which skill to call.
- Policies implement low‑level robot control.

---

## 5. Runtime Scenarios (End‑to‑End)

### 5.1 Mock Browser Demo – “Find Red Cup”

1. User runs:
   ```bash
   python3 -m so100_vla_demo.demo_script
   ```
2. `demo_script.py` sets `USE_MOCK_ROBOT=true` and starts FastAPI/uvicorn.
3. User opens `http://localhost:8000/static/index.html`.
4. UI:
   - Connects to `/ws`.
   - Sends `{"type": "command", "action": "start_stream"}`.
   - Displays incoming `frame` messages.
5. User clicks “Find Red Cup”:
   - UI sends:
     ```json
     {"type": "command", "action": "search_and_grasp", "object": "red cup"}
     ```
6. Server:
   - Starts `_mock_search_and_grasp("red cup")`.
   - Emits:
     - Status: `"searching"`, then `"grasping"`, then `"done"`.
     - Reasoning messages describing each step.
   - Updates mock joints and synthetic scene over time.
7. UI:
   - Shows evolving frames.
   - Shows reasoning log as if a planner was operating.

### 5.2 Real SO100 + Policies (CLI)

1. Robot machine:
   - SO100 connected at the right serial port.
   - Wrist camera accessible via OpenCV index.
2. User runs:
   ```bash
   python3 -m so100_vla_demo.run_demo \
     --object-name "tennis ball" \
     --search-policy-path /path/to/search/pretrained_model \
     --grasp-policy-path /path/to/grasp/pretrained_model \
     --so100-port /dev/ttyUSB0 \
     --camera-index 0
   ```
3. `run_demo.py`:
   - Builds config, robot interface, search & grasp skills.
   - Runs `SO100DemoOrchestrator.run("tennis ball")`.
4. Orchestrator:
   - Connects robot.
   - Runs search policy until `detect_fn` says the ball is visible.
   - Runs grasp policy for the configured horizon.
   - Disconnects robot.

### 5.3 Future: Real SO100 + Browser + Policies

Once policies and detection are ready:

1. Start server in real mode:
   ```bash
   USE_MOCK_ROBOT=false \
   SO100_PORT=/dev/ttyUSB0 \
   SO100_CAMERA_INDEX=0 \
   uvicorn so100_vla_demo.server:app --host 0.0.0.0 --port 8000
   ```
2. UI stays the same (`/static/index.html`).
3. Extend server’s `"search_and_grasp"` handler to:
   - Use `SearchPolicySkill` to move until the object is detected.
   - Then use `GraspPolicySkill` to pick the object.
   - Emit rich reasoning/status messages for the UI.
4. Optionally, wire in the LLM via `chat()` and function calling so the LLM chooses when to call `search_and_grasp` vs. other skills.

---

## 6. What Remains for the Full System

- Implement real HTTP clients in `GeminiEngine`, `ClaudeEngine`, `QwenEngine`.
- Design and train:
  - `search_object` policy (active perception),
  - `grasp_object` policy (VLA / diffusion).
- Implement a real object detector:
  - Either VLM‑based (ask the model if the object is visible and where),
  - Or classical (color/shape) for simple objects.
- Replace `_mock_search_and_grasp` with a policy‑driven version that:
  - Uses `SearchPolicySkill` and `GraspPolicySkill`.
  - Supports both CLI and WebSocket triggering.
- Add safety constraints (workspace limits, timeouts, emergency stop).

This document, together with `ARCHITECTURE_HIGH_LEVEL.md` and `README.md`, should give your teammates everything they need to understand and extend the SO100 demo for the hackathon.+
