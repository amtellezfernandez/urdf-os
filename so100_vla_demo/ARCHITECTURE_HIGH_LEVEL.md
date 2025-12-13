# SO100 VLA Demo – High‑Level Architecture

This document gives the big‑picture view of the hackathon demo so new teammates can quickly understand **what the system does** and **how the main blocks connect**, without diving into implementation details.

---

## Core Idea

Demonstrate **hierarchical control** of the SO100 arm:

- A **vision‑language model (VLM/LLM)** reasons at a high level from camera images and user text.
- Learned **robot policies (VLA / diffusion / ACT)** implement low‑level motions as reusable **skills**:
  - `search_object` – active perception: move to reveal the object.
  - `grasp_object` – visuomotor manipulation: grasp once visible.
- A **simple web UI** shows:
  - Live camera feed,
  - LLM “thinking” / reasoning text,
  - Robot phase (idle / searching / grasping / done).

Everything can run either:

- With the real **SO100 arm**, or
- In a **mock mode** with a synthetic scene (for judges or remote teammates).

---

## Main Components

### 1. Web UI (Browser)

- File: `so100_vla_demo/static/index.html`
- Responsibilities:
  - Connect to the backend via WebSocket (`/ws`).
  - Display live camera frames from the robot (or mock).
  - Show current phase: idle / streaming / searching / grasping / done.
  - Display LLM/chat messages and reasoning logs.
  - Provide controls:
    - Connect / Disconnect,
    - Start / Stop camera stream,
    - High‑level commands: “Find Red Cup”, “Find Blue Block”, “Find Green Ball”,
    - Free‑form chat to the LLM.

### 2. Backend Server (FastAPI)

- File: `so100_vla_demo/server.py`
- Responsibilities:
  - Host the WebSocket endpoint `/ws`.
  - Serve the static web UI at `/static/index.html`.
  - Manage **global state**:
    - Robot interface (real or mock),
    - LLM engine (Gemini / Claude / Qwen / stub),
    - Background tasks (camera stream, behavior loops).
  - Translate WebSocket messages into actions:
    - `chat` → call LLM → send answer.
    - `start_stream` / `stop_stream` → start/stop camera streaming.
    - `search_and_grasp` → launch “search then grasp” behavior (mock or, later, real policies).

### 3. Robot Layer (Real vs Mock)

- Files:
  - `so100_vla_demo/config.py` – `SO100DemoConfig` (ports, camera index, mock flags).
  - `so100_vla_demo/robot_interface.py` – `SO100RobotInterface` wrapping LeRobot’s `SO100Follower`.
  - `so100_vla_demo/mock_robot_interface.py` – `MockRobotInterface` for synthetic scenes.
- Responsibilities:
  - Abstract away **how** we get images and send joint commands:
    - Real mode: use SO100 hardware (`SO100Follower`).
    - Mock mode: generate synthetic images and fake joints, no hardware.
  - Provide a **uniform API**:
    - `connect()`, `disconnect()`,
    - `get_observation() -> (image, joints_dict)`,
    - `send_joint_targets(joint_targets_dict)`.

### 4. Skills and Policies (Search / Grasp)

- Files:
  - `so100_vla_demo/search_skill.py` – `SearchPolicySkill`.
  - `so100_vla_demo/grasp_skill.py` – `GraspPolicySkill`.
  - `so100_vla_demo/demo_orchestrator.py` – `SO100DemoOrchestrator`.
  - `so100_vla_demo/run_demo.py` – CLI runner for real policies.
- Responsibilities:
  - Load trained **LeRobot policies** (SmolVLA / XVLA / diffusion / ACT).
  - Provide high‑level skills:
    - `search_object` – run search policy until object visible.
    - `grasp_object` – run grasp policy for a fixed horizon or until success.
  - Orchestrate “search then grasp” from a single call (CLI or server).

> For the web demo, the **mock search_and_grasp** sequence lives directly in `server.py` for now. Once policies are trained, the same pattern will be used there, but calling `SearchPolicySkill` and `GraspPolicySkill`.

### 5. LLM / VLM Layer

- Files:
  - `so100_vla_demo/llm_config.py` – lightweight config object.
  - `so100_vla_demo/llm_config.json` – default provider/model config.
  - `so100_vla_demo/llm_engine.py` – engine classes for Gemini / Claude / Qwen / Stub.
- Responsibilities:
  - Abstract away the choice of provider (Gemini / Claude / Qwen).
  - Provide a single async API:
    - `engine.chat(messages, tools=None) -> response`.
  - For now:
    - `StubEngine` is used by default for safe local testing.
    - Real engines are stubs for your team to complete with proper HTTP calls.

---

## Runtime Modes

### A. Browser Demo – Mock Robot (no hardware)

- Command:
  ```bash
  python3 -m so100_vla_demo.demo_script
  ```
- Flow:
  1. `demo_script.py` sets `USE_MOCK_ROBOT=true` and starts the server.
  2. User opens `http://localhost:8000/static/index.html`.
  3. UI connects to `/ws`, starts stream, shows synthetic scene.
  4. User clicks “Find Red Cup”:
     - UI sends `search_and_grasp` command.
     - Server runs a scripted search + grasp sequence in mock mode.
     - Reasoning/status messages are displayed in the UI.

### B. Browser Demo – Real SO100 (later)

- Command example:
  ```bash
  USE_MOCK_ROBOT=false \
  SO100_PORT=/dev/ttyUSB0 \
  SO100_CAMERA_INDEX=0 \
  uvicorn so100_vla_demo.server:app --host 0.0.0.0 --port 8000
  ```
- Same UI, but frames come from the real wrist camera.
- `search_and_grasp` will eventually call real learned policies instead of mock behavior.

### C. CLI Demo – Real SO100 + Policies

- Command:
  ```bash
  python3 -m so100_vla_demo.run_demo \
    --object-name "tennis ball" \
    --search-policy-path /path/to/search/pretrained_model \
    --grasp-policy-path /path/to/grasp/pretrained_model
  ```
- No web UI; runs “search then grasp” directly using policies and LeRobot.

---

## How This Fits Your Hackathon Story

- **User**: “Find the red cup.”
- **VLM/LLM**: decides to call `search_object` until the red cup appears, then `grasp_object`.
- **Search policy**: human‑like active perception learned from teleop data.
- **Grasp policy**: VLA or diffusion policy trained for manipulation.
- **UI**: shows camera, reasoning, and phases so judges can see the whole thought‑to‑action chain.+

