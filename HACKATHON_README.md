# LeRobot Hackathon – SO100 Hierarchical VLM + VLA Demo

This document is for the **robot team**. It explains **what this repo does**, **how to run the demo**, and **what you need to do next** on the real SO100 arm.

You do **not** need to understand every detail of LeRobot or VLAs to use this. The system is split into clear blocks.

---

## 1. Goal of the Demo

End‑to‑end story:

- User says: **“Find the red cup and grasp it”**.
- A **VLM/LLM** reasons at high level (“I don’t see it yet → search → now I see it → grasp”).
- One or more **robot policies** (VLA / diffusion / ACT) execute low‑level motions:
  - `search_object` – move the camera/arm to reveal the object.
  - `grasp_object` – close the loop and pick it up once visible.
- A **web UI** shows:
  - Live camera stream,
  - Reasoning text,
  - Current phase: idle / searching / grasping / done.

Everything in this repo is already wired; you only need to:

1. Connect it to the real SO100 (instead of mock robot).
2. Train and plug in real policies and detectors.

---

## 2. Where the Demo Lives (Important Paths)

All hackathon‑specific code is under:

- `so100_vla_demo/`

Key files for you:

- `so100_vla_demo/server.py` – FastAPI backend + WebSocket.
- `so100_vla_demo/static/index.html` – Web UI (single HTML file).
- `so100_vla_demo/config.py` – Demo config (ports, camera, mock/real).
- `so100_vla_demo/robot_interface.py` – Real SO100 interface.
- `so100_vla_demo/mock_robot_interface.py` – Mock robot (no hardware).
- `so100_vla_demo/search_skill.py` – Search policy wrapper (to be trained).
- `so100_vla_demo/grasp_skill.py` – Grasp policy wrapper (to be trained).
- `so100_vla_demo/demo_orchestrator.py` – “search then grasp” orchestrator.
- `so100_vla_demo/run_demo.py` – CLI runner for real policies.
- `so100_vla_demo/demo_script.py` – Demo launcher (mock by default).

Architecture docs:

- `so100_vla_demo/ARCHITECTURE_HIGH_LEVEL.md` – Big‑picture explanation.
- `so100_vla_demo/ARCHITECTURE_FULL_SYSTEM.md` – Detailed module‑by‑module view.

If you only read three files, read:

1. This `HACKATHON_README.md`.
2. `so100_vla_demo/ARCHITECTURE_HIGH_LEVEL.md`.
3. `so100_vla_demo/README.md`.

---

## 3. Two Modes: Mock vs Real Robot

The system can run **with** or **without** the SO100 arm.

### 3.1 Mock Mode (no hardware, safe)

- Uses `MockRobotInterface` instead of the real robot.
- Camera frames are synthetic or from a video file.
- Joint states are fake; no motors move.
- Great for:
  - Testing on any laptop / VM.
  - Letting judges run the demo without hardware.
  - Debugging the UI and logic safely.

### 3.2 Real Mode (SO100 hardware)

- Uses `SO100RobotInterface` which wraps LeRobot’s `SO100Follower`.
- Camera frames come from the real wrist camera.
- Joint targets are sent to the real arm.
- This is what you’ll use on the physical robot once everything is ready.

### 3.3 How we switch modes

We use the `USE_MOCK_ROBOT` environment variable:

- `USE_MOCK_ROBOT=true` → mock robot.
- `USE_MOCK_ROBOT=false` (or unset) → real SO100.

The server calls:

- `make_robot_interface(cfg: SO100DemoConfig)` in `robot_interface.py` to choose mock vs real.

---

## 4. Quick Start: Mock Demo (Recommended First)

You can run the **full demo with UI and search/grasp phases** without any robot.

From the repo root (`so100_vla_hackathon`):

```bash
cd /home/charbel/lerobot/so100_vla_hackathon
python3 -m so100_vla_demo.demo_script
```

What this does:

- Sets `USE_MOCK_ROBOT=true` automatically.
- Starts the FastAPI server on `http://localhost:8000`.
- Serves the web UI at `http://localhost:8000/static/index.html`.

Then in your browser:

1. Open: `http://localhost:8000/static/index.html`.
2. Click **Connect**.
3. Click **Start Stream** → you see a synthetic scene (table + colored objects).
4. Click **Find Red Cup** / **Find Blue Block** / **Find Green Ball**.

You will see:

- Live “camera” images (synthetic).
- Status: idle → searching → grasping → done.
- Reasoning log: text describing what the system is doing (mock behavior).

No real robot is touched in this mode.

---

## 5. Real Robot: How the Team Will Use This

Once you are ready to use the SO100 arm:

### 5.1 Start the server in real mode

On the robot machine:

```bash
cd /home/charbel/lerobot/so100_vla_hackathon
export USE_MOCK_ROBOT=false
export SO100_PORT=/dev/ttyUSB0        # adjust for your setup
export SO100_CAMERA_INDEX=0           # OpenCV index for wrist camera

uvicorn so100_vla_demo.server:app --host 0.0.0.0 --port 8000
```

Then on your laptop (same network), open:

- `http://robot-ip:8000/static/index.html`

You will see the **real wrist camera feed** instead of synthetic images.

⚠ **Safety**: at this stage, `search_and_grasp` is still implemented only in **mock** mode, so it will not move the real robot. That is intentional for safety. Motion only happens when you explicitly wire in real policies.

### 5.2 CLI “search then grasp” (when policies are ready)

When you later have trained policies:

```bash
python3 -m so100_vla_demo.run_demo \
  --object-name "tennis ball" \
  --search-policy-path /path/to/search/pretrained_model \
  --grasp-policy-path /path/to/grasp/pretrained_model \
  --so100-port /dev/ttyUSB0 \
  --camera-index 0
```

This runs the **pure Python** orchestrator (`demo_orchestrator.py`) and uses:

- `SearchPolicySkill` to move until the object is detected.
- `GraspPolicySkill` to grasp the object.

You can test this before connecting it to the web UI.

---

## 6. What the Robot Team Needs to Do Next

This repo already gives you:

- A working **mock demo** with UI and behavior phases.
- A **real robot interface** ready to use with SO100.
- Clean abstractions for **search** and **grasp** skills.
- Documentation of the architecture.

Your main tasks:

### 6.1 Verify SO100 + Camera with LeRobot

Using LeRobot directly (outside this demo), confirm:

- The arm connects on the given serial port.
- The wrist camera index is correct and returns frames.
- Basic teleoperation works.

This step ensures hardware is OK before using the hackathon demo.

### 6.2 Record Datasets (Search + Grasp)

On the robot machine, using `lerobot-record`:

1. **Search dataset** (e.g. `my_so100_search_for_ball`):
   - Start with object hidden/partially occluded.
   - Teleoperate to **search until object is visible and centered**.
   - End episode when the object is clearly visible.

2. **Grasp dataset** (e.g. `my_so100_grasp_ball`):
   - Start with object visible.
   - Teleoperate to **grasp and lift** the object.

Save datasets locally (no push to hub). Later copy them to the AMD GPU VM.

### 6.3 Train Policies on the AMD VM

On the MI300X VM, use `lerobot-train`:

- **Search policy**:
  - Learn `(image + state) → Δpose` for search behavior.
- **Grasp policy**:
  - Learn a VLA/diffusion policy for the grasp.

The training commands and hyperparameters can follow LeRobot examples. Each run produces a `pretrained_model` directory.

### 6.4 Plug Policies into this Demo

Once you have `pretrained_model` folders:

1. Copy them to the SO100 machine.
2. Update:
   - `so100_vla_demo/search_skill.py`
   - `so100_vla_demo/grasp_skill.py`
   so that the observation keys and shapes match your training.
3. Use `run_demo.py` to test the policies.
4. Later, extend `server.py`’s `"search_and_grasp"` command to call these skills in **real mode**.

### 6.5 (Optional) Connect a Real VLM (Gemini / Claude)

Right now the LLM is a **stub** (no real API calls).

To use a real VLM:

1. Put your provider/model in:
   - `so100_vla_demo/llm_config.json`
2. Implement `chat()` in the corresponding engine in:
   - `so100_vla_demo/llm_engine.py`
3. Export your API key (e.g. `GEMINI_API_KEY`).

The UI already supports chat; only the backend call needs to be filled in.

---

## 7. Where to Look in the Code (Cheat Sheet)

| Task / Question                            | File(s) to open                           |
|-------------------------------------------|-------------------------------------------|
| Understand overall design                 | `so100_vla_demo/ARCHITECTURE_HIGH_LEVEL.md` |
| See full technical details                | `so100_vla_demo/ARCHITECTURE_FULL_SYSTEM.md` |
| Change robot ports / camera index         | `so100_vla_demo/config.py`                |
| Real vs mock robot selection              | `so100_vla_demo/config.py`, `robot_interface.py`, `mock_robot_interface.py` |
| Web server + WebSocket                    | `so100_vla_demo/server.py`                |
| Web UI layout / buttons / video           | `so100_vla_demo/static/index.html`        |
| How search policy is called               | `so100_vla_demo/search_skill.py`          |
| How grasp policy is called                | `so100_vla_demo/grasp_skill.py`           |
| High‑level “search then grasp” logic      | `so100_vla_demo/demo_orchestrator.py`     |
| CLI demo runner                           | `so100_vla_demo/run_demo.py`              |
| Demo launcher (mock mode, web UI)         | `so100_vla_demo/demo_script.py`           |

---

## 8. Hackathon Pitch Points (for your presentation)

- **Hierarchical AI**: VLM reasons about the scene, VLA executes precise motions.
- **Active perception**: The robot can **search for non‑visible objects**, not just react to what is already in view.
- **Clean UI**: Judges see camera, phases, and reasoning text in real time.
- **Works without hardware**: Mock mode lets anyone run the demo.
- **Modular**: Easy to swap VLM provider, policies, or add new skills.

This README plus the architecture docs should be enough for any team member to understand the system and continue the work on the real SO100 arm.+

