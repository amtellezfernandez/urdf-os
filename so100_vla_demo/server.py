from __future__ import annotations

"""
Minimal FastAPI WebSocket server for the SO100 VLA demo.

Features:
- Streams camera frames from SO100RobotInterface to connected clients.
- Accepts simple chat messages and answers via a pluggable LLM engine.

Usage (backend only):

    uvicorn so100_vla_demo.server:app --host 0.0.0.0 --port 8000

Then connect with a WebSocket client to:
    ws://localhost:8000/ws

Messages:
- From client:
    {"type": "chat", "text": "Hello, what do you see?"}
  or
    {"type": "command", "action": "start_stream"}

- From server:
    {"type": "chat", "text": "..."}         # LLM reply (stub by default)
    {"type": "frame", "shape": [H, W, C]}   # metadata about a frame
    (binary messages with raw JPEG bytes can be added later)
"""

import asyncio
import base64
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Literal, Set

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from .config import SO100DemoConfig
from .llm_config import LLMConfig
from .llm_engine import BaseLLMEngine, StubEngine, make_llm_engine
from .mock_robot_interface import MockRobotInterface
from .robot_interface import SO100RobotInterface, make_robot_interface
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

logger = logging.getLogger(__name__)


app = FastAPI(title="SO100 VLA Demo Server")

# CORS for frontend debugging (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info("Client connected (total=%d)", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info("Client disconnected (total=%d)", len(self.active_connections))

    async def broadcast_json(self, message: Dict[str, Any]) -> None:
        async with self._lock:
            disconnect: Set[WebSocket] = set()
            for ws in self.active_connections:
                try:
                    await ws.send_json(message)
                except Exception as e:  # noqa: BLE001
                    logger.error("Error broadcasting JSON: %s", e)
                    disconnect.add(ws)
            for ws in disconnect:
                self.active_connections.discard(ws)


manager = ConnectionManager()

# Global demo objects (lazy init)
demo_cfg = SO100DemoConfig()

# LLM configuration can be specified via a JSON file or env vars.
_llm_config_path = Path(__file__).with_name("llm_config.json")
if _llm_config_path.is_file():
    llm_cfg = LLMConfig.load(str(_llm_config_path))
else:
    llm_cfg = LLMConfig.load()
llm_engine: BaseLLMEngine = make_llm_engine(llm_cfg)

# Robot interface: real SO100 or mock, depending on config.
robot_interface = make_robot_interface(demo_cfg)

# Control flags
streaming = False
stream_task: asyncio.Task | None = None
behavior_task: asyncio.Task | None = None
# Teleop process state
teleop_process: asyncio.subprocess.Process | None = None
teleop_process_cmd: list[str] = []
teleop_process_started_at: float | None = None
teleop_process_lock = asyncio.Lock()

# Device config / calibration discovery
DEVICE_PORTS_FILE = HF_LEROBOT_CALIBRATION / "device_ports.json"


class DevicePortUpdate(BaseModel):
    role: Literal["teleop", "robot"]
    type: str
    id: str
    port: str


class TeleopStartRequest(BaseModel):
    teleop_id: str
    robot_id: str
    teleop_port: str | None = None
    robot_port: str | None = None
    teleop_type: str = "so100_leader"
    robot_type: str = "so100_follower"
    fps: int = 60
    display_data: bool = False


def _load_device_ports() -> Dict[str, Dict[str, Dict[str, str]]]:
    if DEVICE_PORTS_FILE.is_file():
        try:
            return json.loads(DEVICE_PORTS_FILE.read_text())
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not read device ports file %s: %s", DEVICE_PORTS_FILE, e)
    return {"teleop": {}, "robot": {}}


def _save_device_ports(data: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    DEVICE_PORTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_PORTS_FILE.write_text(json.dumps(data, indent=2))


def _discover_devices() -> list[Dict[str, Any]]:
    """
    Look for calibration files written by `lerobot-calibrate` and merge with
    locally stored port hints so the UI can show `id -> port` mappings.
    """
    devices: list[Dict[str, Any]] = []
    port_map = _load_device_ports()

    def _role_from_dir(role_dir: str) -> Literal["teleop", "robot"] | None:
        if role_dir == "teleoperators":
            return "teleop"
        if role_dir == "robots":
            return "robot"
        return None

    if not HF_LEROBOT_CALIBRATION.exists():
        return devices

    for role_dir in ("teleoperators", "robots"):
        role_root = HF_LEROBOT_CALIBRATION / role_dir
        role_key = _role_from_dir(role_dir)
        if role_key is None or not role_root.is_dir():
            continue
        for type_dir in role_root.iterdir():
            if not type_dir.is_dir():
                continue
            for calib_file in type_dir.glob("*.json"):
                device_id = calib_file.stem
                devices.append(
                    {
                        "role": role_key,
                        "type": type_dir.name,
                        "id": device_id,
                        "calibration_path": str(calib_file),
                        "calibration_mtime": os.path.getmtime(calib_file),
                        "port": port_map.get(role_key, {}).get(type_dir.name, {}).get(device_id),
                    }
                )
    devices.sort(key=lambda d: (d["role"], d["type"], d["id"]))
    return devices


async def _stop_teleop_process(force: bool = False) -> None:
    """
    Stop the background lerobot-teleoperate process if running.
    """
    global teleop_process, teleop_process_cmd, teleop_process_started_at
    if teleop_process is None:
        return
    if teleop_process.returncode is not None:
        teleop_process = None
        teleop_process_cmd = []
        teleop_process_started_at = None
        return

    teleop_process.terminate()
    try:
        await asyncio.wait_for(teleop_process.wait(), timeout=5 if not force else 1)
    except asyncio.TimeoutError:
        teleop_process.kill()
    teleop_process = None
    teleop_process_cmd = []
    teleop_process_started_at = None


async def _pipe_subprocess_output(proc: asyncio.subprocess.Process, name: str) -> None:
    """
    Drain stdout/stderr of a subprocess to avoid blocking and emit to logger.
    """
    assert proc.stdout is not None and proc.stderr is not None

    async def _reader(stream: asyncio.StreamReader, log_fn) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                decoded = line.decode("utf-8", errors="replace").rstrip()
            except Exception:
                decoded = str(line)
            log_fn("%s | %s", name, decoded)

    await asyncio.gather(
        _reader(proc.stdout, logger.info),
        _reader(proc.stderr, logger.error),
    )


async def camera_stream_loop() -> None:
    """
    Background task: grab frames from SO100 and broadcast metadata.

    For debugging we only send JSON with the frame shape and a small
    JPEG thumbnail as base64-like bytes length. You can extend this to
    send actual JPEG bytes or use a separate binary WebSocket channel.
    """
    global streaming
    logger.info("Camera stream loop started.")
    # Connect robot lazily here to avoid blocking app startup if arm is offline
    try:
        # Both the real SO100RobotInterface and MockRobotInterface expose
        # a connect() method. We lazily connect here for streaming.
        robot_interface.connect()
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to connect robot for streaming: %s", e)
        streaming = False
        return

    try:
        while streaming:
            frame, joints = robot_interface.get_observation()
            h, w, c = frame.shape

            # Build a small thumbnail JPEG for debugging and send as base64
            pil_img = Image.fromarray(frame)
            pil_img.thumbnail((320, 240))
            buf = BytesIO()
            pil_img.save(buf, format="JPEG")
            jpeg_bytes = buf.getvalue()
            jpeg_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

            payload = {
                "type": "frame",
                "shape": [h, w, c],
                "image_b64": jpeg_b64,
                "joints": joints,
            }
            await manager.broadcast_json(payload)

            await asyncio.sleep(1.0 / demo_cfg.demo_fps)
    finally:
        logger.info("Camera stream loop stopped.")


def _is_mock_robot() -> bool:
    return isinstance(robot_interface, MockRobotInterface)


async def _mock_search_and_grasp(object_name: str) -> None:
    """
    Simple mock behavior loop used in mock mode when the client sends
    a `search_and_grasp` command.

    This does not require any trained policy. It simulates:
    - a search phase (several steps with status + reasoning messages),
    - followed by a grasp phase.
    """

    await manager.broadcast_json({"type": "status", "phase": "searching"})
    await manager.broadcast_json(
        {
            "type": "reasoning",
            "thought": f"Starting search for '{object_name}' using mock policy.",
        }
    )

    # Search phase: we simply run for a few steps and then "find" the object.
    for step in range(10):
        try:
            frame, joints = robot_interface.get_observation()
        except Exception as e:  # noqa: BLE001
            logger.error("Error getting observation during mock search: %s", e)
            await manager.broadcast_json(
                {"type": "status", "phase": "error", "detail": "observation_failed"}
            )
            return

        # Fake joint-space scanning pattern in mock mode.
        if isinstance(joints, dict):
            new_joints = {}
            for idx, (name, val) in enumerate(joints.items()):
                delta = 0.1 * np.sin(step / 3.0 + idx)
                new_joints[name] = float(val + delta)
            robot_interface.send_joint_targets(new_joints)

        await manager.broadcast_json(
            {
                "type": "reasoning",
                "thought": f"[search step {step}] Panning camera to look for the object...",
            }
        )
        await asyncio.sleep(0.2)

    await manager.broadcast_json(
        {
            "type": "reasoning",
            "thought": f"Object '{object_name}' appears to be visible. Switching to grasp phase.",
        }
    )
    await manager.broadcast_json({"type": "status", "phase": "grasping"})

    # Grasp phase: simple scripted sequence.
    for step in range(5):
        await manager.broadcast_json(
            {
                "type": "reasoning",
                "thought": f"[grasp step {step}] Moving end-effector to grasp the object...",
            }
        )
        await asyncio.sleep(0.3)

    await manager.broadcast_json(
        {
            "type": "reasoning",
            "thought": f"Grasp completed in mock mode for '{object_name}'.",
        }
    )
    await manager.broadcast_json({"type": "status", "phase": "done"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Main WebSocket endpoint.

    - Receives JSON messages from the client:
        {"type": "chat", "text": "..."}
        {"type": "command", "action": "start_stream" | "stop_stream"}
    - Sends back:
        {"type": "chat", "text": "..."} replies
        {"type": "frame", ...} frame metadata from camera_stream_loop
    """
    global streaming, stream_task, behavior_task, llm_engine

    await manager.connect(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "text": "Invalid JSON"})
                continue

            mtype = data.get("type")

            if mtype == "chat":
                text = str(data.get("text", ""))
                # For debugging: call the stub engine (or real LLM if configured)
                try:
                    reply = await llm_engine.chat(
                        messages=[{"role": "user", "content": text}],
                        tools=None,
                    )
                    await websocket.send_json({"type": "chat", "text": reply.get("content", "")})
                except NotImplementedError:
                    # If a real engine is not implemented, fall back to StubEngine
                    stub = StubEngine()
                    reply = await stub.chat(
                        messages=[{"role": "user", "content": text}],
                        tools=None,
                    )
                    await websocket.send_json({"type": "chat", "text": reply.get("content", "")})

            elif mtype == "command":
                action = data.get("action")
                if action == "start_stream":
                    if not streaming:
                        streaming = True
                        stream_task = asyncio.create_task(camera_stream_loop())
                    await websocket.send_json(
                        {"type": "status", "text": "streaming_started", "phase": "streaming"}
                    )
                elif action == "stop_stream":
                    streaming = False
                    if stream_task is not None:
                        stream_task.cancel()
                        stream_task = None
                    await websocket.send_json(
                        {"type": "status", "text": "streaming_stopped", "phase": "idle"}
                    )
                elif action == "search_and_grasp":
                    object_name = str(data.get("object", "object")).strip() or "object"
                    # Cancel any previous behavior.
                    if behavior_task is not None and not behavior_task.done():
                        behavior_task.cancel()
                    if _is_mock_robot():
                        behavior_task = asyncio.create_task(
                            _mock_search_and_grasp(object_name)
                        )
                        await websocket.send_json(
                            {
                                "type": "status",
                                "text": f"search_and_grasp started for '{object_name}' (mock mode)",
                                "phase": "searching",
                            }
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "status",
                                "text": "search_and_grasp is only implemented in mock mode for now.",
                                "phase": "idle",
                            }
                        )
                else:
                    await websocket.send_json(
                        {"type": "error", "text": f"Unknown command action: {action!r}"}
                    )
            else:
                await websocket.send_json(
                    {"type": "error", "text": f"Unknown message type: {mtype!r}"}
                )
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:  # noqa: BLE001
        logger.error("WebSocket error: %s", e)
        await manager.disconnect(websocket)


# REST endpoints -------------------------------------------------------------


@app.get("/api/device-config")
async def get_device_config() -> Dict[str, Any]:
    """
    Expose the calibration root and discovered devices (ids + optional port hints).
    """
    return {
        "calibration_root": str(HF_LEROBOT_CALIBRATION),
        "ports_file": str(DEVICE_PORTS_FILE),
        "devices": _discover_devices(),
    }


@app.post("/api/device-config")
async def set_device_port(update: DevicePortUpdate) -> Dict[str, Any]:
    """
    Persist a device -> port mapping so the UI can reuse it for teleop.
    """
    data = _load_device_ports()
    data.setdefault(update.role, {}).setdefault(update.type, {})[update.id] = update.port
    _save_device_ports(data)
    return {"ok": True, "device": update.dict()}


@app.get("/api/teleop/status")
async def teleop_status() -> Dict[str, Any]:
    running = teleop_process is not None and teleop_process.returncode is None
    return {
        "running": running,
        "pid": teleop_process.pid if teleop_process else None,
        "returncode": teleop_process.returncode if teleop_process else None,
        "cmd": teleop_process_cmd,
        "started_at": teleop_process_started_at,
    }


@app.post("/api/teleop/start")
async def start_teleop(req: TeleopStartRequest) -> Dict[str, Any]:
    """
    Launch `lerobot-teleoperate` in the background using stored port hints or
    values provided in the request body.
    """
    global teleop_process, teleop_process_cmd, teleop_process_started_at
    async with teleop_process_lock:
        if teleop_process is not None and teleop_process.returncode is None:
            raise HTTPException(status_code=400, detail="Teleop already running.")

        stored_ports = _load_device_ports()
        teleop_port = req.teleop_port or stored_ports.get("teleop", {}).get(req.teleop_type, {}).get(req.teleop_id)
        robot_port = req.robot_port or stored_ports.get("robot", {}).get(req.robot_type, {}).get(req.robot_id)
        if not teleop_port:
            raise HTTPException(status_code=400, detail="teleop_port is required (not found in device ports store).")
        if not robot_port:
            raise HTTPException(status_code=400, detail="robot_port is required (not found in device ports store).")

        cmd = [
            "lerobot-teleoperate",
            f"--teleop.type={req.teleop_type}",
            f"--teleop.port={teleop_port}",
            f"--teleop.id={req.teleop_id}",
            f"--robot.type={req.robot_type}",
            f"--robot.port={robot_port}",
            f"--robot.id={req.robot_id}",
            f"--fps={req.fps}",
            f"--display_data={'true' if req.display_data else 'false'}",
        ]

        teleop_process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        teleop_process_cmd = cmd
        teleop_process_started_at = time.time()
        asyncio.create_task(_pipe_subprocess_output(teleop_process, "teleop"))
        logger.info("Started teleop process pid=%s cmd=%s", teleop_process.pid, cmd)
        return {"ok": True, "pid": teleop_process.pid, "cmd": cmd}


@app.post("/api/teleop/stop")
async def stop_teleop() -> Dict[str, Any]:
    async with teleop_process_lock:
        await _stop_teleop_process()
    return {"ok": True}


# Static files (frontend) -----------------------------------------------------

static_dir = Path(__file__).with_name("static")
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
