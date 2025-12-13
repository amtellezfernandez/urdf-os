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
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import SO100DemoConfig
from .llm_config import LLMConfig
from .llm_engine import BaseLLMEngine, StubEngine, make_llm_engine
from .mock_robot_interface import MockRobotInterface
from .robot_interface import SO100RobotInterface, make_robot_interface

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


# Static files (frontend) -----------------------------------------------------

static_dir = Path(__file__).with_name("static")
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
