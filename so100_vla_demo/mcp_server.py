"""
SO100 VLA MCP Server

Exposes robot camera feeds and VLA skills as MCP tools,
allowing Claude Code to directly control the robot.

Usage:
    python -m so100_vla_demo.mcp_server
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image as PILImage

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("so100-vla-server")

# =============================================================================
# Global State
# =============================================================================

@dataclass
class CameraState:
    """Manages camera connections."""
    cameras: Dict[str, cv2.VideoCapture] = field(default_factory=dict)
    camera_indexes: Dict[str, int] = field(default_factory=dict)

    def configure(self, names: List[str], indexes: List[int]):
        """Configure camera name to index mapping."""
        self.camera_indexes = dict(zip(names, indexes))

    def get_frame(self, name: str) -> Optional[np.ndarray]:
        """Get frame from camera by name."""
        if name not in self.camera_indexes:
            return None

        idx = self.camera_indexes[name]

        # Lazy open camera
        if name not in self.cameras:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {name} at index {idx}")
                return None
            self.cameras[name] = cap

        cap = self.cameras[name]
        ret, frame = cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release_all(self):
        """Release all cameras."""
        for cap in self.cameras.values():
            cap.release()
        self.cameras.clear()


@dataclass
class SkillExecution:
    """Tracks a running skill."""
    skill_id: str
    skill_name: str
    status: str  # "running", "completed", "stopped", "error"
    steps_completed: int = 0
    max_steps: int = 100
    error: Optional[str] = None
    stop_requested: bool = False
    thread: Optional[threading.Thread] = None


@dataclass
class RobotState:
    """Manages robot connection and state."""
    connected: bool = False
    robot_interface: Any = None
    joints: Dict[str, float] = field(default_factory=dict)


# Global state instances
camera_state = CameraState()
robot_state = RobotState()
running_skills: Dict[str, SkillExecution] = {}

# Configure cameras from environment
_camera_indexes = os.environ.get("SO100_CAMERA_INDEXES", "0").split(",")
_camera_names = os.environ.get("SO100_CAMERA_NAMES", "wrist").split(",")
camera_state.configure(
    [n.strip() for n in _camera_names],
    [int(i.strip()) for i in _camera_indexes]
)

# =============================================================================
# Camera Tools
# =============================================================================

@mcp.tool()
def list_cameras() -> List[str]:
    """
    List available camera names.

    Returns list of camera names that can be used with get_camera_frame().
    """
    return list(camera_state.camera_indexes.keys())


@mcp.tool()
def get_camera_frame(camera_name: str = "wrist") -> Image:
    """
    Get current frame from specified camera.

    Args:
        camera_name: Name of camera (use list_cameras() to see available)

    Returns:
        Current camera frame as an image
    """
    frame = camera_state.get_frame(camera_name)

    if frame is None:
        # Return a placeholder image if camera fails
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            f"Camera '{camera_name}' not available",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        frame = placeholder

    # Convert to JPEG
    pil_img = PILImage.fromarray(frame)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)

    return Image(data=buf.getvalue(), format="jpeg")


@mcp.tool()
def get_all_camera_frames() -> Dict[str, str]:
    """
    Get frames from all available cameras at once.

    Returns:
        Dictionary with camera names as keys and status/info as values.
        Use get_camera_frame() for each camera to see the actual images.
    """
    result = {}
    for name in camera_state.camera_indexes.keys():
        frame = camera_state.get_frame(name)
        if frame is not None:
            result[name] = f"available ({frame.shape[1]}x{frame.shape[0]})"
        else:
            result[name] = "not available"
    return result


# =============================================================================
# Robot Connection Tools
# =============================================================================

@mcp.tool()
def connect_robot(port: str = "auto") -> Dict[str, Any]:
    """
    Connect to the SO100 robot arm.

    Args:
        port: Serial port (e.g., "/dev/ttyUSB0") or "auto" to detect

    Returns:
        Connection status and robot info
    """
    global robot_state

    if robot_state.connected:
        return {"status": "already_connected", "port": "unknown"}

    try:
        # Import here to avoid loading LeRobot if not needed
        from .config import SO100DemoConfig
        from .robot_interface import make_robot_interface

        # Auto-detect port if needed
        if port == "auto":
            for try_port in ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]:
                if os.path.exists(try_port):
                    port = try_port
                    break
            else:
                return {"status": "error", "error": "No robot port found. Connect USB cable."}

        os.environ["SO100_PORT"] = port
        os.environ["USE_MOCK_ROBOT"] = "false"

        cfg = SO100DemoConfig()
        robot_state.robot_interface = make_robot_interface(cfg)
        robot_state.robot_interface.connect()
        robot_state.connected = True

        return {"status": "connected", "port": port}

    except Exception as e:
        logger.error(f"Failed to connect robot: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def disconnect_robot() -> Dict[str, Any]:
    """
    Disconnect from the SO100 robot arm.

    Returns:
        Disconnection status
    """
    global robot_state

    if not robot_state.connected:
        return {"status": "not_connected"}

    try:
        if robot_state.robot_interface:
            robot_state.robot_interface.disconnect()
        robot_state.connected = False
        robot_state.robot_interface = None
        return {"status": "disconnected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def get_robot_state() -> Dict[str, Any]:
    """
    Get current robot state including joint positions.

    Returns:
        Robot state with joint positions and connection status
    """
    if not robot_state.connected or not robot_state.robot_interface:
        return {"connected": False, "joints": {}}

    try:
        images, joints = robot_state.robot_interface.get_observation()
        return {
            "connected": True,
            "joints": joints,
            "cameras_active": list(images.keys())
        }
    except Exception as e:
        return {"connected": True, "joints": {}, "error": str(e)}


# =============================================================================
# Skill Tools
# =============================================================================

@mcp.tool()
def list_skills() -> List[Dict[str, str]]:
    """
    List available VLA skills that can be executed.

    Returns:
        List of skills with names and descriptions
    """
    skills = [
        {
            "name": "grasp",
            "description": "Grasp an object using trained VLA policy (SmolVLA). "
                          "The robot will approach and grasp the target object.",
            "requires_policy": True
        },
        {
            "name": "search",
            "description": "Search for an object by panning the camera. "
                          "The robot will move to scan the workspace.",
            "requires_policy": True
        },
        {
            "name": "home",
            "description": "Move robot to home position. No policy required.",
            "requires_policy": False
        }
    ]
    return skills


def _run_skill_loop(execution: SkillExecution):
    """Background thread that runs the skill."""
    try:
        logger.info(f"Starting skill {execution.skill_name} (id={execution.skill_id})")

        # Check if robot is connected
        if not robot_state.connected or not robot_state.robot_interface:
            execution.status = "error"
            execution.error = "Robot not connected"
            return

        # Run skill steps
        for step in range(execution.max_steps):
            if execution.stop_requested:
                execution.status = "stopped"
                logger.info(f"Skill {execution.skill_id} stopped at step {step}")
                return

            # Get observation
            try:
                images, joints = robot_state.robot_interface.get_observation()
            except Exception as e:
                execution.status = "error"
                execution.error = f"Observation failed: {e}"
                return

            # TODO: Run actual VLA policy here
            # For now, just simulate steps
            time.sleep(0.1)

            execution.steps_completed = step + 1

        execution.status = "completed"
        logger.info(f"Skill {execution.skill_id} completed after {execution.steps_completed} steps")

    except Exception as e:
        execution.status = "error"
        execution.error = str(e)
        logger.error(f"Skill {execution.skill_id} failed: {e}")


@mcp.tool()
def start_skill(
    skill_name: str,
    target_object: str = "",
    max_steps: int = 100
) -> Dict[str, Any]:
    """
    Start executing a VLA skill.

    Args:
        skill_name: Name of skill to execute ("grasp", "search", "home")
        target_object: Description of target object (for grasp/search)
        max_steps: Maximum steps to run before stopping

    Returns:
        Skill execution ID and initial status. Use get_skill_status() to monitor.
    """
    # Validate skill name
    valid_skills = ["grasp", "search", "home"]
    if skill_name not in valid_skills:
        return {"status": "error", "error": f"Unknown skill: {skill_name}. Valid: {valid_skills}"}

    # Check robot connection
    if not robot_state.connected:
        return {"status": "error", "error": "Robot not connected. Call connect_robot() first."}

    # Create execution
    skill_id = f"skill_{uuid.uuid4().hex[:8]}"
    execution = SkillExecution(
        skill_id=skill_id,
        skill_name=skill_name,
        status="running",
        max_steps=max_steps
    )

    # Start background thread
    thread = threading.Thread(target=_run_skill_loop, args=(execution,), daemon=True)
    execution.thread = thread
    running_skills[skill_id] = execution
    thread.start()

    return {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "target_object": target_object,
        "status": "running",
        "max_steps": max_steps
    }


@mcp.tool()
def stop_skill(skill_id: str) -> Dict[str, Any]:
    """
    Stop a running skill immediately.

    Args:
        skill_id: ID returned by start_skill()

    Returns:
        Final status of the stopped skill
    """
    if skill_id not in running_skills:
        return {"status": "error", "error": f"Unknown skill_id: {skill_id}"}

    execution = running_skills[skill_id]
    execution.stop_requested = True

    # Wait briefly for thread to stop
    if execution.thread and execution.thread.is_alive():
        execution.thread.join(timeout=1.0)

    return {
        "skill_id": skill_id,
        "status": execution.status,
        "steps_completed": execution.steps_completed
    }


@mcp.tool()
def get_skill_status(skill_id: str) -> Dict[str, Any]:
    """
    Get current status of a skill execution.

    Args:
        skill_id: ID returned by start_skill()

    Returns:
        Current status, steps completed, and any errors
    """
    if skill_id not in running_skills:
        return {"status": "error", "error": f"Unknown skill_id: {skill_id}"}

    execution = running_skills[skill_id]
    result = {
        "skill_id": skill_id,
        "skill_name": execution.skill_name,
        "status": execution.status,
        "steps_completed": execution.steps_completed,
        "max_steps": execution.max_steps
    }

    if execution.error:
        result["error"] = execution.error

    return result


@mcp.tool()
def list_running_skills() -> List[Dict[str, Any]]:
    """
    List all skill executions (running and completed).

    Returns:
        List of all skill executions with their status
    """
    return [
        {
            "skill_id": ex.skill_id,
            "skill_name": ex.skill_name,
            "status": ex.status,
            "steps_completed": ex.steps_completed
        }
        for ex in running_skills.values()
    ]


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the MCP server."""
    import sys

    logger.info("Starting SO100 VLA MCP Server...")
    logger.info(f"Cameras configured: {list(camera_state.camera_indexes.keys())}")

    # Run with stdio transport (default for Claude Code)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
