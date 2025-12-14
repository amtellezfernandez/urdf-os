"""
SO100 VLA MCP Server

Exposes robot camera feeds and VLA skills as MCP tools,
allowing Claude Code to directly control the robot.

Usage:
    python -m so100_vla_demo.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image as PILImage

from fastmcp import FastMCP
from fastmcp.utilities.types import Image
import torch
import grp
import pwd

from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_pre_post_processors

from .config import SO100DemoConfig, _parse_camera_sources, _parse_camera_names
from .robot_interface import make_robot_interface
from .search_skill import SearchPolicySkill  # noqa: F401
from .grasp_skill import GraspPolicySkill  # noqa: F401


def _as_pretrained_name_or_path(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_motor_names_from_robot_interface(robot_interface) -> list[str]:
    robot = getattr(robot_interface, "robot", None)
    if robot is None:
        # MockRobotInterface case (or other wrappers): fall back to SO100 motor names.
        return [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    bus = getattr(robot, "bus", None)
    motors = getattr(bus, "motors", None)
    if isinstance(motors, dict):
        return list(motors.keys())
    return []


def _clamp_so100_action(motor: str, value: float) -> float:
    if motor == "gripper":
        return float(max(0.0, min(100.0, value)))
    return float(max(-100.0, min(100.0, value)))


def _hwc_rgb_to_chw(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image from HWC uint8 to CHW uint8.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got shape={getattr(image, 'shape', None)}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.transpose(image, (2, 0, 1))


@dataclass
class VLAPolicyRunner:
    """
    Minimal runtime wrapper for a LeRobot VLA policy (SmolVLA, XVLA, ACT, etc.).

    Requires a LeRobot-exported checkpoint that includes:
    - `config.json`
    - `model.safetensors`
    - ideally `policy_preprocessor.json` + `policy_postprocessor.json` and their stats files
    """

    policy_id: str
    policy: PreTrainedPolicy | None = None
    preprocessor: Any | None = None
    postprocessor: Any | None = None

    def load(self) -> None:
        cfg = PreTrainedConfig.from_pretrained(self.policy_id)
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(self.policy_id, config=cfg)
        try:
            device = getattr(self.policy.config, "device", "cpu")
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.policy.config,
                pretrained_path=self.policy_id,
                preprocessor_overrides={"device_processor": {"device": device}},
                postprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Checkpoint '{self.policy_id}' is missing processor files (policy_preprocessor.json / "
                f"policy_postprocessor.json) or needs processor overrides (device mismatch). Underlying error: {e}"
            ) from e

    def ensure_loaded(self) -> None:
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            self.load()

    def step(
        self,
        *,
        images: Dict[str, np.ndarray],
        joints: Dict[str, float],
        motor_names: list[str],
        instruction: str,
    ) -> Dict[str, float]:
        self.ensure_loaded()
        assert self.policy is not None and self.preprocessor is not None and self.postprocessor is not None

        state = np.asarray([float(joints.get(m, 0.0)) for m in motor_names], dtype=np.float32)

        # Pick a default image for fallbacks (HWC RGB).
        default_img = None
        if "wrist" in images:
            default_img = images["wrist"]
        elif images:
            default_img = images[next(iter(images.keys()))]
        if default_img is None:
            raise RuntimeError("No camera images provided")

        default_img_chw = _hwc_rgb_to_chw(default_img)

        raw_batch: Dict[str, Any] = {"task": instruction, OBS_STATE: state}

        # Provide images matching the policy's expected input feature keys.
        # SmolVLA often expects `observation.images.<name>` rather than `observation.image`.
        expected_inputs = getattr(self.policy.config, "input_features", {}) or {}
        for key in expected_inputs.keys():
            if key == OBS_IMAGE:
                raw_batch[OBS_IMAGE] = default_img_chw
            elif key.startswith(f"{OBS_IMAGES}."):
                cam = key.split(".", 2)[2]  # observation.images.<cam>
                cam_map = _parse_policy_camera_map()
                mapped = cam_map.get(cam, cam)

                # Best-effort aliasing between common camera names.
                candidates = [mapped, cam]
                if cam in {"front", "top"} or mapped in {"front", "top"}:
                    candidates.extend(["overhead", "side", "wrist"])
                elif cam in {"overhead"} or mapped in {"overhead"}:
                    candidates.extend(["front", "top"])
                elif cam in {"agentview"} or mapped in {"agentview"}:
                    candidates.extend(["front", "overhead"])

                selected = None
                for c in candidates:
                    if c in images:
                        selected = images[c]
                        break
                raw_batch[key] = _hwc_rgb_to_chw(selected) if selected is not None else default_img_chw

        # Backward-compat: if the config didn't list an image key, still provide one.
        raw_batch.setdefault(OBS_IMAGE, default_img_chw)

        processed_batch = self.preprocessor(raw_batch)
        with torch.no_grad():
            action: torch.Tensor = self.policy.select_action(processed_batch)
        action = self.postprocessor(action)

        action_vec = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if len(action_vec) < len(motor_names):
            raise RuntimeError(f"Action dim {len(action_vec)} < motors {len(motor_names)}")
        action_vec = action_vec[: len(motor_names)]

        return {m: _clamp_so100_action(m, float(action_vec[i])) for i, m in enumerate(motor_names)}
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
    camera_indexes: Dict[str, Union[int, str, Path]] = field(default_factory=dict)

    def configure(self, names: List[str], indexes: List[Union[int, str, Path]]) -> None:
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
    motion_enabled: bool = False


# Global state instances
camera_state = CameraState()
robot_state = RobotState()
running_skills: Dict[str, SkillExecution] = {}
# Configure cameras from environment using shared parser
_camera_sources = _parse_camera_sources()
_camera_names = _parse_camera_names()
camera_state.configure(_camera_names, _camera_sources)

# Skill objects (lazy-loaded)
_search_policy_path = _as_pretrained_name_or_path(os.environ.get("SEARCH_POLICY_PATH"))
_grasp_policy_path = _as_pretrained_name_or_path(os.environ.get("GRASP_POLICY_PATH"))
_smolvla_policy_id = _as_pretrained_name_or_path(os.environ.get("SMOLVLA_POLICY_ID"))
_xvla_policy_id = _as_pretrained_name_or_path(os.environ.get("XVLA_POLICY_ID"))

policy_runners: Dict[str, VLAPolicyRunner] = {}
if _smolvla_policy_id:
    policy_runners["smolvla"] = VLAPolicyRunner(policy_id=_smolvla_policy_id)
if _xvla_policy_id:
    policy_runners["xvla"] = VLAPolicyRunner(policy_id=_xvla_policy_id)


def _parse_policy_camera_map() -> dict[str, str]:
    """
    Map policy camera names -> robot camera names.

    Example:
      SO100_POLICY_CAMERA_MAP='{"front":"overhead","wrist":"wrist"}'
    """
    raw = os.environ.get("SO100_POLICY_CAMERA_MAP", "").strip()
    if not raw:
        return {"front": "overhead", "top": "overhead", "wrist": "wrist"}
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {"front": "overhead", "top": "overhead", "wrist": "wrist"}
    if not isinstance(data, dict):
        return {"front": "overhead", "top": "overhead", "wrist": "wrist"}
    result: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str) and k and v:
            result[k] = v
    return result

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
        port: Serial port (e.g., "/dev/ttyUSB0"), "auto" to detect, or "mock" for safe mock mode.

    Returns:
        Connection status and robot info
    """
    global robot_state

    if robot_state.connected:
        return {"status": "already_connected", "port": port}

    try:
        if port.strip().lower() == "mock":
            os.environ["USE_MOCK_ROBOT"] = "true"
            cfg = SO100DemoConfig()
            robot_state.robot_interface = make_robot_interface(cfg)
            robot_state.robot_interface.connect()
            robot_state.connected = True
            robot_state.motion_enabled = True
            return {"status": "connected", "port": "mock"}

        # Let SO100RobotInterface handle discovery when asked.
        if port.strip().lower() in {"auto", "auto-detect", "autodetect"}:
            os.environ["SO100_PORT"] = "auto"
        else:
            os.environ["SO100_PORT"] = port
        os.environ["USE_MOCK_ROBOT"] = "false"

        cfg = SO100DemoConfig()
        robot_state.robot_interface = make_robot_interface(cfg)
        robot_state.robot_interface.connect()
        robot_state.connected = True

        # Best-effort actual port used (especially when auto-detect is enabled).
        connected_port = None
        try:
            connected_port = getattr(getattr(robot_state.robot_interface, "config", None), "port", None)
        except Exception:  # noqa: BLE001
            connected_port = None

        robot_state.motion_enabled = os.environ.get("SO100_ENABLE_MOTION", "false").lower() in {"1", "true", "yes"}
        return {"status": "connected", "port": connected_port or os.environ.get("SO100_PORT", port)}

    except Exception as e:
        logger.error(f"Failed to connect robot: {e}")
        msg = str(e)
        if "Permission denied" in msg or "permission denied" in msg:
            msg = (
                f"{msg} (Hint: on Linux, add your user to the 'dialout' group for /dev/ttyACM* access: "
                "`sudo usermod -a -G dialout $USER` then log out/in.)"
            )
        return {"status": "error", "error": msg}


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
        robot_state.motion_enabled = False
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
            "cameras_active": list(images.keys()),
            "motion_enabled": bool(robot_state.motion_enabled),
        }
    except Exception as e:
        return {"connected": True, "joints": {}, "motion_enabled": bool(robot_state.motion_enabled), "error": str(e)}


@mcp.tool()
def get_robot_camera_frame(camera_name: str = "wrist") -> Image:
    """
    Get camera frame from the connected robot's cameras.

    This is different from get_camera_frame() - it uses the robot interface's
    cameras which are already open when the robot is connected.

    Args:
        camera_name: Name of camera (use get_robot_state() to see cameras_active)

    Returns:
        Current camera frame as an image
    """
    if not robot_state.connected or not robot_state.robot_interface:
        # Return error placeholder
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Robot not connected",
            (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        pil_img = PILImage.fromarray(placeholder)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return Image(data=buf.getvalue(), format="jpeg")

    try:
        images, _ = robot_state.robot_interface.get_observation()
        if camera_name not in images:
            available = list(images.keys())
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                f"Camera '{camera_name}' not found",
                (50, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                placeholder,
                f"Available: {available}",
                (50, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2
            )
            pil_img = PILImage.fromarray(placeholder)
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            return Image(data=buf.getvalue(), format="jpeg")

        frame = images[camera_name]
        # Ensure RGB and uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        pil_img = PILImage.fromarray(frame)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return Image(data=buf.getvalue(), format="jpeg")

    except Exception as e:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            f"Error: {str(e)[:40]}",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 100, 100),
            2
        )
        pil_img = PILImage.fromarray(placeholder)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return Image(data=buf.getvalue(), format="jpeg")


@mcp.tool()
def list_serial_ports() -> List[Dict[str, str]]:
    """
    List likely SO100 serial ports and their permissions (Linux).

    This helps debug connect failures (most commonly: user not in 'dialout').
    """
    ports: list[str] = []
    for p in ["/dev/serial/by-id", "/dev"]:
        try:
            base = Path(p)
            if not base.exists():
                continue
            if base.name == "by-id":
                for item in sorted(base.iterdir(), key=lambda x: x.name):
                    if item.is_symlink():
                        ports.append(str(item))
            else:
                for pattern in ("ttyACM*", "ttyUSB*"):
                    for item in sorted(base.glob(pattern), key=lambda x: x.name):
                        ports.append(str(item))
        except Exception:  # noqa: BLE001
            continue

    seen: set[str] = set()
    results: list[dict[str, str]] = []
    for p in ports:
        if p in seen:
            continue
        seen.add(p)
        try:
            st = os.stat(p)
            mode = oct(st.st_mode & 0o777)
            try:
                user = pwd.getpwuid(st.st_uid).pw_name
            except Exception:  # noqa: BLE001
                user = str(st.st_uid)
            try:
                group = grp.getgrgid(st.st_gid).gr_name
            except Exception:  # noqa: BLE001
                group = str(st.st_gid)
            results.append(
                {
                    "port": p,
                    "mode": mode,
                    "owner": user,
                    "group": group,
                    "can_read_write": str(bool(os.access(p, os.R_OK | os.W_OK))),
                }
            )
        except Exception as e:  # noqa: BLE001
            results.append({"port": p, "error": str(e)})
    return results


@mcp.tool()
def enable_motion(enable: bool = False) -> Dict[str, Any]:
    """
    Explicitly enable/disable robot motion for MCP-triggered skills.

    Safety: on real hardware we require this to be enabled before `start_skill()` will send actions.
    """
    robot_state.motion_enabled = bool(enable)
    return {"status": "ok", "motion_enabled": bool(robot_state.motion_enabled)}


# =============================================================================
# Skill Tools
# =============================================================================

@mcp.tool()
def list_skills() -> List[Dict[str, Any]]:
    """
    List available VLA skills that can be executed.

    Returns:
        List of skills with names and descriptions
    """
    skills: list[dict[str, Any]] = [
        {
            "name": "smolvla",
            "description": "Run SmolVLA policy with a natural-language instruction (e.g. 'pick up the blue block').",
            "requires_policy": True,
            "policy_env": "SMOLVLA_POLICY_ID",
            "policy_id": os.environ.get("SMOLVLA_POLICY_ID", ""),
        },
        {
            "name": "xvla",
            "description": "Run XVLA policy with a natural-language instruction (if you have an XVLA checkpoint).",
            "requires_policy": True,
            "policy_env": "XVLA_POLICY_ID",
            "policy_id": os.environ.get("XVLA_POLICY_ID", ""),
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

        # Dispatch to specific skills
        robot = robot_state.robot_interface

        if execution.skill_name == "home":
            # Simple home: drive all joints back toward 0 based on current obs
            images, joints = robot.get_observation()
            targets = {name: 0.0 for name in joints.keys()}
            robot.send_joint_targets(targets)
            execution.steps_completed = 1
            execution.status = "completed"
            logger.info("Home skill sent zero targets to all joints.")
            return

        if execution.skill_name in {"smolvla", "xvla"}:
            runner = policy_runners.get(execution.skill_name)
            if runner is None:
                execution.status = "error"
                execution.error = f"No policy configured for {execution.skill_name}. Set {execution.skill_name.upper()}_POLICY_ID."
                return
            motor_names = _get_motor_names_from_robot_interface(robot)
            if not motor_names:
                execution.status = "error"
                execution.error = "Could not determine motor names (robot not connected?)"
                return
            instruction = getattr(execution, "instruction", "").strip()
            if not instruction:
                target = getattr(execution, "target_object", "").strip()
                instruction = f"pick up the {target}" if target else "do the task"

            for step in range(execution.max_steps):
                if execution.stop_requested:
                    execution.status = "stopped"
                    execution.steps_completed = step
                    return
                images, joints = robot.get_observation()
                if not isinstance(images, dict) or not images:
                    execution.status = "error"
                    execution.error = "No camera image available"
                    return
                joint_targets = runner.step(
                    images=images,
                    joints=joints,
                    motor_names=motor_names,
                    instruction=instruction,
                )
                robot.send_joint_targets(joint_targets)
                execution.steps_completed = step + 1
                time.sleep(0.05)

            execution.status = "completed"
            return

        execution.status = "error"
        execution.error = f"Unknown skill {execution.skill_name}"

    except Exception as e:
        execution.status = "error"
        execution.error = str(e)
        logger.error(f"Skill {execution.skill_id} failed: {e}")


@mcp.tool()
def start_skill(
    skill_name: str,
    target_object: str = "",
    max_steps: int = 100,
    instruction: str = "",
) -> Dict[str, Any]:
    """
    Start executing a VLA skill.

    Args:
        skill_name: Name of skill to execute ("smolvla", "xvla", "home")
        target_object: Optional legacy object string (used only if instruction is empty)
        max_steps: Maximum steps to run before stopping
        instruction: Natural-language instruction for the policy (e.g. "pick up the red cup")

    Returns:
        Skill execution ID and initial status. Use get_skill_status() to monitor.
    """
    # Validate skill name
    valid_skills = ["home", "smolvla", "xvla"]
    if skill_name not in valid_skills:
        return {"status": "error", "error": f"Unknown skill: {skill_name}. Valid: {valid_skills}"}

    # Check robot connection
    if not robot_state.connected:
        return {"status": "error", "error": "Robot not connected. Call connect_robot() first."}

    # Safety gate: require explicit enable on real robot.
    is_mock = type(robot_state.robot_interface).__name__.lower().startswith("mock")
    if not is_mock and not robot_state.motion_enabled:
        return {
            "status": "error",
            "error": "Motion is disabled for safety. Call enable_motion(true) (or set SO100_ENABLE_MOTION=true) before start_skill().",
        }

    # Create execution
    skill_id = f"skill_{uuid.uuid4().hex[:8]}"
    execution = SkillExecution(
        skill_id=skill_id,
        skill_name=skill_name,
        status="running",
        max_steps=max_steps
    )
    execution.target_object = target_object  # type: ignore[attr-defined]
    execution.instruction = instruction  # type: ignore[attr-defined]

    # Start background thread
    thread = threading.Thread(target=_run_skill_loop, args=(execution,), daemon=True)
    execution.thread = thread
    running_skills[skill_id] = execution
    thread.start()

    return {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "target_object": target_object,
        "instruction": instruction,
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


@mcp.tool()
def set_policy(policy_name: str, policy_id: str) -> Dict[str, Any]:
    """
    Configure (or replace) a policy checkpoint for a VLA skill.

    Args:
        policy_name: "smolvla" or "xvla"
        policy_id: Hugging Face repo id (e.g. "Gurkinator/smolvla_so100_policy") or local folder path.

    Returns:
        Status (policy is loaded lazily on first use).
    """
    policy_name = policy_name.strip().lower()
    if policy_name not in {"smolvla", "xvla"}:
        return {"status": "error", "error": "policy_name must be 'smolvla' or 'xvla'"}
    policy_id = policy_id.strip()
    if not policy_id:
        return {"status": "error", "error": "policy_id is required"}

    policy_runners[policy_name] = VLAPolicyRunner(policy_id=policy_id)
    env_key = f"{policy_name.upper()}_POLICY_ID"
    os.environ[env_key] = policy_id
    return {"status": "ok", "policy_name": policy_name, "policy_id": policy_id, "env": env_key}


@mcp.tool()
def warmup_policy(policy_name: str) -> Dict[str, Any]:
    """
    Pre-load a policy checkpoint into memory.

    Useful to do once before a live demo so `start_skill()` doesn't block on initial model download/load.
    """
    policy_name = policy_name.strip().lower()
    runner = policy_runners.get(policy_name)
    if runner is None:
        env_key = f"{policy_name.upper()}_POLICY_ID"
        return {
            "status": "error",
            "error": f"Policy '{policy_name}' is not configured. Set via set_policy() or {env_key}.",
        }
    try:
        runner.ensure_loaded()
        return {"status": "ok", "policy_name": policy_name, "policy_id": runner.policy_id}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e), "policy_name": policy_name, "policy_id": runner.policy_id}


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="SO100 VLA MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: 'stdio' for Claude Code, 'sse' for HTTP server (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for SSE server (default: 8765)"
    )
    args = parser.parse_args()

    logger.info("Starting SO100 VLA MCP Server...")
    logger.info(f"Cameras configured: {list(camera_state.camera_indexes.keys())}")
    logger.info(f"Transport: {args.transport}")

    if args.transport == "sse":
        logger.info(f"SSE server running on http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        # stdio transport for Claude Code integration
        # IMPORTANT: suppress CLI banner for stdio clients (it would corrupt the protocol stream).
        mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
