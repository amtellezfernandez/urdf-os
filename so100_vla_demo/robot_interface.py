from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

from .config import SO100DemoConfig
from .mock_robot_interface import MockRobotInterface, make_mock_robot_interface

logger = logging.getLogger(__name__)


def _discover_serial_ports() -> list[str]:
    """
    Best-effort serial port discovery for SO100 (Linux-first).

    Returns a prioritized list:
    1) Stable symlinks under `/dev/serial/by-id/` (if present)
    2) `/dev/ttyACM*` then `/dev/ttyUSB*`
    """
    ports: list[str] = []

    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        for p in sorted(by_id.iterdir(), key=lambda x: x.name):
            ports.append(str(p))

    dev = Path("/dev")
    for pattern in ("ttyACM*", "ttyUSB*"):
        for p in sorted(dev.glob(pattern), key=lambda x: x.name):
            ports.append(str(p))

    # De-dupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in ports:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


@dataclass
class SO100RobotInterface:
    """
    Thin wrapper around LeRobot's SO100Follower robot.

    Responsibilities:
    - connect / disconnect robot
    - fetch all camera frames + joint state
    - send a joint-space command
    """

    config: SO100FollowerConfig
    robot: SO100Follower | None = None
    _connected: bool = False

    def connect(self) -> None:
        if self.robot is not None and self.robot.is_connected:
            logger.warning("SO100RobotInterface.connect called but robot is already connected.")
            return
        calibrate = os.environ.get("SO100_CALIBRATE", "false").lower() in {"1", "true", "yes"}

        requested_port = (self.config.port or "").strip()
        should_auto = requested_port.lower() in {"auto", "auto-detect", "autodetect", ""}
        port_exists = bool(requested_port) and Path(requested_port).exists()

        ports_to_try: list[str] = []
        if not should_auto:
            ports_to_try.append(requested_port)
        if should_auto or not port_exists:
            ports_to_try.extend(_discover_serial_ports())

        last_err: Exception | None = None
        for port in ports_to_try:
            try:
                self.config.port = port
                self.robot = SO100Follower(self.config)
                logger.info("Connecting SO100Follower on port=%s ...", port)
                self.robot.connect(calibrate=calibrate)
                self._connected = True
                logger.info("SO100Follower connected on port=%s.", port)
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("Failed to connect SO100Follower on port=%s: %s", port, e)
                try:
                    if self.robot is not None and self.robot.is_connected:
                        self.robot.disconnect()
                except Exception:  # noqa: BLE001
                    pass
                self.robot = None
                self._connected = False

        hint = (
            "Set SO100_PORT=/dev/ttyACM0 (or your port), or set SO100_PORT=auto to auto-detect."
        )
        perm_hint = ""
        if last_err is not None:
            msg = str(last_err)
            if isinstance(last_err, PermissionError) or "Permission denied" in msg or "permission denied" in msg:
                perm_hint = (
                    " If you're on Linux, you likely need serial permissions: "
                    "run `sudo usermod -a -G dialout $USER` then log out/in."
                )
        raise RuntimeError(f"Could not connect SO100Follower. Tried ports={ports_to_try}. {hint}{perm_hint}") from last_err

    def disconnect(self) -> None:
        if self.robot is None:
            return
        try:
            self.robot.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error while disconnecting robot: {e}")
        finally:
            self.robot = None
            self._connected = False

    def get_observation(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Returns:
            images: dict of camera_name -> HxWxC uint8 numpy array for all configured cameras.
            joints: dict of joint_name -> position (float).
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected.")

        obs: Dict[str, Any] = self.robot.get_observation()

        # Extract all camera frames
        images: Dict[str, np.ndarray] = {}
        for cam_name in self.robot.cameras.keys():
            if cam_name in obs:
                images[cam_name] = obs[cam_name]

        if not images:
            raise RuntimeError("No camera frames received from SO100Follower.")

        # Extract joint positions
        joints: Dict[str, float] = {}
        for name in self.robot.bus.motors.keys():
            key = f"{name}.pos"
            if key in obs:
                joints[name] = float(obs[key])

        return images, joints

    def send_joint_targets(self, joint_targets: Dict[str, float]) -> None:
        """
        Send joint targets in the normalised units expected by SO100Follower.

        For the hackathon, you can:
        - send absolute positions (e.g. degrees if use_degrees=True in config)
        - or convert from delta commands before calling this function.
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected.")

        # Build action dict in the format expected by robot.send_action
        action: Dict[str, float] = {}
        for name, target in joint_targets.items():
            key = f"{name}.pos"
            action[key] = target

        self.robot.send_action(action)


def make_robot_interface(cfg: SO100DemoConfig) -> SO100RobotInterface | MockRobotInterface:
    """
    Factory that returns either a real SO100RobotInterface or a MockRobotInterface
    depending on the configuration.

    This allows the rest of the demo (server, orchestrator) to be written against
    a single interface without worrying about hardware availability.
    """

    if cfg.use_mock:
        return make_mock_robot_interface(cfg)
    robot_cfg: SO100FollowerConfig = cfg.to_robot_config()
    return SO100RobotInterface(robot_cfg)
