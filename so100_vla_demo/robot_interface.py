from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

from .config import SO100DemoConfig
from .mock_robot_interface import MockRobotInterface, make_mock_robot_interface
logger = logging.getLogger(__name__)


@dataclass
class SO100RobotInterface:
    """
    Thin wrapper around LeRobot's SO100Follower robot.

    Responsibilities:
    - connect / disconnect robot
    - fetch a single camera frame + joint state
    - send a joint-space command
    """

    config: SO100FollowerConfig
    robot: SO100Follower | None = None

    def connect(self) -> None:
        if self.robot is not None and self.robot.is_connected:
            logger.warning("SO100RobotInterface.connect called but robot is already connected.")
            return
        self.robot = SO100Follower(self.config)
        logger.info("Connecting SO100Follower...")
        self.robot.connect()
        logger.info("SO100Follower connected.")

    def disconnect(self) -> None:
        if self.robot is None:
            return
        try:
            self.robot.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error while disconnecting robot: {e}")
        finally:
            self.robot = None

    def get_observation(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Returns:
            image: last frame from the first configured camera, as HxWxC uint8 numpy array.
            joints: dict of joint_name -> position (float).
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected.")

        obs: Dict[str, Any] = self.robot.get_observation()

        # Extract first camera frame
        cam_names = list(self.robot.cameras.keys())
        if not cam_names:
            raise RuntimeError("No cameras configured on SO100Follower.")
        cam_key = cam_names[0]
        image = obs[cam_key]

        # Extract joint positions
        joints: Dict[str, float] = {}
        for name in self.robot.bus.motors.keys():
            key = f"{name}.pos"
            if key in obs:
                joints[name] = float(obs[key])

        return image, joints

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


