from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR

from .robot_interface import SO101RobotInterface

logger = logging.getLogger(__name__)


@dataclass
class GraspPolicySkill:
    """
    Wraps a trained manipulation policy (e.g. SmolVLA, XVLA, Diffusion)
    into a simple `grasp_object` skill.

    You will:
    - train a policy on your grasp dataset
    - point `policy_path` to its `pretrained_model` directory
    - optionally pass in the corresponding pre/post processors
    """

    policy_path: Path
    preprocessor: PolicyProcessorPipeline | None = None
    postprocessor: PolicyProcessorPipeline | None = None
    policy: PreTrainedPolicy | None = None

    def load(self) -> None:
        logger.info(f"Loading grasp policy from {self.policy_path}...")
        self.policy = PreTrainedPolicy.from_pretrained(self.policy_path)
        logger.info(f"Loaded policy type={self.policy.name} on device={self.policy.config.device}")

    def _ensure_loaded(self) -> None:
        if self.policy is None:
            self.load()

    def step(self, image: np.ndarray, joints: Dict[str, float]) -> Dict[str, float]:
        """
        Single forward pass of the grasp policy.

        Returns:
            action dict keyed by ACTION-prefixed feature names.
        """
        self._ensure_loaded()
        assert self.policy is not None

        obs_frame: Dict[str, np.ndarray] = {
            f"{OBS_STR}.image": image,
            # TODO: add observation.state or other features to match training.
        }
        batch = {OBS_STR: obs_frame}

        if self.preprocessor is not None:
            batch = self.preprocessor(batch)

        with self.policy.no_grad():
            action = self.policy.select_action(batch)

        if self.postprocessor is not None:
            action = self.postprocessor(action)

        flat_action: Dict[str, float] = {}
        for k, v in action.items():
            if k.startswith(ACTION):
                flat_action[k] = float(np.asarray(v).squeeze())

        return flat_action

    def run_grasp_loop(
        self,
        robot: SO101RobotInterface,
        max_steps: int = 100,
    ) -> None:
        """
        Run the grasp policy for a fixed horizon.

        In a real demo, you can add:
        - success detection (e.g. gripper closed + object height)
        - early stopping when success is detected
        """
        self._ensure_loaded()

        for step in range(max_steps):
            image, joints = robot.get_observation()
            action_dict = self.step(image, joints)
            robot.send_joint_targets(
                {
                    name.replace(".pos", ""): val
                    for name, val in action_dict.items()
                    if name.endswith(".pos")
                }
            )
            logger.debug(f"[grasp_skill] step={step}")


