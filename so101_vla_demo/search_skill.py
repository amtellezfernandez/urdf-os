from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR

from .robot_interface import SO101RobotInterface

logger = logging.getLogger(__name__)


DetectFn = Callable[[np.ndarray, str], bool]


@dataclass
class SearchPolicySkill:
    """
    Wraps a trained search policy into a simple `search_object` skill.

    This class is intentionally generic:
    - it loads any LeRobot policy saved with .save_pretrained()
    - it assumes the policy expects a single image + state observation
      and outputs a single action vector.

    You are expected to train this policy on your `my_so101_search_for_ball`
    dataset and point `policy_path` to the `pretrained_model` directory.
    """

    policy_path: Optional[Path]
    preprocessor: PolicyProcessorPipeline | None = None
    postprocessor: PolicyProcessorPipeline | None = None
    policy: PreTrainedPolicy | None = None

    def load(self) -> None:
        if self.policy_path is None:
            raise RuntimeError(
                "SearchPolicySkill.load called without a policy_path. "
                "Provide --search-policy-path pointing to a trained search policy."
            )
        logger.info(f"Loading search policy from {self.policy_path}...")
        self.policy = PreTrainedPolicy.from_pretrained(self.policy_path)
        logger.info(f"Loaded policy type={self.policy.name} on device={self.policy.config.device}")

    def _ensure_loaded(self) -> None:
        if self.policy is None:
            self.load()

    def step(self, image: np.ndarray, joints: Dict[str, float]) -> Dict[str, float]:
        """
        Single forward pass of the search policy.

        Returns:
            action dict keyed by ACTION-prefixed feature names.

        Note: you must make sure that the dataset you used for training
        and the features used here match (image + state keys).
        """
        self._ensure_loaded()
        assert self.policy is not None

        # Build a minimal batch with one frame.
        # Keys must match what you used during training (e.g. observation.image, observation.state).
        obs_frame: Dict[str, np.ndarray] = {
            f"{OBS_STR}.image": image,
            # TODO: add observation.state or per-joint features as needed.
        }
        batch = {OBS_STR: obs_frame}

        if self.preprocessor is not None:
            batch = self.preprocessor(batch)

        with self.policy.no_grad():
            action: Dict[str, np.ndarray] = self.policy.select_action(batch)

        if self.postprocessor is not None:
            action = self.postprocessor(action)

        # Convert numpy outputs to plain floats for robot control.
        flat_action: Dict[str, float] = {}
        for k, v in action.items():
            if k.startswith(ACTION):
                flat_action[k] = float(np.asarray(v).squeeze())

        return flat_action

    def run_search_loop(
        self,
        robot: SO101RobotInterface,
        object_name: str,
        detect_fn: DetectFn,
        max_steps: int = 50,
    ) -> Tuple[bool, int]:
        """
        Repeatedly runs the search policy until the object is visible or we exhaust steps.

        Args:
            robot: connected SO101RobotInterface.
            object_name: e.g. "tennis ball".
            detect_fn: function(frame, object_name) -> bool telling if the object is visible.
            max_steps: maximum policy steps before giving up.

        Returns:
            (found, steps_taken)
        """
        self._ensure_loaded()

        for step in range(max_steps):
            image, joints = robot.get_observation()

            if detect_fn(image, object_name):
                logger.info(f"[search_skill] {object_name} visible after {step} steps.")
                return True, step

            action_dict = self.step(image, joints)
            # For now we assume action_dict already matches robot.send_action format.
            robot.send_joint_targets(
                {
                    name.replace(".pos", ""): val
                    for name, val in action_dict.items()
                    if name.endswith(".pos")
                }
            )

        logger.info(f"[search_skill] Failed to find {object_name} after {max_steps} steps.")
        return False, max_steps

