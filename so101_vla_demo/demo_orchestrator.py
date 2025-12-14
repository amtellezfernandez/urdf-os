from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import SO101DemoConfig
from .grasp_skill import GraspPolicySkill
from .robot_interface import SO101RobotInterface
from .search_skill import SearchPolicySkill

logger = logging.getLogger(__name__)


DetectFn = Callable[[np.ndarray, str], bool]


@dataclass
class SO100DemoOrchestrator:
    """
    High-level "search then grasp" demo:
    - use SearchPolicySkill to find the object
    - use GraspPolicySkill to grasp it
    """

    cfg: SO101DemoConfig
    robot: SO101RobotInterface
    search_skill: SearchPolicySkill
    grasp_skill: GraspPolicySkill
    detect_fn: DetectFn

    def run(self, object_name: str) -> None:
        logger.info(f"Starting SO101 demo for object: {object_name}")
        self.robot.connect()
        try:
            # If no search policy is configured, skip search and go straight to grasp.
            if self.search_skill.policy_path is None:
                logger.info("No search policy configured; skipping search phase and running grasp directly.")
            else:
                found, steps = self.search_skill.run_search_loop(
                    robot=self.robot,
                    object_name=object_name,
                    detect_fn=self.detect_fn,
                    max_steps=50,
                )
                if not found:
                    logger.warning("Search failed; skipping grasp.")
                    return
                logger.info(f"Search succeeded in {steps} steps; starting grasp...")

            self.grasp_skill.run_grasp_loop(self.robot, max_steps=100)
            logger.info("Grasp loop finished.")
        finally:
            self.robot.disconnect()

