from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import SO100DemoConfig
from .demo_orchestrator import SO100DemoOrchestrator
from .grasp_skill import GraspPolicySkill
from .robot_interface import SO100RobotInterface
from .search_skill import SearchPolicySkill


def simple_pixel_detector(frame: np.ndarray, object_name: str) -> bool:
    """
    Placeholder detection function.

    Replace this with a real VLM / detector that checks if `object_name`
    is visible in the frame. For now it always returns False so you
    don't accidentally move the robot without a proper policy.
    """
    _ = frame, object_name
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO100 VLA search-and-grasp demo scaffold.")
    parser.add_argument("--object-name", type=str, default="tennis ball", help="Name of the object to search.")
    parser.add_argument(
        "--search-policy-path",
        type=str,
        required=False,
        help="Path to trained search policy `pretrained_model` directory.",
    )
    parser.add_argument(
        "--grasp-policy-path",
        type=str,
        required=False,
        help="Path to trained grasp policy `pretrained_model` directory.",
    )
    parser.add_argument(
        "--so100-port",
        type=str,
        default=None,
        help="Serial port for the SO100 follower arm. Overrides SO100_PORT env var.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="OpenCV camera index for the wrist camera. Overrides SO100_CAMERA_INDEX env var.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    cfg_kwargs: dict[str, Any] = {}
    if args.so100_port is not None:
        cfg_kwargs["port"] = args.so100_port
    if args.camera_index is not None:
        cfg_kwargs["camera_index"] = args.camera_index

    cfg = SO100DemoConfig(**cfg_kwargs)

    if args.grasp_policy_path is None:
        logging.warning(
            "You did not provide --grasp-policy-path. "
            "The demo scaffold will construct, but the grasp skill will not run a real policy."
        )

    robot_cfg = cfg.to_robot_config()
    robot = SO100RobotInterface(robot_cfg)

    search_policy_path = Path(args.search_policy_path) if args.search_policy_path else None
    search_skill = SearchPolicySkill(policy_path=search_policy_path)

    grasp_policy_path = Path(args.grasp_policy_path or "MISSING_GRASP_POLICY")
    grasp_skill = GraspPolicySkill(policy_path=grasp_policy_path)

    orchestrator = SO100DemoOrchestrator(
        cfg=cfg,
        robot=robot,
        search_skill=search_skill,
        grasp_skill=grasp_skill,
        detect_fn=simple_pixel_detector,
    )

    orchestrator.run(object_name=args.object_name)


if __name__ == "__main__":
    main()
