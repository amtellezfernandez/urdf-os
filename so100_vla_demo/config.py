from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower import SO100FollowerConfig


@dataclass
class SO100DemoConfig:
    """
    Basic configuration for the SO100 demo.

    Adjust the defaults or override via environment variables:
    - SO100_PORT
    - SO100_CAMERA_INDEX
    """

    port: str = os.environ.get("SO100_PORT", "/dev/ttyUSB0")
    camera_index: int = int(os.environ.get("SO100_CAMERA_INDEX", "0"))
    demo_fps: int = 15
    # If True, use a local mock robot instead of real SO100 hardware.
    # This is useful for running the demo on any laptop / VM.
    use_mock: bool = os.environ.get("USE_MOCK_ROBOT", "false").lower() in {"1", "true", "yes"}
    # Optional sources for mock camera images.
    mock_video_path: Optional[str] = os.environ.get("MOCK_VIDEO_PATH")
    mock_static_image_path: Optional[str] = os.environ.get("MOCK_STATIC_IMAGE_PATH")
    # Optional path to a trained search policy checkpoint
    search_policy_path: Optional[str] = None
    # Optional path to a trained grasp policy checkpoint (e.g. SmolVLA/XVLA)
    grasp_policy_path: Optional[str] = None

    def to_robot_config(self) -> SO100FollowerConfig:
        cam_cfg = OpenCVCameraConfig(
            index_or_path=self.camera_index,
            width=640,
            height=480,
            fps=self.demo_fps,
        )
        # The cameras field is a dict[name -> CameraConfig]
        return SO100FollowerConfig(
            port=self.port,
            cameras={"wrist": cam_cfg},
        )

