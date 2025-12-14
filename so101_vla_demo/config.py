from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower import SO100FollowerConfig


def _parse_camera_sources() -> List[int | Path]:
    """
    Parse camera sources from environment variables (comma-separated).

    Supports:
    - `SO101_CAMERA_SOURCES` (preferred): e.g. "0,2,/dev/video4"
    - `SO101_CAMERA_INDEXES` (legacy plural): e.g. "0,2,4"
    - `SO101_CAMERA_INDEX` (legacy singular): e.g. "6"
    """
    raw = os.environ.get("SO101_CAMERA_SOURCES")
    if raw is None:
        raw = os.environ.get("SO101_CAMERA_INDEXES")
    if raw is None:
        raw = os.environ.get("SO101_CAMERA_INDEX")
    if not raw:
        raw = "0"

    sources: List[int | Path] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            sources.append(int(token))
        except ValueError:
            sources.append(Path(token).expanduser())
    return sources


def _parse_camera_names() -> List[str]:
    """Parse camera names from environment variable (comma-separated)."""
    raw = os.environ.get("SO101_CAMERA_NAMES", "wrist,overhead,side")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _use_real_cameras() -> bool:
    """Check if we should use real cameras (even in mock robot mode)."""
    return os.environ.get("USE_REAL_CAMERAS", "false").lower() in {"1", "true", "yes"}


def _parse_optional_int(env_name: str) -> int | None:
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_optional_str(env_name: str) -> str | None:
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


@dataclass
class SO101DemoConfig:
    """
    Basic configuration for the SO101 demo.

    Adjust the defaults or override via environment variables:
    - SO101_PORT
    - SO101_ROBOT_ID (must match calibration ID, e.g., "my_so101")
    - SO101_CAMERA_SOURCES (comma-separated, e.g., "0,2,/dev/video4")
      (also supports legacy SO101_CAMERA_INDEXES / SO101_CAMERA_INDEX)
    - SO101_CAMERA_NAMES (comma-separated, e.g., "wrist,overhead,side")
    """

    port: str = os.environ.get("SO101_PORT", "/dev/ttyUSB0")
    # Robot ID - must match the ID used during calibration
    robot_id: Optional[str] = os.environ.get("SO101_ROBOT_ID")
    # Multi-camera support: list of camera sources (OpenCV index int or device path Path)
    camera_indexes: List[int | Path] = field(default_factory=_parse_camera_sources)
    # Camera names corresponding to each index
    camera_names: List[str] = field(default_factory=_parse_camera_names)
    demo_fps: int = 15
    # Mock mode removed; always use real robot.
    use_mock: bool = False
    mock_video_path: Optional[str] = None
    mock_static_image_path: Optional[str] = None
    use_real_cameras: bool = False
    # Camera capture settings (leave unset to use each camera's defaults).
    camera_width: int | None = field(default_factory=lambda: _parse_optional_int("SO101_CAMERA_WIDTH"))
    camera_height: int | None = field(default_factory=lambda: _parse_optional_int("SO101_CAMERA_HEIGHT"))
    camera_fps: int | None = field(default_factory=lambda: _parse_optional_int("SO101_CAMERA_FPS"))
    camera_fourcc: str | None = field(default_factory=lambda: _parse_optional_str("SO101_CAMERA_FOURCC"))
    # Optional path to a trained search policy checkpoint (local or HuggingFace repo_id)
    search_policy_path: Optional[str] = os.environ.get("SEARCH_POLICY_PATH")
    # Optional path to a trained grasp policy checkpoint (e.g. SmolVLA/XVLA, HuggingFace repo_id)
    grasp_policy_path: Optional[str] = os.environ.get("GRASP_POLICY_PATH")

    def to_robot_config(self) -> SO100FollowerConfig:
        """Build SO100FollowerConfig with multiple cameras."""
        cameras = {}
        for i, cam_idx in enumerate(self.camera_indexes):
            # Use camera name if available, otherwise generate one
            if i < len(self.camera_names):
                name = self.camera_names[i]
            else:
                name = f"camera_{i}"
            cameras[name] = OpenCVCameraConfig(
                index_or_path=cam_idx,
                width=self.camera_width or 640,
                height=self.camera_height or 480,
                fps=self.camera_fps or 30,
                fourcc=self.camera_fourcc,
            )
        return SO100FollowerConfig(
            port=self.port,
            cameras=cameras,
            id=self.robot_id,
        )

    def get_camera_names(self) -> List[str]:
        """Get the list of camera names that will be used."""
        names = []
        for i in range(len(self.camera_indexes)):
            if i < len(self.camera_names):
                names.append(self.camera_names[i])
            else:
                names.append(f"camera_{i}")
        return names
