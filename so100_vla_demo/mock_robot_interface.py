from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .config import SO100DemoConfig

logger = logging.getLogger(__name__)

try:  # Optional OpenCV dependency for loading videos/images if available.
    import cv2  # type: ignore[import]
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]


@dataclass
class MockRobotInterface:
    """
    Lightweight mock robot used for debugging the demo without hardware.

    It returns:
    - synthetic images with simple colored shapes, or
    - frames from a video file, or
    - a static image,
    and maintains a fake joint-state dictionary updated by `send_joint_targets`.
    """

    cfg: SO100DemoConfig
    joints: Dict[str, float] = field(
        default_factory=lambda: {"joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0}
    )
    _static_image: np.ndarray | None = None
    _video_cap: "cv2.VideoCapture | None" = None  # type: ignore[name-defined]
    _real_camera_caps: Dict[str, "cv2.VideoCapture"] | None = None  # type: ignore[name-defined]
    _last_real_frames: Dict[str, np.ndarray] = field(default_factory=dict)
    _frame_index: int = 0
    _connected: bool = False

    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        logger.info("MockRobotInterface connected (mock mode).")

        # Optional: use real cameras even in mock mode (for UI streaming/debug).
        # This is best-effort: if a camera fails to open, we fall back to synthetic frames.
        if self.cfg.use_real_cameras and cv2 is not None:
            self._real_camera_caps = {}
            cam_names = self.cfg.get_camera_names()
            for name, source in zip(cam_names, self.cfg.camera_indexes):
                try:
                    cap_source = int(source) if isinstance(source, int) else str(source)
                    cap = cv2.VideoCapture(cap_source)
                    if not cap.isOpened():
                        cap.release()
                        logger.warning("Could not open real camera %s (%s); using synthetic fallback.", name, source)
                        continue
                    if self.cfg.camera_fourcc:
                        try:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.camera_fourcc))
                        except Exception:  # noqa: BLE001
                            pass
                    if self.cfg.camera_width:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.camera_width))
                    if self.cfg.camera_height:
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.camera_height))
                    if self.cfg.camera_fps:
                        cap.set(cv2.CAP_PROP_FPS, float(self.cfg.camera_fps))
                    self._real_camera_caps[name] = cap
                    logger.info("Opened real camera %s (%s)", name, source)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Error opening real camera %s (%s): %s", name, source, e)
            if not self._real_camera_caps:
                self._real_camera_caps = None

        # Lazy-load static image if configured.
        if self.cfg.mock_static_image_path:
            try:
                self._static_image = self._load_image(self.cfg.mock_static_image_path)
                logger.info("Loaded mock static image from %s", self.cfg.mock_static_image_path)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to load mock static image: %s", e)
                self._static_image = None

        # Lazy-open video if configured and OpenCV is available.
        if self.cfg.mock_video_path and cv2 is not None:
            try:
                self._video_cap = cv2.VideoCapture(self.cfg.mock_video_path)
                if not self._video_cap.isOpened():
                    logger.error("Failed to open mock video at %s", self.cfg.mock_video_path)
                    self._video_cap = None
            except Exception as e:  # noqa: BLE001
                logger.error("Exception while opening mock video: %s", e)
                self._video_cap = None

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._video_cap = None
        if self._real_camera_caps:
            for cap in self._real_camera_caps.values():
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass
        self._real_camera_caps = None
        self._last_real_frames.clear()
        self._connected = False
        logger.info("MockRobotInterface disconnected.")

    # Public API expected by the rest of the demo ---------------------

    def get_observation(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Returns:
            images: dict of camera_name -> HxWxC uint8 RGB array for each configured camera.
            joints: dict of fake joint positions.
        """
        if not self._connected:
            raise RuntimeError("MockRobotInterface not connected.")

        camera_names = self.cfg.get_camera_names()
        images: Dict[str, np.ndarray] = {}
        for name in camera_names:
            frame = self._get_frame(camera_name=name)
            images[name] = frame.copy()
        return images, dict(self.joints)

    def send_joint_targets(self, joint_targets: Dict[str, float]) -> None:
        """
        Update internal joint state without touching any hardware.
        """
        self.joints.update(joint_targets)

    # Helpers ----------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        if cv2 is not None:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                raise RuntimeError(f"cv2.imread failed for {path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        # Fallback to PIL if OpenCV is not available.
        from PIL import Image  # lazy import

        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _get_frame(self, camera_name: str | None = None) -> np.ndarray:
        """
        Choose the appropriate image source in this order:
        0. Real camera (if enabled and available).
        1. Video file (if configured and available).
        2. Static image (if configured).
        3. Synthetic scene.
        """
        # 0) Real camera source (best-effort per camera).
        if self._real_camera_caps is not None and cv2 is not None and camera_name is not None:
            cap = self._real_camera_caps.get(camera_name)
            if cap is not None:
                ret, frame_bgr = cap.read()
                if ret and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # type: ignore[arg-type]
                    self._last_real_frames[camera_name] = frame_rgb
                    return frame_rgb
                # If a read fails, reuse last good frame if any.
                last = self._last_real_frames.get(camera_name)
                if last is not None:
                    return last

        # 1) Video source
        if self._video_cap is not None:
            ret, frame_bgr = self._video_cap.read()
            if not ret:
                # Loop video
                self._video_cap.set(0, 0)
                ret, frame_bgr = self._video_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # type: ignore[arg-type]
                return frame_rgb

        # 2) Static image source
        if self._static_image is not None:
            return self._static_image.copy()

        # 3) Synthetic scene
        return self._generate_synthetic_scene()

    def _generate_synthetic_scene(self) -> np.ndarray:
        """
        Generate a simple synthetic RGB image with colored objects.

        - Background: dark gray
        - "Table": lighter rectangle
        - Objects: red cup, blue block, green ball (solid colored shapes)
        """
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (40, 40, 40)  # dark background

        # Table area
        img[int(h * 0.6) : int(h * 0.8), int(w * 0.1) : int(w * 0.9)] = (90, 90, 90)

        # Simple placement that can change over time to simulate "search"
        self._frame_index += 1
        phase = (self._frame_index // 20) % 3

        # Red cup
        if phase >= 0:
            img[int(h * 0.55) : int(h * 0.7), int(w * 0.2) : int(w * 0.25)] = (255, 0, 0)
        # Blue block (appears later)
        if phase >= 1:
            img[int(h * 0.55) : int(h * 0.7), int(w * 0.45) : int(w * 0.5)] = (0, 0, 255)
        # Green ball (appears even later)
        if phase >= 2:
            cy, cx = int(h * 0.65), int(w * 0.75)
            r = int(min(h, w) * 0.05)
            yy, xx = np.ogrid[:h, :w]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
            img[mask] = (0, 255, 0)

        return img


def make_mock_robot_interface(cfg: SO100DemoConfig) -> MockRobotInterface:
    """
    Convenience factory to build a MockRobotInterface from SO100DemoConfig.
    """

    return MockRobotInterface(cfg=cfg)
