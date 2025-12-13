import threading
import time
import logging
from typing import Optional

from app.features.joint_data import get_joint_positions_from_robot
from app.services.websocket_manager import manager


logger = logging.getLogger(__name__)


class JointBroadcaster:
    """
    Background service that polls joint positions from a robot and broadcasts
    them to connected WebSocket clients using the existing ConnectionManager.

    This keeps WebSocket publishing decoupled from TeleoperationManager.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._robot = None
        self._joint_names: list[str] = []
        self._rate_hz: float = 20.0

    def start(
        self,
        robot,
        joint_names: Optional[list[str]] = None,
        rate_hz: float = 20.0,
    ):
        with self._lock:
            if self._running:
                logger.info(
                    "JointBroadcaster already running; restarting with new params"
                )
                self.stop()

            self._robot = robot
            self._joint_names = joint_names or []
            self._rate_hz = max(1.0, float(rate_hz))
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._running = True
            logger.info("JointBroadcaster started at %.1f Hz", self._rate_hz)

    def stop(self):
        with self._lock:
            if not self._running:
                return
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=1.0)
            self._thread = None
            self._robot = None
            self._running = False
            logger.info("JointBroadcaster stopped")

    def is_running(self) -> bool:
        return self._running

    def _run(self):
        try:
            interval = 1.0 / self._rate_hz
            last_send = 0.0
            seq = 0
            meta_sent = False

            while not self._stop.is_set():
                if self._robot is None:
                    time.sleep(0.01)
                    continue

                now = time.time()

                # Send meta once
                if not meta_sent:
                    try:
                        initial_positions = get_joint_positions_from_robot(
                            self._joint_names, self._robot
                        )
                        if initial_positions is not None:
                            joint_meta = {
                                "type": "joint_meta",
                                "units": "radians",
                                "names": list(initial_positions.keys()),
                            }
                            manager.broadcast_joint_data_sync(joint_meta)
                            meta_sent = True
                    except Exception as e:
                        logger.error("Failed to send joint_meta: %s", e)

                # Rate-limited broadcast
                if now - last_send >= interval:
                    try:
                        positions = get_joint_positions_from_robot(
                            self._joint_names, self._robot
                        )
                        if positions is not None:
                            seq += 1
                            joint_data = {
                                "type": "joint_update",
                                "units": "radians",
                                "seq": seq,
                                "ts": now,
                                "joints": positions,
                            }
                            manager.broadcast_joint_data_sync(joint_data)
                            last_send = now
                    except Exception as e:
                        logger.error("Error broadcasting joint data: %s", e)

                time.sleep(0.001)
        except Exception as e:
            logger.error("JointBroadcaster thread crashed: %s", e)
        finally:
            with self._lock:
                self._running = False
                self._thread = None
                self._robot = None
