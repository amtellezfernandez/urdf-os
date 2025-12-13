import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# From lerobot
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

# From app
from app.config import setup_calibration_files
from app.types.so_arm import TeleoperateRequest
from app.features.joint_data import get_joint_positions_from_robot
from app.services.websocket_manager import manager

logger = logging.getLogger(__name__)


class TeleoperationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._robot = None
        self._leader = None
        self._active = False

    def start(self, request: TeleoperateRequest):
        with self._lock:
            if self._active:
                return {
                    "success": False,
                    "message": "Teleoperation is already active",
                }
            self._stop.clear()

            # Prepare calibration and device configs
            try:
                logger.info(
                    f"Starting teleoperation with leader port: {request.leader_port}, follower port: {request.follower_port}"
                )

                leader_config_name, follower_config_name = setup_calibration_files(
                    request.leader_config, request.follower_config
                )

                follower_config = SO101FollowerConfig(
                    port=request.follower_port,
                    id=follower_config_name,
                )

                leader_config = SO101LeaderConfig(
                    port=request.leader_port,
                    id=leader_config_name,
                )
            except Exception as e:
                logger.error(f"Failed to prepare teleoperation configs: {e}")
                return {
                    "success": False,
                    "message": f"Failed to start teleoperation: {str(e)}",
                }

            def worker():
                try:
                    self._active = True

                    logger.info("Initializing leader and follower device...")
                    follower = SO101Follower(follower_config)
                    leader = SO101Leader(leader_config)

                    self._robot = follower
                    self._leader = leader

                    follower.bus.connect()
                    leader.bus.connect()

                    # Write calibration to motors' memory
                    follower.bus.write_calibration(follower.calibration)
                    leader.bus.write_calibration(leader.calibration)

                    # Connect cameras and configure motors
                    # logger.info("Connecting cameras and configuring motors...")
                    # for cam in follower.cameras.values():
                    #     cam.connect()
                    follower.configure()
                    leader.configure()

                    logger.info("Starting teleoperation loop...")
                    try:
                        last_broadcast_time = 0.0
                        broadcast_interval = 0.05  # 20 Hz
                        seq_counter = 0
                        meta_sent = False

                        while not self._stop.is_set():
                            action = leader.get_action()
                            follower.send_action(action)

                            # One-time metadata send (names, units)
                            if not meta_sent:
                                try:
                                    initial_positions = get_joint_positions_from_robot(
                                        [], follower
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
                                    logger.error(f"Failed to send joint_meta: {e}")

                            now = time.time()
                            if now - last_broadcast_time >= broadcast_interval:
                                try:
                                    positions = get_joint_positions_from_robot(
                                        [], follower
                                    )
                                    if positions is not None:
                                        seq_counter += 1
                                        joint_data = {
                                            "type": "joint_update",
                                            "units": "radians",
                                            "seq": seq_counter,
                                            "ts": now,
                                            "joints": positions,
                                        }
                                        manager.broadcast_joint_data_sync(joint_data)
                                        last_broadcast_time = now
                                except Exception as e:
                                    logger.error(f"Error broadcasting joint data: {e}")

                            # Small delay to prevent excessive CPU usage
                            time.sleep(0.001)
                    finally:
                        follower.disconnect()
                        leader.disconnect()
                        logger.info("Teleoperation stopped")

                except Exception as e:
                    logger.error(f"Error during teleoperation: {e}")
                finally:
                    self._active = False
                    self._robot = None
                    self._leader = None

            self._future = self._executor.submit(worker)
            return {"success": True, "message": "Teleoperation started"}

    def stop(self):
        with self._lock:
            if not self._active:
                return {"success": False, "message": "Teleoperation is not active"}
            self._stop.set()
            if self._future:
                try:
                    self._future.cancel()
                except Exception as e:
                    logger.error(f"Error stopping teleoperation: {e}")
                    pass
            return {
                "success": True,
                "message": "Teleoperation stopped",
            }

    def status(self):
        return {"teleoperation_active": self._active}
