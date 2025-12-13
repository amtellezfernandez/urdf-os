import asyncio
import glob
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import (
    FOLLOWER_CONFIG_PATH,
    LEADER_CONFIG_PATH,
    detect_port_after_disconnect,
    find_available_ports,
    find_robot_port,
    get_default_robot_port,
    get_saved_robot_port,
)
from . import config
from .features.calibrate import CalibrationRequest, calibration_manager
from .features.teleoperate import TeleoperateRequest, TeleoperationManager
from .services.websocket_manager import manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.teleop = TeleoperationManager()
    try:
        yield
    finally:
        logger.info("FastAPI shutting down, cleaning up background tasks")
        if manager:
            manager.stop_broadcast_thread()
        logger.info("Cleanup completed")


def get_teleop_manager(request: Request) -> TeleoperationManager:
    return request.app.state.teleop


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify server is running."""
    return {"status": "ok", "message": "FastAPI server is running"}


@app.post("/teleop/start")
def teleoperate_arm(
    req: TeleoperateRequest, m: TeleoperationManager = Depends(get_teleop_manager)
):
    """Start teleoperation of the robot arm."""
    return m.start(req)


@app.post("/teleop/stop")
def stop_teleoperation(m: TeleoperationManager = Depends(get_teleop_manager)):
    """Stop the current teleoperation session."""
    return m.stop()


@app.get("/teleop/status")
def teleoperation_status(m: TeleoperationManager = Depends(get_teleop_manager)):
    """Get the current teleoperation status."""
    return m.status()


@app.websocket("/ws/joint-data")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await manager.connect(websocket)
        logger.info("WebSocket connection established")

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                logger.debug("Received WebSocket message: %s", data)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        manager.disconnect(websocket)
        logger.info("WebSocket connection cleaned up")


@app.post("/start-calibration")
def start_calibration(request: CalibrationRequest):
    """Start calibration process."""
    return calibration_manager.start_calibration(request)


@app.post("/stop-calibration")
def stop_calibration():
    """Stop calibration process."""
    return calibration_manager.stop_calibration_process()


@app.get("/calibration-status")
def calibration_status():
    """Get current calibration status."""
    from dataclasses import asdict

    status = calibration_manager.get_status()
    return asdict(status)


@app.post("/complete-calibration-step")
def complete_calibration_step():
    """Complete the current calibration step."""
    return calibration_manager.complete_step()


@app.get("/calibration-configs/{device_type}")
def get_calibration_configs(device_type: str):
    """Get all calibration config files for a specific device type."""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        configs = []
        if os.path.exists(config_path):
            for file in os.listdir(config_path):
                if file.endswith(".json"):
                    config_name = os.path.splitext(file)[0]
                    file_path = os.path.join(config_path, file)
                    file_size = os.path.getsize(file_path)
                    modified_time = os.path.getmtime(file_path)

                    configs.append(
                        {
                            "name": config_name,
                            "filename": file,
                            "size": file_size,
                            "modified": modified_time,
                        }
                    )

        return {"success": True, "configs": configs, "device_type": device_type}

    except Exception as e:
        logger.error("Error getting calibration configs: %s", e)
        return {"success": False, "message": str(e)}


@app.delete("/calibration-configs/{device_type}/{config_name}")
def delete_calibration_config(device_type: str, config_name: str):
    """Delete a calibration config file."""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        filename = f"{config_name}.json"
        file_path = os.path.join(config_path, filename)

        if not os.path.exists(file_path):
            return {"success": False, "message": "Configuration file not found"}

        os.remove(file_path)
        logger.info("Deleted calibration config: %s", file_path)

        return {
            "success": True,
            "message": f"Configuration '{config_name}' deleted successfully",
        }

    except Exception as e:
        logger.error("Error deleting calibration config: %s", e)
        return {"success": False, "message": str(e)}


@app.get("/get-configs")
def get_configs():
    leader_configs = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(LEADER_CONFIG_PATH, "*.json"))
    ]
    follower_configs = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(FOLLOWER_CONFIG_PATH, "*.json"))
    ]

    return {"leader_configs": leader_configs, "follower_configs": follower_configs}


@app.post("/save-robot-config")
def save_robot_config_endpoint(data: dict):
    """Save a robot configuration for future use."""
    try:
        robot_type = data.get("robot_type")
        config_name = data.get("config_name")

        if not robot_type or not config_name:
            return {"status": "error", "message": "Missing robot_type or config_name"}

        success = config.save_robot_config(robot_type, config_name)

        if success:
            return {
                "status": "success",
                "message": f"Configuration saved for {robot_type}",
            }
        return {"status": "error", "message": "Failed to save configuration"}

    except Exception as e:
        logger.error("Error saving robot configuration: %s", e)
        return {"status": "error", "message": str(e)}


@app.get("/robot-config/{robot_type}")
def get_robot_config(robot_type: str, available_configs: str = ""):
    """Get the saved configuration for a robot type."""
    try:
        available_configs_list = []
        if available_configs:
            available_configs_list = [
                cfg.strip() for cfg in available_configs.split(",") if cfg.strip()
            ]

        saved_config = config.get_saved_robot_config(robot_type)
        default_config = config.get_default_robot_config(
            robot_type, available_configs_list
        )

        return {
            "status": "success",
            "saved_config": saved_config,
            "default_config": default_config,
        }
    except Exception as e:
        logger.error("Error getting robot configuration: %s", e)
        return {"status": "error", "message": str(e)}


@app.get("/setup/ports")
def get_available_ports():
    """Get all available serial ports."""
    try:
        ports = find_available_ports()
        return {"status": "success", "ports": ports}
    except Exception as e:
        logger.error("Error getting available ports: %s", e)
        return {"status": "error", "message": str(e)}


@app.post("/start-port-detection")
def start_port_detection(data: dict):
    """Start port detection process for a robot."""
    try:
        robot_type = data.get("robot_type", "robot")
        result = find_robot_port(robot_type)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error("Error starting port detection: %s", e)
        return {"status": "error", "message": str(e)}


@app.post("/detect-port-after-disconnect")
def detect_port_after_disconnect_endpoint(data: dict):
    """Detect port after disconnection."""
    try:
        ports_before = data.get("ports_before", [])
        detected_port = detect_port_after_disconnect(ports_before)
        return {"status": "success", "port": detected_port}
    except Exception as e:
        logger.error("Error detecting port: %s", e)
        return {"status": "error", "message": str(e)}


@app.get("/robot-port/{robot_type}")
def get_robot_port(robot_type: str):
    """Get the saved port for a robot type."""
    try:
        saved_port = get_saved_robot_port(robot_type)
        default_port = get_default_robot_port(robot_type)
        return {
            "status": "success",
            "saved_port": saved_port,
            "default_port": default_port,
        }
    except Exception as e:
        logger.error("Error getting robot port: %s", e)
        return {"status": "error", "message": str(e)}
