from __future__ import annotations

"""
Convenience launcher for the SO101 VLA demo server.

Usage:

    python -m so101_vla_demo.demo_script

This will:
- default to real hardware when it looks like SO101 is connected, otherwise
  fall back to mock robot mode (USE_MOCK_ROBOT=true),
- start the FastAPI/uvicorn server on http://localhost:8000,
- print instructions for opening the web UI.
"""

import os
import webbrowser
from pathlib import Path

import uvicorn


def _likely_so101_connected() -> bool:
    """
    Best-effort heuristic to decide whether to default to real hardware.

    If this returns True and USE_MOCK_ROBOT is not explicitly set, we default to
    real mode so the UI shows real camera streams on the robot machine.
    """
    port = (os.environ.get("SO101_PORT") or "").strip()
    if port and port.lower() not in {"auto", "auto-detect", "autodetect"}:
        try:
            if Path(port).exists():
                return True
        except Exception:
            pass

    if Path("/dev/serial/by-id").is_dir():
        try:
            if any(Path("/dev/serial/by-id").iterdir()):
                return True
        except Exception:
            pass

    dev = Path("/dev")
    for pattern in ("ttyACM*", "ttyUSB*"):
        try:
            if any(dev.glob(pattern)):
                return True
        except Exception:
            continue
    return False


def main() -> None:
    # Default to real mode on the robot machine, otherwise fall back to mock.
    if "USE_MOCK_ROBOT" not in os.environ:
        os.environ["USE_MOCK_ROBOT"] = "false" if _likely_so101_connected() else "true"

    host = os.environ.get("SO100_DEMO_HOST", "0.0.0.0")
    port_str = os.environ.get("SO100_DEMO_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    url = f"http://localhost:{port}/static/index.html"
    print(
        f"[so101_vla_demo] Starting server on {host}:{port} (mock robot = "
        f"{os.environ.get('USE_MOCK_ROBOT')})."
    )
    print(f"[so101_vla_demo] Open {url} in your browser.")
    if os.environ.get("USE_MOCK_ROBOT", "").lower() in {"1", "true", "yes"}:
        print("[so101_vla_demo] Tip: to use real hardware, run with USE_MOCK_ROBOT=false.")
    else:
        print("[so101_vla_demo] Tip: set SO101_PORT and SO101_CAMERA_SOURCES if auto config is wrong.")

    # Try to open the browser on local machines (no-op on headless).
    try:
        if host in {"127.0.0.1", "0.0.0.0", "localhost"}:
            webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("so101_vla_demo.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
