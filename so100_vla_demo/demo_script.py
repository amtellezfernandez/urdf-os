from __future__ import annotations

"""
Convenience launcher for the SO100 VLA demo server.

Usage:

    python -m so100_vla_demo.demo_script

This will:
- default to mock robot mode (USE_MOCK_ROBOT=true) so you can run the demo
  without SO100 hardware,
- start the FastAPI/uvicorn server on http://localhost:8000,
- print instructions for opening the web UI.
"""

import os
import webbrowser

import uvicorn


def main() -> None:
    # Default to mock robot mode so judges / teammates can run the demo anywhere.
    os.environ.setdefault("USE_MOCK_ROBOT", "true")

    host = os.environ.get("SO100_DEMO_HOST", "0.0.0.0")
    port_str = os.environ.get("SO100_DEMO_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    url = f"http://localhost:{port}/static/index.html"
    print(
        f"[so100_vla_demo] Starting server on {host}:{port} (mock robot = "
        f"{os.environ.get('USE_MOCK_ROBOT')})."
    )
    print(f"[so100_vla_demo] Open {url} in your browser.")

    # Try to open the browser on local machines (no-op on headless).
    try:
        if host in {"127.0.0.1", "0.0.0.0", "localhost"}:
            webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("so100_vla_demo.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

