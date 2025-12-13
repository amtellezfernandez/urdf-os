"""Launcher for the backend API, runs the FastAPI server with uvicorn."""

import logging

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Start the server."""
    logger.info("Starting lerobotweb server...")
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )


if __name__ == "__main__":
    main()
