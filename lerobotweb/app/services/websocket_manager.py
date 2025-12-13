import logging
import queue
import threading
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.broadcast_queue = queue.Queue()
        self.broadcast_thread = None
        self.is_running = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            "WebSocket connected. Total connections: %d",
            len(self.active_connections),
        )

        if not self.is_running:
            self.start_broadcast_thread()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                "WebSocket disconnected. Total connections: %d",
                len(self.active_connections),
            )

        if not self.active_connections and self.is_running:
            self.stop_broadcast_thread()

    def start_broadcast_thread(self):
        """Start the background thread for broadcasting data."""
        if self.is_running:
            return

        self.is_running = True
        self.broadcast_thread = threading.Thread(
            target=self._broadcast_worker, daemon=True
        )
        self.broadcast_thread.start()
        logger.info("Broadcast thread started")

    def stop_broadcast_thread(self):
        """Stop the background thread."""
        self.is_running = False
        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=1.0)
            logger.info("Broadcast thread stopped")

    def _broadcast_worker(self):
        """Background worker thread for broadcasting WebSocket data."""
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.is_running:
                try:
                    data = self.broadcast_queue.get(timeout=0.1)
                    if data is None:
                        break

                    if self.active_connections:
                        loop.run_until_complete(self._send_to_all_connections(data))

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error("Error in broadcast worker: %s", e)

        finally:
            loop.close()

    async def _send_to_all_connections(self, data: dict[str, Any]):
        """Send data to all active WebSocket connections."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error("Error sending data to WebSocket: %s", e)
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

    def broadcast_joint_data_sync(self, data: dict[str, Any]):
        """Thread-safe method to queue data for broadcasting."""
        if self.is_running and self.active_connections:
            try:
                self.broadcast_queue.put_nowait(data)
            except queue.Full:
                logger.warning("Broadcast queue is full, dropping data")


manager = ConnectionManager()
