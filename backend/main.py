"""
FedFraud Backend — FastAPI Application Entry Point
WebSocket + REST API for real-time federated learning dashboard.
"""

import logging
import json
import asyncio
from typing import Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, set_broadcast_fn
from services.fl_service import experiment_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── WebSocket Connection Manager ─────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections for real-time broadcasting."""

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info(f"WS connected | total={len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info(f"WS disconnected | total={len(self.active)}")

    async def broadcast(self, payload: dict):
        """Send payload to all connected clients."""
        if not self.active:
            return
        message = json.dumps(payload)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


manager = ConnectionManager()


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wire up WebSocket broadcast to training service
    set_broadcast_fn(manager.broadcast)
    logger.info("🏦 FedFraud backend ready")
    yield
    logger.info("FedFraud backend shutting down")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="FedFraud API",
    description="Privacy-preserving federated fraud detection backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send current state on connect
    try:
        await ws.send_text(json.dumps({
            "type": "init",
            "state": experiment_state.to_dict(),
        }))
    except Exception:
        pass

    try:
        while True:
            # Keep connection alive; client may send pings
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.warning(f"WS error: {e}")
        manager.disconnect(ws)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "FedFraud API v2.0"}
