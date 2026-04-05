"""
NeuroLens WebSocket Handler
Real-time streaming inference with session-based analysis.
"""

import json
import uuid
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.analyzer import AnalysisOrchestrator
from app.utils.logger import logger

router = APIRouter()

# ── Shared orchestrator ───────────────────────────────────────────
orchestrator = AnalysisOrchestrator()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming analysis.

    Protocol:
    1. Client connects → receives session_id
    2. Client sends: {"text": "...", "explain": true}
    3. Server streams: progress updates → final result
    4. Client can send multiple messages in same session
    """
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, session_id)

    if not orchestrator._initialized:
        orchestrator.initialize()

    try:
        # Send session info
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "NeuroLens engine connected",
        })

        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            text = message.get("text", "")
            if not text:
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty text",
                })
                continue

            explain = message.get("explain", True)

            # Stream analysis phases
            try:
                # Phase 1: Acknowledge
                await websocket.send_json({
                    "type": "progress",
                    "phase": "received",
                    "progress": 0.05,
                    "message": "Text received, starting analysis...",
                })

                # Phase 2: Stream inference
                async for update in orchestrator.inference_engine.predict_streaming(text):
                    await websocket.send_json({
                        "type": "progress",
                        **update,
                    })
                    await asyncio.sleep(0.05)  # Small delay for visual effect

                # Phase 3: Full result
                result = await orchestrator.analyze(
                    text=text,
                    session_id=session_id,
                    explain=explain,
                )

                await websocket.send_json({
                    "type": "result",
                    "data": result,
                })

            except Exception as e:
                logger.error(f"Analysis error in WS: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Analysis failed: {str(e)}",
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)
