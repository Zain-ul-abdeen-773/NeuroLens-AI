"""
NeuroLens FastAPI Application
Main application entry point with CORS, middleware, and lifespan management.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.routes import router as api_router, orchestrator
from app.api.websocket import router as ws_router
from app.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════
# Application Lifespan
# ═══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # ── Startup ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  🧠 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"  Device: {settings.device}")
    logger.info(f"  Debug: {settings.DEBUG}")
    logger.info("=" * 60)

    # Initialize the analysis engine
    orchestrator.initialize()
    logger.info("Analysis engine initialized")

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Shutting down NeuroLens...")
    orchestrator.cache.clear()


# ═══════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI Behavioral Intelligence Engine — Analyze text for deception, "
        "emotional intent, manipulation patterns, and psychological states."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Timing Middleware ─────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(duration, 2))
    return response


# ── Include Routers ───────────────────────────────────────────────
app.include_router(api_router)
app.include_router(ws_router)


# ── Root Endpoint ─────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "AI Behavioral Intelligence Engine",
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "batch_analyze": "/api/v1/batch-analyze",
            "train": "/api/v1/train",
            "metrics": "/api/v1/metrics",
            "websocket": "/ws/analyze",
            "health": "/api/v1/health",
            "docs": "/docs",
        },
    }
