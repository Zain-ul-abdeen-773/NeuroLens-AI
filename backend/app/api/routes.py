"""
NeuroLens API Routes
REST endpoints for text analysis, batch processing, training, and metrics.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.services.analyzer import AnalysisOrchestrator
from app.utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# ── Shared orchestrator instance ──────────────────────────────────
orchestrator = AnalysisOrchestrator()


# ═══════════════════════════════════════════════════════════════════
# Request / Response Schemas
# ═══════════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    session_id: Optional[str] = Field(None, description="Session ID for behavioral tracking")
    explain: bool = Field(True, description="Include explainability layer")

class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    session_id: Optional[str] = None

class TrainRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to training dataset")
    dataset_type: str = Field("custom", description="Dataset type: liar, goemotions, custom")
    epochs: Optional[int] = Field(None, ge=1, le=100)
    learning_rate: Optional[float] = Field(None, gt=0, lt=1)
    batch_size: Optional[int] = Field(None, ge=1, le=128)

class TimelineRequest(BaseModel):
    session_id: str

class AnalysisResponse(BaseModel):
    text: str
    deception: dict
    emotions: list
    manipulation: dict
    confidence_score: float
    processing_time_ms: float
    session_id: Optional[str] = None

    class Config:
        extra = "allow"


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════

@router.post("/analyze", response_model=None)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze a single text for deception, emotion, and manipulation.

    Returns full analysis with:
    - Deception probability and verdict
    - Multi-label emotion spectrum
    - Manipulation detection
    - Linguistic explanations
    - Token importance heatmap
    """
    try:
        result = await orchestrator.analyze(
            text=request.text,
            session_id=request.session_id,
            explain=request.explain,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch-analyze")
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    Analyze multiple texts in a single request.
    Optimized for throughput with minimal per-request overhead.
    """
    try:
        results = await orchestrator.batch_analyze(
            texts=request.texts,
            session_id=request.session_id,
        )
        return JSONResponse(content={
            "results": results,
            "total": len(results),
        })
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def trigger_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger a training run in the background.
    Returns immediately with a job ID for status polling.
    """
    from app.ml.preprocessing import load_custom_dataset, load_liar_dataset, NeuroLensDataset
    from app.ml.model import NeuroLensModel
    from app.ml.trainer import NeuroLensTrainer
    from app.ml.preprocessing import create_dataloader

    try:
        config = {}
        if request.epochs:
            config["epochs"] = request.epochs
        if request.learning_rate:
            config["learning_rate"] = request.learning_rate

        def run_training():
            logger.info("Background training started")
            try:
                # Load data
                if request.dataset_type == "liar":
                    texts, labels = load_liar_dataset(request.dataset_path)
                    dataset = NeuroLensDataset(
                        texts=texts,
                        deception_labels=labels,
                        augment=True,
                    )
                else:
                    data = load_custom_dataset(request.dataset_path)
                    dataset = NeuroLensDataset(
                        texts=data["texts"],
                        deception_labels=data.get("deception_labels"),
                        emotion_labels=data.get("emotion_labels"),
                        manipulation_labels=data.get("manipulation_labels"),
                        augment=True,
                    )

                dataloader = create_dataloader(dataset)
                model = NeuroLensModel()
                trainer = NeuroLensTrainer(model, dataloader, config=config)
                result = trainer.train()
                trainer.save_model()
                logger.info(f"Training complete: {result}")
            except Exception as e:
                logger.error(f"Training failed: {e}", exc_info=True)

        background_tasks.add_task(run_training)

        return JSONResponse(content={
            "status": "training_started",
            "message": "Training is running in the background",
            "config": config,
        })
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """
    Get current model performance metrics and system stats.
    """
    cache_stats = orchestrator.cache.stats()

    return JSONResponse(content={
        "model": {
            "name": "NeuroLens Hybrid v1.0",
            "architecture": "DeBERTa + BiLSTM + Dense",
            "tasks": ["deception", "emotion", "manipulation"],
            "status": "loaded" if orchestrator._initialized else "not_loaded",
        },
        "cache": cache_stats,
        "sessions": {
            "active": len(orchestrator.session_manager.sessions),
        },
    })


@router.post("/timeline")
async def get_timeline(request: TimelineRequest):
    """
    Get behavioral timeline for a specific session.
    Shows how deception, emotion, and manipulation signals evolve over time.
    """
    timeline = orchestrator.get_session_timeline(request.session_id)
    profile = orchestrator.session_manager.get_profile(request.session_id)

    return JSONResponse(content={
        "session_id": request.session_id,
        "timeline": timeline,
        "profile": profile,
    })


@router.get("/health")
async def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "engine": "NeuroLens AI v1.0"}
