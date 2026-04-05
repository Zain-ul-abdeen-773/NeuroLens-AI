"""
NeuroLens Services - Analysis Orchestrator
Coordinates preprocessing, inference, and session-based behavioral profiling.
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict

from app.ml.inference import InferenceEngine
from app.ml.features import FeatureExtractor
from app.services.session import SessionManager
from app.services.cache import AnalysisCache
from app.utils.helpers import generate_hash, timestamp_now
from app.utils.logger import logger


class AnalysisOrchestrator:
    """
    Central orchestrator for text analysis.
    Manages inference, session memory, behavioral tracking, and anomaly detection.
    """

    def __init__(self):
        self.inference_engine = InferenceEngine()
        self.feature_extractor = FeatureExtractor()
        self.session_manager = SessionManager()
        self.cache = AnalysisCache()
        self._initialized = False

    def initialize(self, model_path: Optional[str] = None) -> None:
        """Initialize the orchestrator with model loading."""
        if model_path:
            self.inference_engine.load_model(model_path)
        else:
            self.inference_engine._initialize_demo_model()
        self._initialized = True
        logger.info("AnalysisOrchestrator initialized")

    async def analyze(
        self,
        text: str,
        session_id: Optional[str] = None,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """
        Full analysis pipeline for a single text.

        Args:
            text: Input text to analyze
            session_id: Optional session ID for behavioral tracking
            explain: Whether to include explainability

        Returns:
            Complete analysis result with predictions, explanations, and profiling
        """
        if not self._initialized:
            self.initialize()

        # Check cache
        text_hash = generate_hash(text)
        cached = self.cache.get(text_hash)
        if cached:
            logger.debug(f"Cache hit for {text_hash}")
            return cached

        # Run prediction
        result = self.inference_engine.predict(text, explain=explain)
        result["timestamp"] = timestamp_now()
        result["session_id"] = session_id

        # Session-based profiling
        if session_id:
            self.session_manager.add_entry(session_id, result)
            profile = self.session_manager.get_profile(session_id)
            result["session_profile"] = profile

            # Anomaly detection
            anomalies = self._detect_anomalies(session_id, result)
            if anomalies:
                result["anomalies"] = anomalies

        # Cache result
        self.cache.set(text_hash, result)

        return result

    async def batch_analyze(
        self, texts: List[str], session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Batch analysis for multiple texts."""
        results = []
        for text in texts:
            result = await self.analyze(text, session_id, explain=False)
            results.append(result)
        return results

    def _detect_anomalies(
        self, session_id: str, current: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Detect behavioral anomalies by comparing current analysis
        with session history.
        """
        anomalies = []
        history = self.session_manager.get_history(session_id)

        if len(history) < 3:
            return anomalies

        # Check for sudden deception probability spike
        dec_probs = [
            h.get("deception", {}).get("probability", 0) for h in history[-5:]
        ]
        current_dec = current.get("deception", {}).get("probability", 0)

        if dec_probs and current_dec > 0.7 and all(p < 0.4 for p in dec_probs):
            anomalies.append({
                "type": "deception_spike",
                "message": "Sudden increase in deception probability detected",
                "severity": "high",
            })

        # Check for emotional instability
        recent_emotions = [
            h.get("dominant_emotions", [{}])[0].get("emotion", "neutral")
            for h in history[-5:]
            if h.get("dominant_emotions")
        ]
        if len(set(recent_emotions)) > 3:
            anomalies.append({
                "type": "emotional_instability",
                "message": "Rapid emotional state changes detected across messages",
                "severity": "medium",
            })

        # Check for manipulation pattern escalation
        man_types = [
            h.get("manipulation", {}).get("detected")
            for h in history[-5:]
            if h.get("manipulation", {}).get("detected")
        ]
        if len(man_types) >= 3:
            anomalies.append({
                "type": "manipulation_pattern",
                "message": f"Recurring manipulation pattern detected: {', '.join(set(man_types))}",
                "severity": "critical",
            })

        return anomalies

    def get_session_timeline(self, session_id: str) -> List[Dict[str, Any]]:
        """Get behavioral timeline for a session."""
        history = self.session_manager.get_history(session_id)
        timeline = []

        for entry in history:
            timeline.append({
                "timestamp": entry.get("timestamp"),
                "deception_probability": entry.get("deception", {}).get("probability", 0),
                "dominant_emotion": (
                    entry.get("dominant_emotions", [{}])[0].get("emotion", "neutral")
                    if entry.get("dominant_emotions") else "neutral"
                ),
                "manipulation_detected": entry.get("manipulation", {}).get("detected"),
                "confidence": entry.get("confidence_score", 0),
            })

        return timeline
