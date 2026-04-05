"""
NeuroLens Session Manager
Manages user conversation history and behavioral profiling over time.
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from app.config import settings
from app.utils.logger import logger


@dataclass
class SessionData:
    """Container for a single user session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    In-memory session store for user conversation history.
    Tracks behavioral patterns and builds personality profiles over time.
    """

    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.max_history = settings.MAX_SESSION_HISTORY
        self.expiry = settings.SESSION_EXPIRY_SECONDS

    def get_or_create(self, session_id: str) -> SessionData:
        """Get existing session or create a new one."""
        self._cleanup_expired()

        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData(session_id=session_id)
            logger.debug(f"New session created: {session_id}")

        session = self.sessions[session_id]
        session.last_active = time.time()
        return session

    def add_entry(self, session_id: str, analysis_result: Dict[str, Any]) -> None:
        """Add an analysis result to session history."""
        session = self.get_or_create(session_id)

        # Keep history bounded
        if len(session.history) >= self.max_history:
            session.history.pop(0)

        session.history.append(analysis_result)
        session.profile = self._build_profile(session.history)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get analysis history for a session."""
        if session_id in self.sessions:
            return self.sessions[session_id].history
        return []

    def get_profile(self, session_id: str) -> Dict[str, Any]:
        """Get behavioral profile for a session."""
        if session_id in self.sessions:
            return self.sessions[session_id].profile
        return {}

    def _build_profile(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a behavioral profile from conversation history.
        Tracks personality traits, communication patterns, and risk indicators.
        """
        if not history:
            return {}

        # Aggregate deception signals
        dec_probs = [
            h.get("deception", {}).get("probability", 0) for h in history
        ]

        # Aggregate emotions
        emotion_counts = defaultdict(int)
        for h in history:
            for e in h.get("dominant_emotions", []):
                emotion_counts[e.get("emotion", "neutral")] += 1

        # Aggregate manipulation patterns
        manipulation_counts = defaultdict(int)
        for h in history:
            man_type = h.get("manipulation", {}).get("detected")
            if man_type:
                manipulation_counts[man_type] += 1

        # Communication style indicators
        avg_confidence = (
            sum(h.get("confidence_score", 0) for h in history) / len(history)
        )

        return {
            "message_count": len(history),
            "avg_deception_probability": round(sum(dec_probs) / len(dec_probs), 3),
            "max_deception_probability": round(max(dec_probs), 3),
            "deception_trend": self._compute_trend(dec_probs),
            "dominant_emotions": dict(
                sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "emotional_diversity": len(emotion_counts),
            "manipulation_patterns": dict(manipulation_counts),
            "avg_confidence": round(avg_confidence, 3),
            "risk_level": self._assess_risk(dec_probs, manipulation_counts),
            "personality_indicators": self._infer_personality(
                emotion_counts, dec_probs
            ),
        }

    @staticmethod
    def _compute_trend(values: List[float]) -> str:
        """Compute trend direction from a series of values."""
        if len(values) < 3:
            return "insufficient_data"

        recent = values[-3:]
        older = values[-6:-3] if len(values) >= 6 else values[:3]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        diff = recent_avg - older_avg
        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        return "stable"

    @staticmethod
    def _assess_risk(
        dec_probs: List[float], manipulation_counts: Dict[str, int]
    ) -> str:
        """Assess overall risk level from behavioral patterns."""
        avg_dec = sum(dec_probs) / max(len(dec_probs), 1)
        total_manipulations = sum(manipulation_counts.values())

        if avg_dec > 0.7 or total_manipulations > 5:
            return "critical"
        elif avg_dec > 0.5 or total_manipulations > 2:
            return "high"
        elif avg_dec > 0.3 or total_manipulations > 0:
            return "moderate"
        return "low"

    @staticmethod
    def _infer_personality(
        emotion_counts: Dict[str, int],
        dec_probs: List[float],
    ) -> Dict[str, str]:
        """
        Infer Big Five personality traits approximation
        from behavioral patterns.
        """
        total_emotions = sum(emotion_counts.values()) or 1
        personality = {}

        # Openness: emotional diversity
        diversity_ratio = len(emotion_counts) / max(total_emotions, 1)
        personality["openness"] = "high" if diversity_ratio > 0.5 else "moderate"

        # Agreeableness: positive vs negative emotions
        positive = sum(
            emotion_counts.get(e, 0)
            for e in ["admiration", "approval", "caring", "gratitude", "love", "joy"]
        )
        negative = sum(
            emotion_counts.get(e, 0)
            for e in ["anger", "annoyance", "disgust", "disapproval"]
        )
        if positive > negative * 2:
            personality["agreeableness"] = "high"
        elif negative > positive:
            personality["agreeableness"] = "low"
        else:
            personality["agreeableness"] = "moderate"

        # Neuroticism: anxiety/fear/sadness signals
        neurotic = sum(
            emotion_counts.get(e, 0)
            for e in ["fear", "nervousness", "sadness", "grief", "embarrassment"]
        )
        personality["neuroticism"] = (
            "high" if neurotic / total_emotions > 0.3
            else "moderate" if neurotic / total_emotions > 0.1
            else "low"
        )

        return personality

    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_active > self.expiry
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.debug(f"Session expired: {sid}")
