"""
NeuroLens Configuration Module
Centralized settings using Pydantic for type-safe configuration management.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from enum import Enum
import torch


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class Settings(BaseSettings):
    """Application-wide configuration."""

    # ── API Settings ──────────────────────────────────────────────
    APP_NAME: str = "NeuroLens AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Model Settings ────────────────────────────────────────────
    TRANSFORMER_MODEL: str = "microsoft/deberta-v3-base"
    MAX_SEQ_LENGTH: int = 512
    HIDDEN_DIM: int = 256
    LSTM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.3
    NUM_EMOTIONS: int = 28  # GoEmotions label count
    META_FEATURE_DIM: int = 64
    DROPOUT_RATE: float = 0.3

    # ── Training Settings ─────────────────────────────────────────
    LEARNING_RATE: float = 2e-5
    WEIGHT_DECAY: float = 0.01
    BATCH_SIZE: int = 16
    EPOCHS: int = 20
    GRADIENT_ACCUMULATION_STEPS: int = 4
    MAX_GRAD_NORM: float = 1.0
    WARMUP_RATIO: float = 0.1
    LABEL_SMOOTHING: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 5
    K_FOLDS: int = 5
    USE_MIXED_PRECISION: bool = True

    # ── Device Settings ───────────────────────────────────────────
    DEVICE_TYPE: DeviceType = DeviceType.AUTO

    # ── Paths ─────────────────────────────────────────────────────
    MODEL_SAVE_DIR: str = "./models"
    LOG_DIR: str = "./logs"
    DATA_DIR: str = "./data"
    TENSORBOARD_DIR: str = "./runs"

    # ── Cache Settings ────────────────────────────────────────────
    CACHE_MAX_SIZE: int = 1000
    CACHE_TTL_SECONDS: int = 3600

    # ── Session Settings ──────────────────────────────────────────
    MAX_SESSION_HISTORY: int = 100
    SESSION_EXPIRY_SECONDS: int = 7200

    # ── Optuna Settings ───────────────────────────────────────────
    OPTUNA_N_TRIALS: int = 50
    OPTUNA_TIMEOUT: Optional[int] = 3600

    @property
    def device(self) -> torch.device:
        if self.DEVICE_TYPE == DeviceType.AUTO:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.DEVICE_TYPE.value)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# ── Emotion Labels (GoEmotions taxonomy) ──────────────────────────
EMOTION_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# ── Deception Labels ─────────────────────────────────────────────
DECEPTION_LABELS: List[str] = ["truthful", "deceptive"]

# ── Manipulation Labels ──────────────────────────────────────────
MANIPULATION_LABELS: List[str] = [
    "none", "guilt_tripping", "gaslighting", "love_bombing",
    "fear_mongering", "flattery", "coercion"
]

# ── LIWC-Style Feature Categories ────────────────────────────────
LIWC_CATEGORIES = {
    "certainty_words": [
        "always", "never", "definitely", "certainly", "absolutely",
        "clearly", "obviously", "undoubtedly", "surely", "totally"
    ],
    "hedging_words": [
        "maybe", "perhaps", "possibly", "might", "could",
        "somewhat", "sort of", "kind of", "apparently", "supposedly"
    ],
    "negative_emotion": [
        "hate", "angry", "sad", "fear", "terrible",
        "horrible", "awful", "disgusting", "miserable", "furious"
    ],
    "positive_emotion": [
        "love", "happy", "wonderful", "great", "amazing",
        "fantastic", "excellent", "beautiful", "brilliant", "joyful"
    ],
    "deception_markers": [
        "honestly", "truthfully", "believe me", "trust me", "frankly",
        "to be honest", "i swear", "seriously", "no lie", "for real"
    ],
    "distancing_language": [
        "that person", "the thing", "one might", "it happened",
        "people say", "they did", "someone", "something"
    ],
    "power_words": [
        "must", "need", "should", "have to", "demand",
        "require", "insist", "command", "force", "control"
    ],
    "first_person_singular": ["i", "me", "my", "mine", "myself"],
    "first_person_plural": ["we", "us", "our", "ours", "ourselves"],
    "third_person": ["he", "she", "they", "him", "her", "them"],
}

settings = Settings()
