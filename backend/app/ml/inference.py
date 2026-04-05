"""
NeuroLens Inference Engine
Real-time prediction, batch processing, and streaming inference.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from app.config import settings, EMOTION_LABELS, DECEPTION_LABELS, MANIPULATION_LABELS
from app.ml.model import NeuroLensModel
from app.ml.features import FeatureExtractor
from app.ml.evaluator import ExplainabilityEngine, TokenImportanceEstimator
from app.utils.helpers import clean_text, generate_hash
from app.utils.logger import logger


class InferenceEngine:
    """
    Production inference engine for NeuroLens.
    Handles model loading, caching, single/batch/streaming predictions,
    and explainability generation.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = settings.device
        self.model: Optional[NeuroLensModel] = None
        self.tokenizer = None
        self.feature_extractor = FeatureExtractor()
        self.explainer = None
        self.importance_estimator = None
        self._loaded = False

        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str) -> None:
        """Load a trained model from checkpoint."""
        try:
            self.model = NeuroLensModel(
                meta_feature_dim=self.feature_extractor.get_feature_dim()
            )

            checkpoint_path = Path(path)
            if checkpoint_path.exists():
                checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Model loaded from {path}")
            else:
                logger.warning(f"No checkpoint found at {path}, using untrained model")

            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(settings.TRANSFORMER_MODEL)
            self.explainer = ExplainabilityEngine(self.tokenizer)
            self.importance_estimator = TokenImportanceEstimator(
                self.model, self.tokenizer, self.device
            )
            self._loaded = True
            logger.info("Inference engine ready")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Initialize with untrained model for demo purposes
            self._initialize_demo_model()

    def _initialize_demo_model(self) -> None:
        """Initialize model without pretrained weights (for demo/testing)."""
        try:
            self.model = NeuroLensModel(
                meta_feature_dim=self.feature_extractor.get_feature_dim()
            )
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(settings.TRANSFORMER_MODEL)
            self.explainer = ExplainabilityEngine(self.tokenizer)
            self.importance_estimator = TokenImportanceEstimator(
                self.model, self.tokenizer, self.device
            )
            self._loaded = True
            logger.info("Demo model initialized (untrained)")
        except Exception as e:
            logger.error(f"Failed to initialize demo model: {e}")

    @torch.no_grad()
    def predict(self, text: str, explain: bool = True) -> Dict[str, Any]:
        """
        Run full analysis on a single text.

        Returns:
            {
                "text": original text,
                "deception": {probability, verdict, confidence, reasons},
                "emotions": [{emotion, probability}, ...],
                "manipulation": {detected, probabilities, risk_level},
                "confidence_score": float,
                "token_importance": [...],
                "linguistic_insights": [...],
                "processing_time_ms": float,
            }
        """
        if not self._loaded:
            self._initialize_demo_model()

        start_time = time.time()

        # Preprocess
        cleaned = clean_text(text)

        # Extract meta features
        meta_features = self.feature_extractor.extract_meta_features(cleaned)
        meta_tensor = torch.tensor(
            list(meta_features.values()), dtype=torch.float
        ).unsqueeze(0).to(self.device)

        # Tokenize
        encoding = self.tokenizer(
            cleaned,
            max_length=settings.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask, meta_tensor)

        # Get tokens for explainability
        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0][:attention_mask[0].sum()]
        )

        # Generate explanation
        if explain and self.explainer:
            explanation = self.explainer.explain_prediction(
                text, outputs, tokens, meta_features
            )
        else:
            explanation = self._basic_prediction(outputs)

        # Compute overall confidence
        dec_probs = F.softmax(outputs["deception_logits"], dim=-1)[0]
        confidence = float(dec_probs.max())

        processing_time = (time.time() - start_time) * 1000

        return {
            "text": text,
            "text_hash": generate_hash(text),
            **explanation,
            "confidence_score": confidence,
            "meta_features": {k: round(v, 4) for k, v in meta_features.items()},
            "processing_time_ms": round(processing_time, 2),
        }

    def _basic_prediction(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate basic prediction without full explainability."""
        dec_probs = F.softmax(outputs["deception_logits"], dim=-1)[0]
        emo_probs = torch.sigmoid(outputs["emotion_logits"])[0]
        man_probs = F.softmax(outputs["manipulation_logits"], dim=-1)[0]

        return {
            "deception": {
                "probability": float(dec_probs[1]),
                "verdict": "deceptive" if dec_probs[1] > 0.5 else "truthful",
                "confidence": float(dec_probs.max()),
            },
            "emotions": sorted(
                [
                    {"emotion": label, "probability": float(emo_probs[i])}
                    for i, label in enumerate(EMOTION_LABELS) if i < len(emo_probs)
                ],
                key=lambda x: x["probability"],
                reverse=True,
            ),
            "manipulation": {
                "detected": MANIPULATION_LABELS[int(man_probs.argmax())],
                "probabilities": [
                    {"type": label, "probability": float(man_probs[i])}
                    for i, label in enumerate(MANIPULATION_LABELS) if i < len(man_probs)
                ],
            },
        }

    def predict_batch(
        self, texts: List[str], explain: bool = False
    ) -> List[Dict[str, Any]]:
        """Run batch prediction on multiple texts."""
        return [self.predict(text, explain=explain) for text in texts]

    async def predict_streaming(self, text: str):
        """
        Generator for streaming inference results.
        Yields partial results as they become available.
        """
        # Phase 1: Quick deception check
        yield {
            "phase": "preprocessing",
            "status": "analyzing_text",
            "progress": 0.1,
        }

        cleaned = clean_text(text)
        meta_features = self.feature_extractor.extract_meta_features(cleaned)

        yield {
            "phase": "features",
            "status": "extracting_features",
            "progress": 0.3,
            "meta_features": {k: round(v, 4) for k, v in list(meta_features.items())[:5]},
        }

        # Phase 2: Model inference
        result = self.predict(text, explain=True)

        yield {
            "phase": "inference",
            "status": "computing_predictions",
            "progress": 0.7,
            "deception": result.get("deception"),
        }

        # Phase 3: Full results
        yield {
            "phase": "complete",
            "status": "analysis_complete",
            "progress": 1.0,
            "result": result,
        }

    def get_token_importance(self, text: str, task: str = "deception") -> List[Dict]:
        """Get detailed token-level importance for explainability."""
        if self.importance_estimator:
            return self.importance_estimator.estimate(text, task)
        return []
