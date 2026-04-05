"""
NeuroLens Evaluation & Explainability Module
Metrics computation, SHAP values, attention visualization, and calibration analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

from app.config import EMOTION_LABELS, DECEPTION_LABELS, MANIPULATION_LABELS
from app.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════
# Classification Metrics
# ═══════════════════════════════════════════════════════════════════

class MetricsComputer:
    """Compute comprehensive classification metrics for all tasks."""

    @staticmethod
    def compute_deception_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute deception detection metrics."""
        metrics = {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_per_class": {
                label: float(score)
                for label, score in zip(
                    DECEPTION_LABELS,
                    f1_score(y_true, y_pred, average=None, zero_division=0),
                )
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=DECEPTION_LABELS, output_dict=True,
                zero_division=0,
            ),
        }

        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            except (ValueError, IndexError):
                metrics["roc_auc"] = None

        return metrics

    @staticmethod
    def compute_emotion_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute multi-label emotion classification metrics."""
        metrics = {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "f1_per_emotion": {},
        }

        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, label in enumerate(EMOTION_LABELS):
            if i < len(per_label_f1):
                metrics["f1_per_emotion"][label] = float(per_label_f1[i])

        return metrics

    @staticmethod
    def compute_manipulation_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute manipulation detection metrics."""
        metrics = {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                )
            except ValueError:
                metrics["roc_auc_ovr"] = None

        return metrics

    @staticmethod
    def compute_calibration(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute calibration curve data for reliability diagrams."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_true_freqs = []
        bin_counts = []

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(float(y_prob[mask].mean()))
                bin_true_freqs.append(float(y_true[mask].mean()))
                bin_counts.append(int(mask.sum()))

        # Expected Calibration Error
        total = sum(bin_counts)
        ece = sum(
            (count / total) * abs(mean - freq)
            for mean, freq, count in zip(bin_means, bin_true_freqs, bin_counts)
        ) if total > 0 else 0

        return {
            "bin_means": bin_means,
            "bin_true_frequencies": bin_true_freqs,
            "bin_counts": bin_counts,
            "ece": float(ece),
        }


# ═══════════════════════════════════════════════════════════════════
# Explainability Engine
# ═══════════════════════════════════════════════════════════════════

class ExplainabilityEngine:
    """
    Generates human-readable explanations for model predictions.
    Uses attention weights, feature importance, and linguistic pattern analysis.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def explain_prediction(
        self,
        text: str,
        outputs: Dict[str, torch.Tensor],
        tokens: List[str],
        meta_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.

        Returns attention-based token importance, top contributing features,
        and natural language reasoning.
        """
        explanation = {}

        # ── Token importance from attention weights ───────────────
        if "attention_weights" in outputs:
            attn = outputs["attention_weights"].detach().cpu().numpy()
            if attn.ndim == 2:
                attn = attn[0]  # Take first batch element

            # Map attention weights to tokens
            token_importance = []
            for i, (token, weight) in enumerate(zip(tokens, attn[:len(tokens)])):
                token_importance.append({
                    "token": token,
                    "importance": float(weight),
                    "index": i,
                })

            # Sort by importance
            token_importance.sort(key=lambda x: x["importance"], reverse=True)
            explanation["token_importance"] = token_importance
            explanation["top_tokens"] = token_importance[:10]

        # ── Deception reasoning ───────────────────────────────────
        if "deception_logits" in outputs:
            probs = F.softmax(outputs["deception_logits"], dim=-1).detach().cpu()
            if probs.ndim == 2:
                probs = probs[0]
            deception_prob = float(probs[1]) if len(probs) > 1 else 0.0

            reasons = self._generate_deception_reasons(
                text, deception_prob, meta_features
            )
            explanation["deception"] = {
                "probability": deception_prob,
                "verdict": "deceptive" if deception_prob > 0.5 else "truthful",
                "confidence": float(max(probs)),
                "reasons": reasons,
            }

        # ── Emotion analysis ──────────────────────────────────────
        if "emotion_logits" in outputs:
            emo_probs = torch.sigmoid(outputs["emotion_logits"]).detach().cpu()
            if emo_probs.ndim == 2:
                emo_probs = emo_probs[0]

            emotions = []
            for i, label in enumerate(EMOTION_LABELS):
                if i < len(emo_probs):
                    emotions.append({
                        "emotion": label,
                        "probability": float(emo_probs[i]),
                    })

            emotions.sort(key=lambda x: x["probability"], reverse=True)
            explanation["emotions"] = emotions
            explanation["dominant_emotions"] = [
                e for e in emotions if e["probability"] > 0.3
            ]

        # ── Manipulation analysis ─────────────────────────────────
        if "manipulation_logits" in outputs:
            man_probs = F.softmax(outputs["manipulation_logits"], dim=-1).detach().cpu()
            if man_probs.ndim == 2:
                man_probs = man_probs[0]

            manipulations = []
            for i, label in enumerate(MANIPULATION_LABELS):
                if i < len(man_probs):
                    manipulations.append({
                        "type": label,
                        "probability": float(man_probs[i]),
                    })

            manipulations.sort(key=lambda x: x["probability"], reverse=True)
            explanation["manipulation"] = {
                "detected": manipulations[0]["type"] if manipulations[0]["type"] != "none" else None,
                "probabilities": manipulations,
                "risk_level": self._assess_manipulation_risk(man_probs),
            }

        # ── Meta-feature insights ─────────────────────────────────
        if meta_features:
            explanation["linguistic_insights"] = self._interpret_meta_features(
                meta_features
            )

        return explanation

    def _generate_deception_reasons(
        self,
        text: str,
        deception_prob: float,
        meta_features: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Generate natural-language reasons for deception prediction."""
        reasons = []

        if meta_features:
            # High uncertainty
            if meta_features.get("uncertainty_ratio", 0) > 0.05:
                reasons.append(
                    "High use of hedging language and uncertainty markers "
                    "(e.g., 'maybe', 'I think', 'possibly')"
                )

            # Low lexical diversity (repetitive language)
            if meta_features.get("lexical_diversity", 1) < 0.4:
                reasons.append(
                    "Low lexical diversity — repetitive word usage may indicate "
                    "rehearsed or fabricated narrative"
                )

            # Excessive certainty (overcompensation)
            if meta_features.get("liwc_certainty_words", 0) > 0.03:
                reasons.append(
                    "Overuse of certainty words ('always', 'never', 'definitely') — "
                    "potential overcompensation for deception"
                )

            # Deception markers
            if meta_features.get("liwc_deception_markers", 0) > 0.02:
                reasons.append(
                    "Presence of classic deception markers ('honestly', 'trust me', "
                    "'believe me') — unprompted truthfulness claims"
                )

            # Low sensory detail
            if meta_features.get("sensory_detail_ratio", 0) < 0.005:
                reasons.append(
                    "Lack of sensory details — truthful accounts typically include "
                    "more vivid sensory information"
                )

            # Sentiment volatility
            if meta_features.get("sentiment_std", 0) > 0.3:
                reasons.append(
                    "High emotional volatility across sentences — inconsistent "
                    "sentiment may indicate fabrication"
                )

            # Distancing language
            if meta_features.get("pronoun_distancing", 0) > 0.5:
                reasons.append(
                    "Heavy use of distancing language and third-person pronouns — "
                    "cognitive distancing from deceptive content"
                )

        if not reasons:
            if deception_prob > 0.5:
                reasons.append(
                    "Pattern analysis detected subtle linguistic cues correlated "
                    "with deceptive communication"
                )
            else:
                reasons.append(
                    "Text exhibits linguistic patterns consistent with truthful "
                    "communication"
                )

        return reasons

    @staticmethod
    def _assess_manipulation_risk(probs: torch.Tensor) -> str:
        """Categorize manipulation risk level."""
        max_prob = float(probs.max())
        none_prob = float(probs[0]) if len(probs) > 0 else 1.0

        if none_prob > 0.7:
            return "low"
        elif none_prob > 0.4:
            return "moderate"
        elif none_prob > 0.2:
            return "high"
        return "critical"

    @staticmethod
    def _interpret_meta_features(features: Dict[str, float]) -> List[Dict[str, str]]:
        """Convert meta-features into human-readable insights."""
        insights = []

        if features.get("sentiment_trajectory_slope", 0) < -0.1:
            insights.append({
                "category": "Sentiment",
                "insight": "Declining emotional tone detected throughout the message",
                "severity": "medium",
            })
        elif features.get("sentiment_trajectory_slope", 0) > 0.1:
            insights.append({
                "category": "Sentiment",
                "insight": "Escalating positive sentiment throughout the message",
                "severity": "low",
            })

        if features.get("polarity_flip_count", 0) > 2:
            insights.append({
                "category": "Emotional Consistency",
                "insight": "Multiple emotional polarity shifts detected — mixed signals",
                "severity": "high",
            })

        if features.get("question_ratio", 0) > 0.5:
            insights.append({
                "category": "Communication Style",
                "insight": "High question density — possible interrogative or deflective pattern",
                "severity": "medium",
            })

        if features.get("modal_verb_density", 0) > 0.05:
            insights.append({
                "category": "Certainty",
                "insight": "Elevated use of modal verbs indicating conditional or tentative reasoning",
                "severity": "low",
            })

        if features.get("reading_ease", 100) < 30:
            insights.append({
                "category": "Complexity",
                "insight": "Very complex language structure — may indicate obfuscation",
                "severity": "medium",
            })

        return insights


# ═══════════════════════════════════════════════════════════════════
# SHAP-style Importance Estimator
# ═══════════════════════════════════════════════════════════════════

class TokenImportanceEstimator:
    """
    Estimate token-level importance using occlusion-based approach.
    Approximates SHAP values by measuring prediction change when
    individual tokens are masked.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def estimate(
        self,
        text: str,
        task: str = "deception",
        max_tokens: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Estimate importance of each token by measuring prediction delta
        when that token is replaced with [MASK].

        Args:
            text: Input text
            task: Which task head to evaluate ('deception', 'emotion', 'manipulation')
            max_tokens: Maximum tokens to evaluate
        """
        self.model.eval()

        # Baseline prediction
        encoding = self.tokenizer(
            text,
            max_length=settings.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        baseline = self.model(encoding["input_ids"], encoding["attention_mask"])

        if task == "deception":
            baseline_prob = F.softmax(baseline["deception_logits"], dim=-1)[0][1].item()
        elif task == "manipulation":
            baseline_prob = F.softmax(baseline["manipulation_logits"], dim=-1)[0].max().item()
        else:
            baseline_prob = torch.sigmoid(baseline["emotion_logits"])[0].max().item()

        # Get token list
        tokens = self.tokenizer.tokenize(text)[:max_tokens]
        importances = []

        mask_id = self.tokenizer.mask_token_id or self.tokenizer.pad_token_id

        for i, token in enumerate(tokens):
            # Replace token with mask
            masked_ids = encoding["input_ids"].clone()
            # +1 for CLS token offset
            if i + 1 < masked_ids.size(1):
                masked_ids[0, i + 1] = mask_id

            masked_output = self.model(masked_ids, encoding["attention_mask"])

            if task == "deception":
                masked_prob = F.softmax(masked_output["deception_logits"], dim=-1)[0][1].item()
            elif task == "manipulation":
                masked_prob = F.softmax(masked_output["manipulation_logits"], dim=-1)[0].max().item()
            else:
                masked_prob = torch.sigmoid(masked_output["emotion_logits"])[0].max().item()

            importance = abs(baseline_prob - masked_prob)
            importances.append({
                "token": token,
                "importance": float(importance),
                "direction": "positive" if masked_prob < baseline_prob else "negative",
                "index": i,
            })

        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances
