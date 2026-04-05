"""
NeuroLens Hybrid Multi-Task Model
Combines Transformer + BiLSTM + Dense Meta-Feature layers
with multi-task classification heads for deception, emotion, and manipulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple

from app.config import settings, EMOTION_LABELS, DECEPTION_LABELS, MANIPULATION_LABELS
from app.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════
# Attention Pooling Layer
# ═══════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    """Learnable attention pooling over sequence outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) attention mask
        Returns:
            pooled: (batch, hidden_dim)
            weights: (batch, seq_len) attention weights for explainability
        """
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled, weights


# ═══════════════════════════════════════════════════════════════════
# Multi-Task Classification Heads
# ═══════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """Shared architecture for task-specific classification heads."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════
# NeuroLens Hybrid Model
# ═══════════════════════════════════════════════════════════════════

class NeuroLensModel(nn.Module):
    """
    Production-grade hybrid multi-task model for behavioral analysis.

    Architecture:
    ┌──────────────────────────────────────────────┐
    │  Input Text                                  │
    │     ↓                                        │
    │  Transformer (DeBERTa/RoBERTa)               │
    │     ↓                                        │
    │  BiLSTM (sequential pattern capture)         │
    │     ↓                                        │
    │  Attention Pooling (explainable)              │
    │     ↓                                        │
    │  ┌──────────┐  ┌────────────┐                │
    │  │ Meta     │  │ Pooled     │                │
    │  │ Features │  │ Repr.      │                │
    │  └────┬─────┘  └─────┬──────┘                │
    │       └──────┬───────┘                       │
    │              ↓                               │
    │        Fused Representation                   │
    │       ↙      ↓        ↘                      │
    │  Deception  Emotion  Manipulation            │
    │  Head       Head     Head                    │
    └──────────────────────────────────────────────┘
    """

    def __init__(
        self,
        transformer_name: str = None,
        hidden_dim: int = None,
        lstm_layers: int = None,
        lstm_dropout: float = None,
        num_emotions: int = None,
        meta_feature_dim: int = None,
        dropout: float = None,
    ):
        super().__init__()

        transformer_name = transformer_name or settings.TRANSFORMER_MODEL
        hidden_dim = hidden_dim or settings.HIDDEN_DIM
        lstm_layers = lstm_layers or settings.LSTM_LAYERS
        lstm_dropout = lstm_dropout or settings.LSTM_DROPOUT
        num_emotions = num_emotions or settings.NUM_EMOTIONS
        meta_feature_dim = meta_feature_dim or settings.META_FEATURE_DIM
        dropout = dropout or settings.DROPOUT_RATE

        # ── Transformer Backbone ──────────────────────────────────
        self.config = AutoConfig.from_pretrained(transformer_name)
        self.transformer = AutoModel.from_pretrained(transformer_name)
        transformer_dim = self.config.hidden_size

        # Freeze lower transformer layers for efficiency
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False

        # ── BiLSTM Layer ──────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=transformer_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim * 2)

        # ── Attention Pooling ─────────────────────────────────────
        self.attention_pool = AttentionPooling(hidden_dim * 2)

        # ── Meta Feature Projection ──────────────────────────────
        self.meta_projector = nn.Sequential(
            nn.Linear(meta_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Fusion Layer ─────────────────────────────────────────
        fused_dim = hidden_dim * 2 + hidden_dim  # BiLSTM + meta
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Classification Heads ──────────────────────────────────
        head_input_dim = hidden_dim * 2
        self.deception_head = ClassificationHead(
            head_input_dim, len(DECEPTION_LABELS), dropout
        )
        self.emotion_head = ClassificationHead(
            head_input_dim, num_emotions, dropout
        )
        self.manipulation_head = ClassificationHead(
            head_input_dim, len(MANIPULATION_LABELS), dropout
        )

        # Log model architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"NeuroLensModel initialized │ "
            f"total_params={total_params:,} │ "
            f"trainable_params={trainable_params:,} │ "
            f"transformer={transformer_name}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model.

        Returns:
            Dict with keys:
                - deception_logits: (batch, 2)
                - emotion_logits: (batch, num_emotions)
                - manipulation_logits: (batch, num_manipulation)
                - attention_weights: (batch, seq_len) for explainability
        """
        # ── Transformer encoding ──────────────────────────────────
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_output.last_hidden_state  # (batch, seq, dim)

        # ── BiLSTM sequential encoding ────────────────────────────
        lstm_output, _ = self.bilstm(hidden_states)  # (batch, seq, hidden*2)
        lstm_output = self.lstm_norm(lstm_output)

        # ── Attention pooling ─────────────────────────────────────
        pooled, attention_weights = self.attention_pool(lstm_output, attention_mask)

        # ── Meta feature fusion ───────────────────────────────────
        if meta_features is not None:
            meta_proj = self.meta_projector(meta_features)
            fused = torch.cat([pooled, meta_proj], dim=-1)
        else:
            # If no meta features, pad with zeros
            batch_size = pooled.size(0)
            zero_meta = torch.zeros(
                batch_size, pooled.size(-1) // 2,
                device=pooled.device, dtype=pooled.dtype
            )
            fused = torch.cat([pooled, zero_meta], dim=-1)

        fused = self.fusion(fused)

        # ── Classification ────────────────────────────────────────
        deception_logits = self.deception_head(fused)
        emotion_logits = self.emotion_head(fused)
        manipulation_logits = self.manipulation_head(fused)

        return {
            "deception_logits": deception_logits,
            "emotion_logits": emotion_logits,
            "manipulation_logits": manipulation_logits,
            "attention_weights": attention_weights,
            "fused_representation": fused,
        }


# ═══════════════════════════════════════════════════════════════════
# Multi-Task Loss Function
# ═══════════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning with learnable task weights.
    Uses uncertainty weighting (Kendall et al., 2018).
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        deception_weight: Optional[torch.Tensor] = None,
        manipulation_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Learnable log-variance parameters for task weighting
        self.log_var_deception = nn.Parameter(torch.zeros(1))
        self.log_var_emotion = nn.Parameter(torch.zeros(1))
        self.log_var_manipulation = nn.Parameter(torch.zeros(1))

        self.deception_loss = nn.CrossEntropyLoss(
            weight=deception_weight,
            label_smoothing=label_smoothing,
        )
        self.emotion_loss = nn.BCEWithLogitsLoss()  # Multi-label
        self.manipulation_loss = nn.CrossEntropyLoss(
            weight=manipulation_weight,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        deception_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        manipulation_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with uncertainty weighting.
        Only computes loss for available labels.
        """
        total_loss = torch.tensor(0.0, device=outputs["deception_logits"].device)
        losses = {}

        if deception_labels is not None:
            dec_loss = self.deception_loss(outputs["deception_logits"], deception_labels)
            precision_d = torch.exp(-self.log_var_deception)
            total_loss += precision_d * dec_loss + self.log_var_deception
            losses["deception_loss"] = dec_loss.item()

        if emotion_labels is not None:
            emo_loss = self.emotion_loss(outputs["emotion_logits"], emotion_labels)
            precision_e = torch.exp(-self.log_var_emotion)
            total_loss += precision_e * emo_loss + self.log_var_emotion
            losses["emotion_loss"] = emo_loss.item()

        if manipulation_labels is not None:
            man_loss = self.manipulation_loss(
                outputs["manipulation_logits"], manipulation_labels
            )
            precision_m = torch.exp(-self.log_var_manipulation)
            total_loss += precision_m * man_loss + self.log_var_manipulation
            losses["manipulation_loss"] = man_loss.item()

        losses["total_loss"] = total_loss.item()
        losses["loss_tensor"] = total_loss

        return losses
