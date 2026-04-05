"""
NeuroLens Advanced Training Pipeline
Production-grade training with mixed precision, gradient accumulation,
cosine scheduling, early stopping, k-fold CV, and Optuna HPO.
"""

import os
import time
import copy
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from app.config import settings
from app.ml.model import NeuroLensModel, MultiTaskLoss
from app.ml.preprocessing import NeuroLensDataset, create_dataloader
from app.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════
# Learning Rate Scheduler with Warmup
# ═══════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.01,
):
    """Cosine decay schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * num_cycles * 2.0 * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════
# Early Stopping
# ═══════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Early stopping handler with model checkpointing."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered │ patience={self.patience} │ "
                    f"best={self.best_score:.4f}"
                )

        return self.should_stop


# ═══════════════════════════════════════════════════════════════════
# Gradient Checking Utility
# ═══════════════════════════════════════════════════════════════════

def gradient_check(
    model: nn.Module,
    loss_fn,
    sample_batch: Dict[str, torch.Tensor],
    eps: float = 1e-5,
    tolerance: float = 1e-3,
) -> Dict[str, float]:
    """
    Numerical gradient checking to verify backpropagation correctness.
    Compares analytical gradients with numerical finite-difference gradients.
    """
    model.eval()
    results = {}

    # Select a small subset of parameters to check
    params_to_check = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.numel() < 100:
            params_to_check.append((name, param))
            if len(params_to_check) >= 3:
                break

    for name, param in params_to_check:
        # Analytical gradient
        model.zero_grad()
        outputs = model(
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch.get("meta_features"),
        )
        losses = loss_fn(
            outputs,
            sample_batch.get("deception_label"),
            sample_batch.get("emotion_labels"),
            sample_batch.get("manipulation_label"),
        )
        losses["loss_tensor"].backward()
        analytical_grad = param.grad.clone()

        # Numerical gradient (for first element)
        idx = 0
        original = param.data.view(-1)[idx].item()

        param.data.view(-1)[idx] = original + eps
        outputs_plus = model(
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch.get("meta_features"),
        )
        loss_plus = loss_fn(
            outputs_plus,
            sample_batch.get("deception_label"),
            sample_batch.get("emotion_labels"),
            sample_batch.get("manipulation_label"),
        )["loss_tensor"].item()

        param.data.view(-1)[idx] = original - eps
        outputs_minus = model(
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch.get("meta_features"),
        )
        loss_minus = loss_fn(
            outputs_minus,
            sample_batch.get("deception_label"),
            sample_batch.get("emotion_labels"),
            sample_batch.get("manipulation_label"),
        )["loss_tensor"].item()

        param.data.view(-1)[idx] = original
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical_val = analytical_grad.view(-1)[idx].item()

        diff = abs(numerical_grad - analytical_val) / (
            abs(numerical_grad) + abs(analytical_val) + 1e-8
        )
        results[name] = diff
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        logger.info(
            f"Gradient check │ {name} │ diff={diff:.6f} │ {status}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════
# Main Trainer Class
# ═══════════════════════════════════════════════════════════════════

class NeuroLensTrainer:
    """
    Production training pipeline with all advanced features.
    """

    def __init__(
        self,
        model: NeuroLensModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
    ):
        self.config = config or {}
        self.device = settings.device
        self.model = model.to(self.device)

        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

        # ── Training hyperparameters ──────────────────────────────
        self.epochs = self.config.get("epochs", settings.EPOCHS)
        self.lr = self.config.get("learning_rate", settings.LEARNING_RATE)
        self.accumulation_steps = self.config.get(
            "gradient_accumulation_steps", settings.GRADIENT_ACCUMULATION_STEPS
        )
        self.max_grad_norm = self.config.get("max_grad_norm", settings.MAX_GRAD_NORM)
        self.use_amp = self.config.get("use_mixed_precision", settings.USE_MIXED_PRECISION)

        # ── Optimizer ─────────────────────────────────────────────
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_params = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": settings.WEIGHT_DECAY,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_params, lr=self.lr, eps=1e-8)

        # ── Scheduler ─────────────────────────────────────────────
        total_steps = len(self.train_dl) * self.epochs // self.accumulation_steps
        warmup_steps = int(total_steps * settings.WARMUP_RATIO)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # ── Loss ──────────────────────────────────────────────────
        self.loss_fn = MultiTaskLoss(
            label_smoothing=settings.LABEL_SMOOTHING
        ).to(self.device)

        # ── Mixed Precision ───────────────────────────────────────
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── Early Stopping ────────────────────────────────────────
        self.early_stopping = EarlyStopping(
            patience=settings.EARLY_STOPPING_PATIENCE, mode="min"
        )

        # ── TensorBoard ──────────────────────────────────────────
        log_dir = Path(settings.TENSORBOARD_DIR) / f"run_{int(time.time())}"
        self.writer = SummaryWriter(str(log_dir))
        logger.info(f"TensorBoard logs → {log_dir}")

        # ── Training state ────────────────────────────────────────
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history = []

    def train(self) -> Dict[str, Any]:
        """Execute the full training loop."""
        logger.info(
            f"Starting training │ epochs={self.epochs} │ "
            f"device={self.device} │ amp={self.use_amp} │ "
            f"accumulation_steps={self.accumulation_steps}"
        )

        for epoch in range(self.epochs):
            # ── Training Phase ────────────────────────────────────
            train_metrics = self._train_epoch(epoch)

            # ── Validation Phase ──────────────────────────────────
            val_metrics = {}
            if self.val_dl:
                val_metrics = self._validate(epoch)

            # ── Logging ───────────────────────────────────────────
            epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.training_history.append(epoch_metrics)

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} │ "
                f"train_loss={train_metrics['train_loss']:.4f} │ "
                + (f"val_loss={val_metrics.get('val_loss', 'N/A')}" if val_metrics else "")
            )

            # ── Early Stopping Check ──────────────────────────────
            if self.val_dl:
                if self.early_stopping(val_metrics["val_loss"], self.model):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(self.early_stopping.best_model_state)
                    break

        self.writer.close()
        return {
            "history": self.training_history,
            "best_val_loss": self.early_stopping.best_score,
            "epochs_trained": len(self.training_history),
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0
        total_steps = 0
        grad_norms = []

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_dl):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast(device_type=str(self.device).split(":")[0], enabled=self.use_amp):
                outputs = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("meta_features"),
                )
                losses = self.loss_fn(
                    outputs,
                    batch.get("deception_label"),
                    batch.get("emotion_labels"),
                    batch.get("manipulation_label"),
                )
                loss = losses["loss_tensor"] / self.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                grad_norms.append(grad_norm.item())

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # TensorBoard logging
                self.writer.add_scalar("train/loss", losses["total_loss"], self.global_step)
                self.writer.add_scalar("train/learning_rate",
                                       self.scheduler.get_last_lr()[0], self.global_step)
                self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)

                for loss_name, loss_val in losses.items():
                    if loss_name != "loss_tensor" and loss_name != "total_loss":
                        self.writer.add_scalar(
                            f"train/{loss_name}", loss_val, self.global_step
                        )

            total_loss += losses["total_loss"]
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0

        self.writer.add_scalar("epoch/train_loss", avg_loss, epoch)
        self.writer.add_scalar("epoch/avg_grad_norm", avg_grad_norm, epoch)

        return {
            "train_loss": avg_loss,
            "avg_grad_norm": avg_grad_norm,
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_steps = 0

        for batch in self.val_dl:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast(device_type=str(self.device).split(":")[0], enabled=self.use_amp):
                outputs = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("meta_features"),
                )
                losses = self.loss_fn(
                    outputs,
                    batch.get("deception_label"),
                    batch.get("emotion_labels"),
                    batch.get("manipulation_label"),
                )

            total_loss += losses["total_loss"]
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        self.writer.add_scalar("epoch/val_loss", avg_loss, epoch)

        return {"val_loss": avg_loss}

    def save_model(self, path: str = None) -> str:
        """Save model checkpoint."""
        save_dir = Path(path or settings.MODEL_SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "neurolens_model.pt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_history": self.training_history,
            "config": self.config,
            "global_step": self.global_step,
        }, save_path)

        logger.info(f"Model saved → {save_path}")
        return str(save_path)


# ═══════════════════════════════════════════════════════════════════
# K-Fold Cross Validation
# ═══════════════════════════════════════════════════════════════════

def run_kfold_cv(
    dataset: NeuroLensDataset,
    model_factory,
    k: int = None,
    config: Dict = None,
) -> Dict[str, Any]:
    """
    Run stratified K-Fold cross validation.

    Args:
        dataset: The full dataset
        model_factory: Callable that returns a fresh model instance
        k: Number of folds
        config: Training config overrides
    """
    k = k or settings.K_FOLDS
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Use deception labels for stratification if available
    labels = dataset.deception_labels or [0] * len(dataset)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        logger.info(f"═══ Fold {fold + 1}/{k} ═══")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_dl = DataLoader(train_subset, batch_size=settings.BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_subset, batch_size=settings.BATCH_SIZE, shuffle=False)

        model = model_factory()
        trainer = NeuroLensTrainer(model, train_dl, val_dl, config)
        result = trainer.train()
        fold_results.append(result)

    # Aggregate results
    avg_val_loss = np.mean([r["best_val_loss"] for r in fold_results if r["best_val_loss"]])
    logger.info(f"K-Fold CV Complete │ avg_val_loss={avg_val_loss:.4f}")

    return {
        "fold_results": fold_results,
        "avg_val_loss": float(avg_val_loss),
        "k": k,
    }


# ═══════════════════════════════════════════════════════════════════
# Optuna Hyperparameter Optimization
# ═══════════════════════════════════════════════════════════════════

def run_hyperparameter_search(
    dataset: NeuroLensDataset,
    n_trials: int = None,
    timeout: int = None,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search.
    """
    import optuna

    n_trials = n_trials or settings.OPTUNA_N_TRIALS
    timeout = timeout or settings.OPTUNA_TIMEOUT

    def objective(trial: optuna.Trial) -> float:
        config = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
            "epochs": trial.suggest_int("epochs", 3, 15),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [2, 4, 8]
            ),
        }

        # Simple train/val split for HPO
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_dl = DataLoader(train_subset, batch_size=settings.BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_subset, batch_size=settings.BATCH_SIZE, shuffle=False)

        model = NeuroLensModel()
        trainer = NeuroLensTrainer(model, train_dl, val_dl, config)
        result = trainer.train()

        return result["best_val_loss"] or float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"HPO Complete │ best_val_loss={study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
