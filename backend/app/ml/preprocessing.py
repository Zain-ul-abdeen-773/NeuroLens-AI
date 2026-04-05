"""
NeuroLens Text Preprocessing Pipeline
Handles text cleaning, tokenization, normalization, and dataset loading.
"""

import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from app.config import settings, EMOTION_LABELS, DECEPTION_LABELS, MANIPULATION_LABELS
from app.utils.helpers import clean_text
from app.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProcessedSample:
    """A single preprocessed training/inference sample."""
    text: str
    clean_text: str
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    deception_label: Optional[int] = None
    emotion_labels: Optional[List[int]] = None
    manipulation_label: Optional[int] = None
    meta_features: Optional[Dict[str, float]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Text Preprocessing Functions
# ═══════════════════════════════════════════════════════════════════

def normalize_tokens(text: str) -> str:
    """Normalize contractions, slang, and informal language."""
    replacements = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am",
        "gonna": "going to", "wanna": "want to", "gotta": "got to",
        "kinda": "kind of", "sorta": "sort of", "lotta": "lot of",
        "gimme": "give me", "lemme": "let me", "ya": "you",
        "u": "you", "r": "are", "ur": "your",
        "bc": "because", "tbh": "to be honest", "imo": "in my opinion",
        "btw": "by the way", "idk": "I don't know", "smh": "shaking my head",
    }
    lower = text.lower()
    for old, new in replacements.items():
        lower = lower.replace(old, new)
    return lower


def segment_sentences(text: str) -> List[str]:
    """Split text into sentences using multiple delimiters."""
    # Handle common abbreviations to avoid false splits
    text = re.sub(r"(Mr|Mrs|Dr|Prof|Sr|Jr)\.", r"\1<DOT>", text)
    text = re.sub(r"(\d)\.", r"\1<DOT>", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]
    return sentences


def augment_synonym_replacement(text: str, n: int = 2) -> str:
    """
    Simple synonym replacement augmentation.
    Replaces random adjectives/adverbs with synonyms for training diversity.
    """
    synonym_map = {
        "good": ["great", "excellent", "fine", "decent", "solid"],
        "bad": ["terrible", "awful", "poor", "dreadful", "lousy"],
        "happy": ["joyful", "glad", "pleased", "delighted", "cheerful"],
        "sad": ["unhappy", "sorrowful", "gloomy", "melancholy", "downcast"],
        "big": ["large", "huge", "enormous", "massive", "vast"],
        "small": ["tiny", "little", "miniature", "compact", "slight"],
        "fast": ["quick", "rapid", "swift", "speedy", "brisk"],
        "slow": ["sluggish", "gradual", "leisurely", "unhurried", "plodding"],
        "important": ["crucial", "vital", "essential", "significant", "critical"],
        "difficult": ["hard", "challenging", "tough", "complex", "demanding"],
        "angry": ["furious", "irate", "enraged", "livid", "incensed"],
        "scared": ["afraid", "frightened", "terrified", "fearful", "alarmed"],
    }
    words = text.split()
    indices = [i for i, w in enumerate(words) if w.lower() in synonym_map]
    if not indices:
        return text
    chosen = random.sample(indices, min(n, len(indices)))
    for idx in chosen:
        key = words[idx].lower()
        words[idx] = random.choice(synonym_map[key])
    return " ".join(words)


def augment_random_swap(text: str, n: int = 2) -> str:
    """Randomly swap adjacent words for augmentation."""
    words = text.split()
    if len(words) < 4:
        return text
    for _ in range(n):
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)


# ═══════════════════════════════════════════════════════════════════
# Dataset Classes
# ═══════════════════════════════════════════════════════════════════

class NeuroLensDataset(Dataset):
    """
    Multi-task dataset for NeuroLens model training.
    Handles tokenization and label encoding for all three tasks.
    """

    def __init__(
        self,
        texts: List[str],
        deception_labels: Optional[List[int]] = None,
        emotion_labels: Optional[List[List[int]]] = None,
        manipulation_labels: Optional[List[int]] = None,
        meta_features: Optional[List[Dict[str, float]]] = None,
        tokenizer_name: str = None,
        max_length: int = None,
        augment: bool = False,
    ):
        self.texts = texts
        self.deception_labels = deception_labels
        self.emotion_labels = emotion_labels
        self.manipulation_labels = manipulation_labels
        self.meta_features = meta_features
        self.augment = augment
        self.max_length = max_length or settings.MAX_SEQ_LENGTH

        tokenizer_name = tokenizer_name or settings.TRANSFORMER_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(
            f"NeuroLens Dataset initialized │ samples={len(texts)} │ "
            f"augment={augment} │ max_length={self.max_length}"
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = clean_text(self.texts[idx])
        text = normalize_tokens(text)

        # Apply augmentations during training
        if self.augment and random.random() > 0.5:
            if random.random() > 0.5:
                text = augment_synonym_replacement(text)
            else:
                text = augment_random_swap(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        # Add labels if available
        if self.deception_labels is not None:
            item["deception_label"] = torch.tensor(
                self.deception_labels[idx], dtype=torch.long
            )

        if self.emotion_labels is not None:
            item["emotion_labels"] = torch.tensor(
                self.emotion_labels[idx], dtype=torch.float
            )

        if self.manipulation_labels is not None:
            item["manipulation_label"] = torch.tensor(
                self.manipulation_labels[idx], dtype=torch.long
            )

        # Add meta features
        if self.meta_features is not None:
            feat_values = list(self.meta_features[idx].values())
            item["meta_features"] = torch.tensor(feat_values, dtype=torch.float)

        return item


def create_dataloader(
    dataset: NeuroLensDataset,
    batch_size: int = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with production-grade settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size or settings.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ═══════════════════════════════════════════════════════════════════
# Dataset Loaders (LIAR, GoEmotions, Custom)
# ═══════════════════════════════════════════════════════════════════

def load_liar_dataset(filepath: str) -> Tuple[List[str], List[int]]:
    """
    Load LIAR dataset for deception detection.
    Labels: pants-fire, false, barely-true, half-true, mostly-true, true
    Mapped to binary: deceptive (0-2) vs truthful (3-5)
    """
    texts, labels = [], []
    label_map = {
        "pants-fire": 1, "false": 1, "barely-true": 1,
        "half-true": 0, "mostly-true": 0, "true": 0
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    label_str = parts[1].lower()
                    if label_str in label_map:
                        texts.append(parts[2])
                        labels.append(label_map[label_str])
        logger.info(f"Loaded LIAR dataset │ samples={len(texts)}")
    except FileNotFoundError:
        logger.warning(f"LIAR dataset not found at {filepath}")

    return texts, labels


def load_goemotions_dataset(filepath: str) -> Tuple[List[str], List[List[int]]]:
    """
    Load GoEmotions dataset for multi-label emotion classification.
    """
    import csv

    texts, labels = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["text"])
                label_indices = [int(i) for i in row["labels"].split(",")]
                multi_hot = [0] * len(EMOTION_LABELS)
                for idx in label_indices:
                    if idx < len(multi_hot):
                        multi_hot[idx] = 1
                labels.append(multi_hot)
        logger.info(f"Loaded GoEmotions dataset │ samples={len(texts)}")
    except FileNotFoundError:
        logger.warning(f"GoEmotions dataset not found at {filepath}")

    return texts, labels


def load_custom_dataset(filepath: str) -> Dict[str, Any]:
    """
    Load custom JSON dataset with flexible schema.
    Expected format:
    [{"text": "...", "deception": 0/1, "emotions": [...], "manipulation": 0-6}, ...]
    """
    import json

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [d["text"] for d in data]
        result = {"texts": texts}

        if "deception" in data[0]:
            result["deception_labels"] = [d["deception"] for d in data]
        if "emotions" in data[0]:
            result["emotion_labels"] = [d["emotions"] for d in data]
        if "manipulation" in data[0]:
            result["manipulation_labels"] = [d["manipulation"] for d in data]

        logger.info(f"Loaded custom dataset │ samples={len(texts)}")
        return result
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load custom dataset: {e}")
        return {"texts": []}
