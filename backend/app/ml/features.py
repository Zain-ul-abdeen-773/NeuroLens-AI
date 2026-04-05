"""
NeuroLens Feature Engineering Module
Extracts lexical, syntactic, psychological, and sentiment features from text.
"""

import re
import math
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import LIWC_CATEGORIES
from app.utils.helpers import compute_text_statistics
from app.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════
# TF-IDF Feature Extractor
# ═══════════════════════════════════════════════════════════════════

class TFIDFExtractor:
    """TF-IDF feature extraction with configurable parameters."""

    def __init__(self, max_features: int = 5000, ngram_range=(1, 3)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
            max_df=0.95,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "TFIDFExtractor":
        self.vectorizer.fit(texts)
        self._fitted = True
        logger.info(f"TF-IDF fitted │ vocab_size={len(self.vectorizer.vocabulary_)}")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            return self.fit(texts).transform(texts)
        return self.vectorizer.transform(texts).toarray()

    def get_top_features(self, text: str, n: int = 10) -> List[Dict]:
        """Get top TF-IDF features for a single text."""
        if not self._fitted:
            return []
        vec = self.vectorizer.transform([text]).toarray()[0]
        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = np.argsort(vec)[-n:][::-1]
        return [
            {"feature": feature_names[i], "score": float(vec[i])}
            for i in top_indices if vec[i] > 0
        ]


# ═══════════════════════════════════════════════════════════════════
# LIWC-Style Psychological Feature Extractor
# ═══════════════════════════════════════════════════════════════════

class PsychologicalFeatureExtractor:
    """
    Extract LIWC-inspired psychological & linguistic cues.
    Computes category proportions and patterns indicative of
    deception, manipulation, and emotional state.
    """

    def __init__(self, categories: Dict[str, List[str]] = None):
        self.categories = categories or LIWC_CATEGORIES

    def extract(self, text: str) -> Dict[str, float]:
        """Extract all psychological features from text."""
        words = text.lower().split()
        word_count = max(len(words), 1)
        features = {}

        # ── Category proportions ──────────────────────────────────
        for category, word_list in self.categories.items():
            count = sum(1 for w in words if w in word_list)
            # Also check multi-word phrases
            text_lower = text.lower()
            for phrase in word_list:
                if " " in phrase and phrase in text_lower:
                    count += text_lower.count(phrase)
            features[f"liwc_{category}"] = count / word_count

        # ── Pronoun ratios ────────────────────────────────────────
        i_count = features.get("liwc_first_person_singular", 0)
        we_count = features.get("liwc_first_person_plural", 0)
        they_count = features.get("liwc_third_person", 0)
        features["pronoun_self_focus"] = i_count / max(i_count + we_count + they_count, 0.001)
        features["pronoun_distancing"] = they_count / max(i_count + we_count + they_count, 0.001)

        # ── Complexity metrics ────────────────────────────────────
        features["lexical_diversity"] = len(set(words)) / word_count
        features["avg_word_length"] = sum(len(w) for w in words) / word_count

        # Flesch-Kincaid approximation
        syllable_count = sum(self._count_syllables(w) for w in words)
        sentences = re.split(r"[.!?]+", text)
        sentence_count = max(len([s for s in sentences if s.strip()]), 1)
        features["reading_ease"] = (
            206.835
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (syllable_count / word_count)
        )

        # ── Deception-specific patterns ───────────────────────────
        features["negation_density"] = sum(
            1 for w in words if w in {"not", "no", "never", "neither", "nobody", "none", "nor", "nothing", "nowhere"}
        ) / word_count

        features["modal_verb_density"] = sum(
            1 for w in words if w in {"would", "could", "should", "might", "may", "can"}
        ) / word_count

        features["exclusive_word_density"] = sum(
            1 for w in words if w in {"but", "except", "however", "although", "unless", "without"}
        ) / word_count

        # Sensory detail ratio (truthful speakers tend to include more)
        sensory_words = {"saw", "heard", "felt", "smelled", "tasted", "touched", "looked", "sounded", "seemed"}
        features["sensory_detail_ratio"] = sum(1 for w in words if w in sensory_words) / word_count

        # Temporal reference density
        temporal_words = {"then", "after", "before", "during", "while", "when", "next", "later", "earlier", "ago"}
        features["temporal_reference_density"] = sum(1 for w in words if w in temporal_words) / word_count

        return features

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower().rstrip("e")
        count = len(re.findall(r"[aeiouy]+", word))
        return max(count, 1)


# ═══════════════════════════════════════════════════════════════════
# Sentiment Trajectory Analyzer
# ═══════════════════════════════════════════════════════════════════

class SentimentTrajectoryAnalyzer:
    """
    Analyzes how sentiment shifts across sentences within a text.
    Detects sudden polarity changes that may indicate deception or manipulation.
    """

    # Simple lexicon-based sentiment (avoids heavy dependency)
    POSITIVE = {
        "good", "great", "love", "happy", "wonderful", "amazing",
        "excellent", "fantastic", "beautiful", "perfect", "best", "enjoy",
        "glad", "pleased", "delighted", "grateful", "thankful", "awesome",
        "incredible", "brilliant", "superb", "outstanding", "magnificent",
    }
    NEGATIVE = {
        "bad", "terrible", "hate", "sad", "awful", "horrible",
        "worst", "ugly", "disgusting", "miserable", "angry", "furious",
        "annoyed", "frustrated", "disappointed", "depressed", "anxious",
        "scared", "worried", "upset", "painful", "dreadful", "nasty",
    }

    def analyze(self, text: str) -> Dict[str, float]:
        """Compute sentiment trajectory features."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return {
                "sentiment_mean": 0.0,
                "sentiment_std": 0.0,
                "sentiment_range": 0.0,
                "sentiment_shifts": 0,
                "max_sentiment_jump": 0.0,
                "polarity_flip_count": 0,
                "sentiment_trajectory_slope": 0.0,
            }

        scores = [self._sentence_sentiment(s) for s in sentences]

        # Compute trajectory features
        shifts = [abs(scores[i] - scores[i - 1]) for i in range(1, len(scores))]
        polarity_flips = sum(
            1 for i in range(1, len(scores))
            if (scores[i] > 0 and scores[i - 1] < 0) or (scores[i] < 0 and scores[i - 1] > 0)
        )

        # Linear regression slope for sentiment trajectory
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / max(denominator, 1e-8)

        return {
            "sentiment_mean": float(np.mean(scores)),
            "sentiment_std": float(np.std(scores)),
            "sentiment_range": float(max(scores) - min(scores)),
            "sentiment_shifts": len([s for s in shifts if s > 0.3]),
            "max_sentiment_jump": float(max(shifts)) if shifts else 0.0,
            "polarity_flip_count": polarity_flips,
            "sentiment_trajectory_slope": float(slope),
        }

    def _sentence_sentiment(self, sentence: str) -> float:
        """Compute basic sentiment score for a sentence."""
        words = sentence.lower().split()
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in self.POSITIVE)
        neg = sum(1 for w in words if w in self.NEGATIVE)
        return (pos - neg) / len(words)


# ═══════════════════════════════════════════════════════════════════
# Uncertainty Marker Detector
# ═══════════════════════════════════════════════════════════════════

class UncertaintyDetector:
    """Detects linguistic markers of uncertainty and hesitation."""

    UNCERTAINTY_PATTERNS = [
        r"\bi think\b", r"\bi guess\b", r"\bi suppose\b",
        r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b",
        r"\bmight\b", r"\bcould be\b", r"\bprobably\b",
        r"\bnot sure\b", r"\bnot certain\b", r"\bi believe\b",
        r"\bit seems\b", r"\bapparently\b", r"\bpresumably\b",
        r"\bsort of\b", r"\bkind of\b", r"\bmore or less\b",
        r"\bto some extent\b", r"\bin a way\b",
        r"\b(um|uh|hmm|er|ah)\b",
    ]

    CONFIDENCE_PATTERNS = [
        r"\bi know\b", r"\bi('m| am) sure\b", r"\bdefinitely\b",
        r"\bcertainly\b", r"\babsolutely\b", r"\bwithout (a )?doubt\b",
        r"\bclearly\b", r"\bobviously\b", r"\bundoubtedly\b",
        r"\bno question\b", r"\bexactly\b", r"\bprecisely\b",
    ]

    def detect(self, text: str) -> Dict[str, float]:
        """Detect uncertainty and confidence markers."""
        text_lower = text.lower()
        word_count = max(len(text_lower.split()), 1)

        uncertainty_count = sum(
            len(re.findall(p, text_lower)) for p in self.UNCERTAINTY_PATTERNS
        )
        confidence_count = sum(
            len(re.findall(p, text_lower)) for p in self.CONFIDENCE_PATTERNS
        )

        total = uncertainty_count + confidence_count
        return {
            "uncertainty_ratio": uncertainty_count / word_count,
            "confidence_ratio": confidence_count / word_count,
            "certainty_balance": (
                (confidence_count - uncertainty_count) / max(total, 1)
            ),
            "hedging_density": uncertainty_count / word_count,
        }


# ═══════════════════════════════════════════════════════════════════
# Syntactic Pattern Extractor
# ═══════════════════════════════════════════════════════════════════

class SyntacticPatternExtractor:
    """Extract syntactic patterns indicative of writing style and intent."""

    def extract(self, text: str) -> Dict[str, float]:
        """Extract syntactic features from text."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        word_count = max(len(words), 1)

        # Sentence-level features
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
        sent_length_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0

        # Punctuation patterns
        features = {
            "avg_sentence_length": float(avg_sent_length),
            "sentence_length_variance": float(sent_length_var),
            "question_ratio": text.count("?") / max(len(sentences), 1),
            "exclamation_ratio": text.count("!") / max(len(sentences), 1),
            "ellipsis_count": text.count("...") / word_count,
            "quoted_speech_ratio": len(re.findall(r'"[^"]*"', text)) / max(len(sentences), 1),
            "parenthetical_ratio": len(re.findall(r"\([^)]*\)", text)) / max(len(sentences), 1),
            "comma_density": text.count(",") / word_count,
            "capitalization_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "repetition_ratio": 1 - len(set(w.lower() for w in words)) / word_count,
        }

        return features


# ═══════════════════════════════════════════════════════════════════
# Master Feature Extractor
# ═══════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Orchestrates all feature extraction pipelines.
    Produces a unified feature vector for each text sample.
    """

    def __init__(self):
        self.tfidf = TFIDFExtractor()
        self.psych = PsychologicalFeatureExtractor()
        self.sentiment = SentimentTrajectoryAnalyzer()
        self.uncertainty = UncertaintyDetector()
        self.syntactic = SyntacticPatternExtractor()
        logger.info("FeatureExtractor initialized with all sub-extractors")

    def extract_meta_features(self, text: str) -> Dict[str, float]:
        """Extract all meta-features for a single text."""
        features = {}
        features.update(compute_text_statistics(text))
        features.update(self.psych.extract(text))
        features.update(self.sentiment.analyze(text))
        features.update(self.uncertainty.detect(text))
        features.update(self.syntactic.extract(text))
        return features

    def extract_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Extract meta-features for a batch of texts."""
        return [self.extract_meta_features(t) for t in texts]

    def get_feature_names(self) -> List[str]:
        """Return ordered list of all meta-feature names."""
        dummy = self.extract_meta_features("This is a sample text for feature inspection.")
        return list(dummy.keys())

    def get_feature_dim(self) -> int:
        """Return the total dimensionality of meta-features."""
        return len(self.get_feature_names())
