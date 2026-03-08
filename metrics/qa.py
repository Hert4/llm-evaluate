"""
QA Metrics: Exact Match, Token F1
"""
from typing import List, Dict, Any, Optional
from collections import Counter
import re
import string

from .base import BaseMetric, MetricResult
from ..data_parsers.base import EvalSample


class ExactMatchMetric(BaseMetric):
    """
    Exact Match (EM) Score.
    Returns 1 if prediction exactly matches reference, 0 otherwise.
    """

    name = "exact_match"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ignore_case = self.config.get("ignore_case", True)
        self.ignore_punctuation = self.config.get("ignore_punctuation", True)
        self.ignore_articles = self.config.get("ignore_articles", True)

    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison"""
        if not text:
            return ""

        # Lowercase
        if self.ignore_case:
            text = text.lower()

        # Remove punctuation
        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove articles
        if self.ignore_articles:
            articles = {"a", "an", "the"}
            tokens = text.split()
            tokens = [t for t in tokens if t not in articles]
            text = " ".join(tokens)

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    def _compute_single(self, prediction: str, reference: str) -> float:
        """Compute exact match for single pair"""
        pred_norm = self._normalize_answer(prediction)
        ref_norm = self._normalize_answer(reference)
        return 1.0 if pred_norm == ref_norm else 0.0

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Exact Match score for all samples"""
        valid_samples = self.validate_samples(samples)

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No valid samples with references"},
            )

        scores = []
        for sample in valid_samples:
            score = self._compute_single(sample.output, sample.reference)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={
                "num_correct": int(sum(scores)),
                "num_samples": len(valid_samples),
                "accuracy_pct": avg_score * 100,
            },
            per_sample_scores=scores,
        )


class TokenF1Metric(BaseMetric):
    """
    Token F1 Score.
    Measures token-level precision and recall between prediction and reference.
    """

    name = "token_f1"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ignore_case = self.config.get("ignore_case", True)
        self.ignore_punctuation = self.config.get("ignore_punctuation", True)

    def _get_tokens(self, text: str) -> List[str]:
        """Tokenize text"""
        if not text:
            return []

        if self.ignore_case:
            text = text.lower()

        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        return text.split()

    def _compute_single(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute token F1 for single pair"""
        pred_tokens = self._get_tokens(prediction)
        ref_tokens = self._get_tokens(reference)

        if not pred_tokens and not ref_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if not pred_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Count common tokens
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        common = sum((pred_counter & ref_counter).values())

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Token F1 score for all samples"""
        valid_samples = self.validate_samples(samples)

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No valid samples with references"},
            )

        f1_scores = []
        precision_scores = []
        recall_scores = []

        for sample in valid_samples:
            scores = self._compute_single(sample.output, sample.reference)
            f1_scores.append(scores["f1"])
            precision_scores.append(scores["precision"])
            recall_scores.append(scores["recall"])

        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)

        return MetricResult(
            name=self.name,
            score=avg_f1,
            details={
                "f1": avg_f1,
                "precision": avg_precision,
                "recall": avg_recall,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=f1_scores,
        )
