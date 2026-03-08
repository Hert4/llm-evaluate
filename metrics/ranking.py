"""
Ranking Metrics: NDCG@K, Recall@K, Precision@K, MRR
"""
from typing import List, Dict, Any, Optional
import math
import numpy as np

from .base import BaseMetric, MetricResult
from ..data_parsers.base import EvalSample


class NDCGMetric(BaseMetric):
    """
    NDCG (Normalized Discounted Cumulative Gain) @ K.
    Measures ranking quality considering position and relevance scores.
    """

    name = "ndcg"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k_values = self.config.get("k_values", [1, 3, 5, 10])

    def _dcg(self, relevances: List[float], k: int) -> float:
        """Compute Discounted Cumulative Gain"""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg

    def _ndcg_at_k(self, relevances: List[float], k: int) -> float:
        """Compute NDCG@K"""
        dcg = self._dcg(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self._dcg(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0.0

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute NDCG scores for all samples"""
        # Filter samples with relevance scores
        valid_samples = [s for s in samples if s.relevance_scores]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with relevance_scores"},
            )

        ndcg_scores = {k: [] for k in self.k_values}

        for sample in valid_samples:
            relevances = sample.relevance_scores
            for k in self.k_values:
                score = self._ndcg_at_k(relevances, k)
                ndcg_scores[k].append(score)

        # Aggregate
        avg_ndcg = {}
        for k in self.k_values:
            if ndcg_scores[k]:
                avg_ndcg[f"ndcg@{k}"] = sum(ndcg_scores[k]) / len(ndcg_scores[k])

        main_k = self.k_values[-1] if self.k_values else 10
        main_score = avg_ndcg.get(f"ndcg@{main_k}", 0.0)

        return MetricResult(
            name=self.name,
            score=main_score,
            details={
                "ndcg_at_k": avg_ndcg,
                "k_values": self.k_values,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=ndcg_scores.get(main_k, []),
        )


class RecallAtKMetric(BaseMetric):
    """
    Recall@K.
    Measures the proportion of relevant items found in top-K results.
    """

    name = "recall_at_k"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k_values = self.config.get("k_values", [1, 3, 5, 10])
        self.relevance_threshold = self.config.get("relevance_threshold", 0.5)

    def _recall_at_k(
        self,
        relevances: List[float],
        k: int,
        threshold: float
    ) -> float:
        """Compute Recall@K"""
        total_relevant = sum(1 for r in relevances if r >= threshold)
        if total_relevant == 0:
            return 0.0

        relevant_at_k = sum(1 for r in relevances[:k] if r >= threshold)
        return relevant_at_k / total_relevant

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Recall@K for all samples"""
        valid_samples = [s for s in samples if s.relevance_scores]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with relevance_scores"},
            )

        recall_scores = {k: [] for k in self.k_values}

        for sample in valid_samples:
            relevances = sample.relevance_scores
            for k in self.k_values:
                score = self._recall_at_k(relevances, k, self.relevance_threshold)
                recall_scores[k].append(score)

        avg_recall = {}
        for k in self.k_values:
            if recall_scores[k]:
                avg_recall[f"recall@{k}"] = sum(recall_scores[k]) / len(recall_scores[k])

        main_k = self.k_values[-1] if self.k_values else 10
        main_score = avg_recall.get(f"recall@{main_k}", 0.0)

        return MetricResult(
            name=self.name,
            score=main_score,
            details={
                "recall_at_k": avg_recall,
                "k_values": self.k_values,
                "relevance_threshold": self.relevance_threshold,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=recall_scores.get(main_k, []),
        )


class PrecisionAtKMetric(BaseMetric):
    """
    Precision@K.
    Measures the proportion of relevant items in top-K results.
    """

    name = "precision_at_k"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k_values = self.config.get("k_values", [1, 3, 5, 10])
        self.relevance_threshold = self.config.get("relevance_threshold", 0.5)

    def _precision_at_k(
        self,
        relevances: List[float],
        k: int,
        threshold: float
    ) -> float:
        """Compute Precision@K"""
        if k == 0:
            return 0.0

        top_k = relevances[:k]
        if not top_k:
            return 0.0

        relevant_at_k = sum(1 for r in top_k if r >= threshold)
        return relevant_at_k / len(top_k)

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Precision@K for all samples"""
        valid_samples = [s for s in samples if s.relevance_scores]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with relevance_scores"},
            )

        precision_scores = {k: [] for k in self.k_values}

        for sample in valid_samples:
            relevances = sample.relevance_scores
            for k in self.k_values:
                score = self._precision_at_k(relevances, k, self.relevance_threshold)
                precision_scores[k].append(score)

        avg_precision = {}
        for k in self.k_values:
            if precision_scores[k]:
                avg_precision[f"precision@{k}"] = sum(precision_scores[k]) / len(precision_scores[k])

        main_k = self.k_values[-1] if self.k_values else 10
        main_score = avg_precision.get(f"precision@{main_k}", 0.0)

        return MetricResult(
            name=self.name,
            score=main_score,
            details={
                "precision_at_k": avg_precision,
                "k_values": self.k_values,
                "relevance_threshold": self.relevance_threshold,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=precision_scores.get(main_k, []),
        )


class MRRMetric(BaseMetric):
    """
    MRR (Mean Reciprocal Rank).
    Average of reciprocal ranks of first relevant result.
    """

    name = "mrr"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.5)

    def _reciprocal_rank(self, relevances: List[float], threshold: float) -> float:
        """Compute reciprocal rank"""
        for i, rel in enumerate(relevances):
            if rel >= threshold:
                return 1.0 / (i + 1)
        return 0.0

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute MRR for all samples"""
        valid_samples = [s for s in samples if s.relevance_scores]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with relevance_scores"},
            )

        rr_scores = []
        for sample in valid_samples:
            rr = self._reciprocal_rank(sample.relevance_scores, self.relevance_threshold)
            rr_scores.append(rr)

        mrr = sum(rr_scores) / len(rr_scores)

        return MetricResult(
            name=self.name,
            score=mrr,
            details={
                "relevance_threshold": self.relevance_threshold,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=rr_scores,
        )
