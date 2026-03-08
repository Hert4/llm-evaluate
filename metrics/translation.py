"""
Translation Metrics: BLEU, COMET
"""
from typing import List, Dict, Any, Optional
import re
from collections import Counter
import math

from .base import BaseMetric, MetricResult
from ..data_parsers.base import EvalSample


class BLEUMetric(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) Score.
    Measures n-gram overlap between candidate and reference translations.
    """

    name = "bleu"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_ngram = self.config.get("max_ngram", 4)
        self.smoothing = self.config.get("smoothing", True)

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts from token list"""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def _compute_modified_precision(
        self,
        candidate_tokens: List[str],
        reference_tokens: List[str],
        n: int
    ) -> tuple[int, int]:
        """Compute modified precision for n-grams"""
        candidate_ngrams = self._get_ngrams(candidate_tokens, n)
        reference_ngrams = self._get_ngrams(reference_tokens, n)

        clipped_count = 0
        total_count = 0

        for ngram, count in candidate_ngrams.items():
            clipped_count += min(count, reference_ngrams.get(ngram, 0))
            total_count += count

        return clipped_count, total_count

    def _compute_bleu_single(
        self,
        candidate: str,
        reference: str
    ) -> float:
        """Compute BLEU score for single candidate-reference pair"""
        # Tokenize
        candidate_tokens = self.tokenize(self.normalize_text(candidate))
        reference_tokens = self.tokenize(self.normalize_text(reference))

        if not candidate_tokens:
            return 0.0

        # Compute modified precision for each n-gram
        precisions = []
        for n in range(1, self.max_ngram + 1):
            clipped, total = self._compute_modified_precision(
                candidate_tokens, reference_tokens, n
            )
            if total == 0:
                if self.smoothing:
                    precision = 1 / (2 ** n)  # Smoothing
                else:
                    precision = 0
            else:
                precision = clipped / total
                if self.smoothing and precision == 0:
                    precision = 1 / (2 ** n)
            precisions.append(precision)

        # Compute geometric mean of precisions
        if any(p == 0 for p in precisions) and not self.smoothing:
            return 0.0

        log_precision_sum = sum(math.log(p) if p > 0 else -float('inf') for p in precisions)
        geo_mean = math.exp(log_precision_sum / len(precisions))

        # Compute brevity penalty
        c = len(candidate_tokens)
        r = len(reference_tokens)
        if c > r:
            bp = 1.0
        else:
            bp = math.exp(1 - r / c) if c > 0 else 0.0

        return bp * geo_mean

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute BLEU score for all samples"""
        valid_samples = self.validate_samples(samples)

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No valid samples with references"},
            )

        scores = []
        for sample in valid_samples:
            score = self._compute_bleu_single(sample.output, sample.reference)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={
                "max_ngram": self.max_ngram,
                "smoothing": self.smoothing,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=scores,
        )


class COMETMetric(BaseMetric):
    """
    COMET (Crosslingual Optimized Metric for Evaluation of Translation).
    Uses a trained model to evaluate translation quality.
    """

    name = "comet"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "Unbabel/wmt22-comet-da")
        self._model = None

    def _load_model(self):
        """Lazy load COMET model"""
        if self._model is None:
            try:
                from comet import download_model, load_from_checkpoint
                model_path = download_model(self.model_name)
                self._model = load_from_checkpoint(model_path)
            except ImportError:
                raise ImportError(
                    "COMET package is required. Install with: pip install unbabel-comet"
                )
        return self._model

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute COMET score for all samples"""
        valid_samples = self.validate_samples(samples)

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No valid samples with references"},
            )

        # Check for source text in input
        data = []
        for sample in valid_samples:
            data.append({
                "src": sample.input,  # Source text
                "mt": sample.output,   # Machine translation
                "ref": sample.reference,  # Reference translation
            })

        try:
            model = self._load_model()
            model_output = model.predict(data, batch_size=8, gpus=1)
            scores = model_output.scores
            avg_score = model_output.system_score
        except Exception as e:
            # Fallback: use simple semantic similarity
            return self._fallback_similarity(valid_samples)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={
                "model": self.model_name,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=scores,
        )

    def _fallback_similarity(self, samples: List[EvalSample]) -> MetricResult:
        """Fallback similarity computation when COMET is unavailable"""
        # Use simple token overlap as fallback
        scores = []
        for sample in samples:
            out_tokens = set(self.tokenize(self.normalize_text(sample.output)))
            ref_tokens = set(self.tokenize(self.normalize_text(sample.reference)))
            if out_tokens or ref_tokens:
                jaccard = len(out_tokens & ref_tokens) / len(out_tokens | ref_tokens)
            else:
                jaccard = 0.0
            scores.append(jaccard)

        return MetricResult(
            name=self.name,
            score=sum(scores) / len(scores) if scores else 0.0,
            details={
                "fallback": True,
                "method": "jaccard_similarity",
                "num_samples": len(samples),
            },
            per_sample_scores=scores,
        )
