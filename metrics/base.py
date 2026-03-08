"""
Base Metric Class
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np

from ..data_parsers.base import EvalSample
from ..config import OpenAIConfig


@dataclass
class MetricResult:
    """Result of metric evaluation"""
    name: str
    score: float  # Main score (0-1 or 0-100 depending on metric)
    details: Dict[str, Any] = field(default_factory=dict)
    per_sample_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "details": self.details,
            "per_sample_scores": self.per_sample_scores,
            "metadata": self.metadata,
        }

    @property
    def score_pct(self) -> float:
        """Score as percentage (0-100)"""
        return self.score * 100 if self.score <= 1 else self.score

    def __repr__(self) -> str:
        return f"{self.name}: {self.score:.4f}"


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.
    """

    name: str = "base"
    requires_reference: bool = True
    requires_context: bool = False
    requires_llm: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metric.

        Args:
            config: Metric-specific configuration
        """
        self.config = config or {}

    @abstractmethod
    def compute(
        self,
        samples: List[EvalSample],
        **kwargs
    ) -> MetricResult:
        """
        Compute metric for a list of samples.

        Args:
            samples: List of evaluation samples
            **kwargs: Additional arguments

        Returns:
            MetricResult with scores
        """
        pass

    def compute_single(
        self,
        sample: EvalSample,
        **kwargs
    ) -> float:
        """
        Compute metric for a single sample.
        Default implementation calls compute on single-element list.
        """
        result = self.compute([sample], **kwargs)
        if result.per_sample_scores:
            return result.per_sample_scores[0]
        return result.score

    def validate_samples(self, samples: List[EvalSample]) -> List[EvalSample]:
        """Validate and filter samples for this metric"""
        valid_samples = []
        for sample in samples:
            if self.requires_reference and not sample.reference:
                continue
            if self.requires_context and not sample.context:
                continue
            valid_samples.append(sample)
        return valid_samples

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        # Lowercase, strip whitespace
        text = text.lower().strip()
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization"""
        return text.split()


class LLMBasedMetric(BaseMetric):
    """
    Base class for metrics that use LLM as judge.
    """

    requires_llm = True

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.openai_config = openai_config
        self._client = None

    def _get_client(self):
        """Lazy initialize OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.openai_config.api_key,
                    base_url=self.openai_config.base_url,
                    timeout=self.openai_config.timeout,
                    max_retries=self.openai_config.max_retries,
                )
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
        return self._client

    def _call_llm(self, prompt: str, **kwargs) -> str:
        """Call LLM and return response"""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.openai_config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.openai_config.temperature),
            max_tokens=kwargs.get("max_tokens", self.openai_config.max_tokens),
        )
        return response.choices[0].message.content.strip()


class AggregatedMetric:
    """Helper for aggregating multiple metric results"""

    @staticmethod
    def mean(scores: List[float]) -> float:
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def median(scores: List[float]) -> float:
        return float(np.median(scores)) if scores else 0.0

    @staticmethod
    def std(scores: List[float]) -> float:
        return float(np.std(scores)) if scores else 0.0

    @staticmethod
    def min(scores: List[float]) -> float:
        return float(np.min(scores)) if scores else 0.0

    @staticmethod
    def max(scores: List[float]) -> float:
        return float(np.max(scores)) if scores else 0.0

    @staticmethod
    def percentile(scores: List[float], p: float) -> float:
        return float(np.percentile(scores, p)) if scores else 0.0
