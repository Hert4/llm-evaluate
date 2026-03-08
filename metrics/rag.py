"""
RAG Metrics: Faithfulness, Context Precision, Context Recall, Answer Relevancy
"""
from typing import List, Dict, Any, Optional
import re

from .base import BaseMetric, LLMBasedMetric, MetricResult
from ..data_parsers.base import EvalSample
from ..config import OpenAIConfig


class FaithfulnessMetric(LLMBasedMetric):
    """
    Faithfulness: Measures if the answer is grounded in the context.
    Detects hallucination.
    """

    name = "faithfulness"
    requires_context = True

    CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer. List each claim on a separate line.

Answer: {answer}

Claims (one per line):"""

    VERIFICATION_PROMPT = """Given the following context and claim, determine if the claim is supported by the context.

Context: {context}

Claim: {claim}

Answer only "SUPPORTED" or "NOT SUPPORTED"."""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(openai_config, config)

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract claims from answer using LLM"""
        prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        response = self._call_llm(prompt)
        claims = [c.strip() for c in response.split('\n') if c.strip()]
        return claims

    def _verify_claim(self, claim: str, context: str) -> bool:
        """Verify if claim is supported by context"""
        prompt = self.VERIFICATION_PROMPT.format(context=context, claim=claim)
        response = self._call_llm(prompt)
        return "SUPPORTED" in response.upper()

    def _compute_single(self, sample: EvalSample) -> float:
        """Compute faithfulness for single sample"""
        if not sample.context:
            return 0.0

        claims = self._extract_claims(sample.output)
        if not claims:
            return 1.0  # No claims = faithful by default

        supported_count = 0
        for claim in claims:
            if self._verify_claim(claim, sample.context):
                supported_count += 1

        return supported_count / len(claims)

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Faithfulness for all samples"""
        valid_samples = [s for s in samples if s.context]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with context"},
            )

        scores = []
        for sample in valid_samples:
            score = self._compute_single(sample)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={
                "num_samples": len(valid_samples),
            },
            per_sample_scores=scores,
        )


class ContextPrecisionMetric(LLMBasedMetric):
    """
    Context Precision: Measures if retrieved context is relevant.
    """

    name = "context_precision"
    requires_context = True

    RELEVANCE_PROMPT = """Given the question and context chunk, determine if the context is relevant to answering the question.

Question: {question}

Context chunk: {chunk}

Answer only "RELEVANT" or "NOT RELEVANT"."""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(openai_config, config)
        self.chunk_size = self.config.get("chunk_size", 200)

    def _split_context(self, context: str) -> List[str]:
        """Split context into chunks"""
        words = context.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i+self.chunk_size])
            chunks.append(chunk)
        return chunks or [context]

    def _is_relevant(self, question: str, chunk: str) -> bool:
        """Check if chunk is relevant to question"""
        prompt = self.RELEVANCE_PROMPT.format(question=question, chunk=chunk)
        response = self._call_llm(prompt)
        return "RELEVANT" in response.upper() and "NOT" not in response.upper()

    def _compute_single(self, sample: EvalSample) -> float:
        """Compute context precision for single sample"""
        if not sample.context:
            return 0.0

        chunks = self._split_context(sample.context)
        relevant_count = 0

        for chunk in chunks:
            if self._is_relevant(sample.input, chunk):
                relevant_count += 1

        return relevant_count / len(chunks) if chunks else 0.0

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Context Precision for all samples"""
        valid_samples = [s for s in samples if s.context]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with context"},
            )

        scores = []
        for sample in valid_samples:
            score = self._compute_single(sample)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={"num_samples": len(valid_samples)},
            per_sample_scores=scores,
        )


class ContextRecallMetric(LLMBasedMetric):
    """
    Context Recall: Measures if context contains all information needed for the reference answer.
    """

    name = "context_recall"
    requires_context = True
    requires_reference = True

    RECALL_PROMPT = """Given the context and reference answer, determine what fraction of the reference answer's information is present in the context.

Context: {context}

Reference Answer: {reference}

On a scale of 0 to 1, what fraction of the reference answer's key information is found in the context?
Answer with just the number (e.g., 0.8)."""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(openai_config, config)

    def _compute_single(self, sample: EvalSample) -> float:
        """Compute context recall for single sample"""
        if not sample.context or not sample.reference:
            return 0.0

        prompt = self.RECALL_PROMPT.format(
            context=sample.context,
            reference=sample.reference
        )
        response = self._call_llm(prompt)

        try:
            score = float(re.search(r"[\d.]+", response).group())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5  # Default

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Context Recall for all samples"""
        valid_samples = [s for s in samples if s.context and s.reference]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with context and reference"},
            )

        scores = []
        for sample in valid_samples:
            score = self._compute_single(sample)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={"num_samples": len(valid_samples)},
            per_sample_scores=scores,
        )


class AnswerRelevancyMetric(LLMBasedMetric):
    """
    Answer Relevancy: Measures if the answer is relevant to the question.
    """

    name = "answer_relevancy"

    RELEVANCY_PROMPT = """Rate how relevant the following answer is to the question on a scale of 0 to 1.

Question: {question}

Answer: {answer}

Consider:
- Does the answer address the question?
- Is the answer on-topic?
- Does it provide useful information?

Answer with just the relevancy score (e.g., 0.85)."""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(openai_config, config)

    def _compute_single(self, sample: EvalSample) -> float:
        """Compute answer relevancy for single sample"""
        prompt = self.RELEVANCY_PROMPT.format(
            question=sample.input,
            answer=sample.output
        )
        response = self._call_llm(prompt)

        try:
            score = float(re.search(r"[\d.]+", response).group())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Answer Relevancy for all samples"""
        if not samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples provided"},
            )

        scores = []
        for sample in samples:
            score = self._compute_single(sample)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return MetricResult(
            name=self.name,
            score=avg_score,
            details={"num_samples": len(samples)},
            per_sample_scores=scores,
        )
