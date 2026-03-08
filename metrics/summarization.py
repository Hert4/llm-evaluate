"""
Summarization Metrics: ROUGE, G-Eval
"""
from typing import List, Dict, Any, Optional
from collections import Counter
import re

from .base import BaseMetric, LLMBasedMetric, MetricResult
from ..data_parsers.base import EvalSample
from ..config import OpenAIConfig


class ROUGEMetric(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation).
    Measures overlap between generated summary and reference.
    """

    name = "rouge"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rouge_types = self.config.get("rouge_types", ["rouge1", "rouge2", "rougeL"])

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts"""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute LCS length using dynamic programming"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _compute_rouge_n(
        self,
        candidate: List[str],
        reference: List[str],
        n: int
    ) -> Dict[str, float]:
        """Compute ROUGE-N score"""
        cand_ngrams = self._get_ngrams(candidate, n)
        ref_ngrams = self._get_ngrams(reference, n)

        overlap = sum((cand_ngrams & ref_ngrams).values())
        cand_total = sum(cand_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precision = overlap / cand_total if cand_total > 0 else 0.0
        recall = overlap / ref_total if ref_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def _compute_rouge_l(
        self,
        candidate: List[str],
        reference: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE-L score using LCS"""
        lcs_len = self._lcs_length(candidate, reference)

        precision = lcs_len / len(candidate) if candidate else 0.0
        recall = lcs_len / len(reference) if reference else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def _compute_single(
        self,
        candidate: str,
        reference: str
    ) -> Dict[str, Dict[str, float]]:
        """Compute all ROUGE scores for single pair"""
        cand_tokens = self.tokenize(self.normalize_text(candidate))
        ref_tokens = self.tokenize(self.normalize_text(reference))

        results = {}

        if "rouge1" in self.rouge_types:
            results["rouge1"] = self._compute_rouge_n(cand_tokens, ref_tokens, 1)

        if "rouge2" in self.rouge_types:
            results["rouge2"] = self._compute_rouge_n(cand_tokens, ref_tokens, 2)

        if "rougeL" in self.rouge_types:
            results["rougeL"] = self._compute_rouge_l(cand_tokens, ref_tokens)

        return results

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute ROUGE scores for all samples"""
        valid_samples = self.validate_samples(samples)

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No valid samples with references"},
            )

        all_scores = {rt: [] for rt in self.rouge_types}

        for sample in valid_samples:
            sample_scores = self._compute_single(sample.output, sample.reference)
            for rt in self.rouge_types:
                if rt in sample_scores:
                    all_scores[rt].append(sample_scores[rt]["f1"])

        # Aggregate scores
        avg_scores = {}
        for rt in self.rouge_types:
            if all_scores[rt]:
                avg_scores[rt] = sum(all_scores[rt]) / len(all_scores[rt])
            else:
                avg_scores[rt] = 0.0

        # Main score is average of all ROUGE types
        main_score = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0

        return MetricResult(
            name=self.name,
            score=main_score,
            details={
                "rouge_scores": avg_scores,
                "rouge_types": self.rouge_types,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=all_scores.get("rougeL", all_scores.get("rouge1", [])),
        )


class GEvalMetric(LLMBasedMetric):
    """
    G-Eval: LLM-based evaluation using GPT-4 or similar models.
    Evaluates on multiple criteria: coherence, fluency, relevance, consistency.
    """

    name = "g_eval"

    EVALUATION_PROMPT = """You are an expert evaluator. Please evaluate the following summary on a scale of 1-5 for each criterion.

Source Text:
{source}

Summary to evaluate:
{summary}

Reference Summary (if available):
{reference}

Evaluate on these criteria:
1. Coherence (1-5): Is the summary well-organized and easy to follow?
2. Fluency (1-5): Is the summary grammatically correct and natural?
3. Relevance (1-5): Does the summary capture the important information?
4. Consistency (1-5): Is the summary factually consistent with the source?

Respond in JSON format:
{{"coherence": <1-5>, "fluency": <1-5>, "relevance": <1-5>, "consistency": <1-5>}}"""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(openai_config, config)
        self.criteria = self.config.get("criteria", ["coherence", "fluency", "relevance", "consistency"])

    def _evaluate_single(self, sample: EvalSample) -> Dict[str, float]:
        """Evaluate single sample using LLM"""
        prompt = self.EVALUATION_PROMPT.format(
            source=sample.input,
            summary=sample.output,
            reference=sample.reference or "Not provided"
        )

        try:
            response = self._call_llm(prompt)
            # Parse JSON response
            import json
            scores = json.loads(response)
            # Normalize to 0-1 scale
            return {k: v / 5.0 for k, v in scores.items() if k in self.criteria}
        except Exception:
            # Return default scores on error
            return {c: 0.5 for c in self.criteria}

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute G-Eval scores for all samples"""
        all_scores = {c: [] for c in self.criteria}
        per_sample = []

        for sample in samples:
            scores = self._evaluate_single(sample)
            avg = sum(scores.values()) / len(scores) if scores else 0.0
            per_sample.append(avg)
            for c in self.criteria:
                all_scores[c].append(scores.get(c, 0.5))

        # Aggregate
        avg_scores = {c: sum(s) / len(s) if s else 0.0 for c, s in all_scores.items()}
        main_score = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0

        return MetricResult(
            name=self.name,
            score=main_score,
            details={
                "criteria_scores": avg_scores,
                "criteria": self.criteria,
                "num_samples": len(samples),
            },
            per_sample_scores=per_sample,
        )
