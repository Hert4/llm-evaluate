"""
Chat Metrics: Win Rate, Pairwise Comparison (LLM-as-Judge)
"""
from typing import List, Dict, Any, Optional, Tuple
import re

from .base import BaseMetric, LLMBasedMetric, MetricResult
from ..data_parsers.base import EvalSample
from ..config import OpenAIConfig


class WinRateMetric(LLMBasedMetric):
    """
    Win Rate using LLM as judge (similar to AlpacaEval).
    Compares model output against a reference/baseline.
    """

    name = "win_rate"

    JUDGE_PROMPT = """You are an impartial judge evaluating two AI assistant responses to a user question.

User Question: {question}

Response A:
{response_a}

Response B (Baseline):
{response_b}

Compare the two responses based on:
1. Helpfulness: Does the response address the user's needs?
2. Accuracy: Is the information correct?
3. Clarity: Is the response well-organized and easy to understand?
4. Completeness: Does it fully answer the question?

Which response is better overall?
Answer ONLY with one of: "A", "B", or "TIE" (if they are equally good)."""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(openai_config, config)
        self.baseline_field = self.config.get("baseline_field", "reference")

    def _judge_pair(
        self,
        question: str,
        response_a: str,
        response_b: str
    ) -> str:
        """Judge which response is better"""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response_a=response_a,
            response_b=response_b
        )
        response = self._call_llm(prompt)

        response_upper = response.upper().strip()
        if "TIE" in response_upper:
            return "tie"
        elif response_upper.startswith("A") or "RESPONSE A" in response_upper:
            return "a"
        elif response_upper.startswith("B") or "RESPONSE B" in response_upper:
            return "b"
        else:
            return "tie"

    def _compute_single(self, sample: EvalSample) -> Dict[str, Any]:
        """Compute win rate for single sample"""
        baseline = sample.reference or ""
        if not baseline:
            return {"result": "no_baseline", "score": 0.5}

        result = self._judge_pair(sample.input, sample.output, baseline)

        if result == "a":
            return {"result": "win", "score": 1.0}
        elif result == "b":
            return {"result": "loss", "score": 0.0}
        else:
            return {"result": "tie", "score": 0.5}

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Win Rate for all samples"""
        valid_samples = [s for s in samples if s.reference]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with baseline/reference"},
            )

        wins = 0
        losses = 0
        ties = 0
        per_sample_scores = []

        for sample in valid_samples:
            result = self._compute_single(sample)
            per_sample_scores.append(result["score"])

            if result["result"] == "win":
                wins += 1
            elif result["result"] == "loss":
                losses += 1
            else:
                ties += 1

        total = len(valid_samples)
        win_rate = (wins + 0.5 * ties) / total  # Count ties as half wins

        return MetricResult(
            name=self.name,
            score=win_rate,
            details={
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "total": total,
                "win_rate_strict": wins / total,  # Wins only
                "win_rate_with_ties": win_rate,  # Wins + half ties
            },
            per_sample_scores=per_sample_scores,
        )


class PairwiseComparisonMetric(LLMBasedMetric):
    """
    Pairwise Comparison: Compare two models directly on same inputs.
    Useful for A/B testing models.
    """

    name = "pairwise_comparison"

    COMPARISON_PROMPT = """Compare these two AI responses to the same question.

Question: {question}

Model A Response:
{response_a}

Model B Response:
{response_b}

Evaluate on a scale from -2 to +2:
-2: Model B is much better
-1: Model B is slightly better
 0: Equal quality
+1: Model A is slightly better
+2: Model A is much better

Provide your score and brief reasoning.

Score: """

    def __init__(
        self,
        openai_config: OpenAIConfig,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(openai_config, config)

    def _compare_responses(
        self,
        question: str,
        response_a: str,
        response_b: str
    ) -> float:
        """Compare two responses, return score for A vs B"""
        prompt = self.COMPARISON_PROMPT.format(
            question=question,
            response_a=response_a,
            response_b=response_b
        )
        response = self._call_llm(prompt)

        # Extract score
        match = re.search(r'[-+]?\d', response)
        if match:
            score = int(match.group())
            return max(-2, min(2, score))  # Clamp to [-2, 2]
        return 0

    def compare_models(
        self,
        samples_a: List[EvalSample],
        samples_b: List[EvalSample],
    ) -> MetricResult:
        """
        Compare two models on the same inputs.

        Args:
            samples_a: Outputs from model A
            samples_b: Outputs from model B

        Returns:
            MetricResult with comparison scores
        """
        if len(samples_a) != len(samples_b):
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "Sample lists must have same length"},
            )

        scores = []
        a_wins = 0
        b_wins = 0
        ties = 0

        for sample_a, sample_b in zip(samples_a, samples_b):
            if sample_a.input != sample_b.input:
                continue  # Skip mismatched inputs

            score = self._compare_responses(
                sample_a.input,
                sample_a.output,
                sample_b.output
            )
            scores.append(score)

            if score > 0:
                a_wins += 1
            elif score < 0:
                b_wins += 1
            else:
                ties += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0
        # Normalize to 0-1 range (0 = B wins, 0.5 = tie, 1 = A wins)
        normalized = (avg_score + 2) / 4

        return MetricResult(
            name=self.name,
            score=normalized,
            details={
                "raw_score": avg_score,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "total": len(scores),
                "a_win_rate": a_wins / len(scores) if scores else 0,
                "b_win_rate": b_wins / len(scores) if scores else 0,
            },
            per_sample_scores=scores,
        )

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """
        For single-model evaluation, compare against reference.
        """
        valid_samples = [s for s in samples if s.reference]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with reference"},
            )

        scores = []
        wins = 0
        losses = 0
        ties = 0

        for sample in valid_samples:
            score = self._compare_responses(
                sample.input,
                sample.output,
                sample.reference
            )
            scores.append(score)

            if score > 0:
                wins += 1
            elif score < 0:
                losses += 1
            else:
                ties += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0
        normalized = (avg_score + 2) / 4

        return MetricResult(
            name=self.name,
            score=normalized,
            details={
                "raw_score": avg_score,
                "wins_vs_reference": wins,
                "losses_vs_reference": losses,
                "ties": ties,
                "total": len(scores),
            },
            per_sample_scores=scores,
        )
