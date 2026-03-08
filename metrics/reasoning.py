"""
Reasoning Metrics: Accuracy (Multiple Choice)
"""
from typing import List, Dict, Any, Optional
import re

from .base import BaseMetric, MetricResult
from ..data_parsers.base import EvalSample


class AccuracyMetric(BaseMetric):
    """
    Accuracy for multiple choice and classification tasks.
    """

    name = "accuracy"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extract_answer = self.config.get("extract_answer", True)
        self.case_sensitive = self.config.get("case_sensitive", False)

    def _extract_answer_from_text(self, text: str, choices: Optional[List[str]] = None) -> str:
        """Extract answer letter or text from model output"""
        if not text:
            return ""

        text = text.strip()

        # Try to find answer pattern like "A)", "(A)", "Answer: A", etc.
        patterns = [
            r"(?:answer|choice|option|đáp án|chọn)[:\s]*([A-Za-z])\)?",
            r"^([A-Za-z])\)",
            r"^\(([A-Za-z])\)",
            r"^([A-Za-z])\.",
            r"^([A-Za-z])$",
            r"\b([A-Za-z])\b$",  # Single letter at end
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1)
                if not self.case_sensitive:
                    answer = answer.upper()
                return answer

        # If choices provided, try to match full text
        if choices:
            text_lower = text.lower()
            for i, choice in enumerate(choices):
                if choice.lower() in text_lower:
                    return chr(65 + i)  # Convert to A, B, C, ...

        # Return first letter if single character response
        if len(text) == 1 and text.isalpha():
            return text.upper() if not self.case_sensitive else text

        return text

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""

        answer = answer.strip()
        if not self.case_sensitive:
            answer = answer.upper()

        return answer

    def _compute_single(self, sample: EvalSample) -> bool:
        """Check if answer is correct"""
        if self.extract_answer:
            predicted = self._extract_answer_from_text(sample.output, sample.choices)
        else:
            predicted = self._normalize_answer(sample.output)

        # Get correct answer
        correct = sample.correct_answer or sample.reference

        if not correct:
            return False

        correct = self._normalize_answer(correct)
        predicted = self._normalize_answer(predicted)

        return predicted == correct

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute accuracy for all samples"""
        valid_samples = [s for s in samples if s.correct_answer or s.reference]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with correct answers"},
            )

        correct_count = 0
        per_sample_scores = []

        for sample in valid_samples:
            is_correct = self._compute_single(sample)
            if is_correct:
                correct_count += 1
                per_sample_scores.append(1.0)
            else:
                per_sample_scores.append(0.0)

        accuracy = correct_count / len(valid_samples)

        return MetricResult(
            name=self.name,
            score=accuracy,
            details={
                "correct_count": correct_count,
                "total_samples": len(valid_samples),
                "accuracy_pct": accuracy * 100,
            },
            per_sample_scores=per_sample_scores,
        )
