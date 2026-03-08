"""
Tool Calling Metrics: AST Accuracy, Task Success Rate
"""
from typing import List, Dict, Any, Optional
import json
import re

from .base import BaseMetric, MetricResult
from ..data_parsers.base import EvalSample


class ASTAccuracyMetric(BaseMetric):
    """
    AST (Abstract Syntax Tree) Accuracy for tool/function calling.
    Checks if the model called the correct function with correct arguments.
    """

    name = "ast_accuracy"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.strict_args = self.config.get("strict_args", True)
        self.check_arg_types = self.config.get("check_arg_types", False)

    def _parse_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        """Parse tool call into standard format"""
        if isinstance(tool_call, str):
            try:
                tool_call = json.loads(tool_call)
            except json.JSONDecodeError:
                return {}

        if isinstance(tool_call, dict):
            # Handle various formats
            name = tool_call.get("name") or tool_call.get("function", {}).get("name") or tool_call.get("tool")
            args = tool_call.get("arguments") or tool_call.get("parameters") or tool_call.get("function", {}).get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            return {"name": name, "arguments": args}

        return {}

    def _compare_args(self, actual: Dict, expected: Dict) -> bool:
        """Compare arguments"""
        if not self.strict_args:
            # Just check that all expected args are present
            for key, value in expected.items():
                if key not in actual:
                    return False
                if self.check_arg_types and type(actual[key]) != type(value):
                    return False
            return True
        else:
            # Strict comparison
            return actual == expected

    def _compute_single(self, sample: EvalSample) -> Dict[str, Any]:
        """Compute accuracy for single sample"""
        actual_calls = sample.tool_calls or []
        expected_calls = sample.expected_tool_calls or []

        if not expected_calls:
            # If no expected calls, model should not make any calls
            if not actual_calls:
                return {"correct": True, "name_match": True, "args_match": True}
            else:
                return {"correct": False, "name_match": False, "args_match": False}

        if not actual_calls:
            return {"correct": False, "name_match": False, "args_match": False}

        # Compare each call
        name_matches = 0
        full_matches = 0

        for expected in expected_calls:
            expected_parsed = self._parse_tool_call(expected)
            expected_name = expected_parsed.get("name")
            expected_args = expected_parsed.get("arguments", {})

            for actual in actual_calls:
                actual_parsed = self._parse_tool_call(actual)
                actual_name = actual_parsed.get("name")
                actual_args = actual_parsed.get("arguments", {})

                if actual_name == expected_name:
                    name_matches += 1
                    if self._compare_args(actual_args, expected_args):
                        full_matches += 1
                    break

        total_expected = len(expected_calls)
        name_accuracy = name_matches / total_expected if total_expected > 0 else 0.0
        full_accuracy = full_matches / total_expected if total_expected > 0 else 0.0

        return {
            "correct": full_matches == total_expected,
            "name_match": name_matches == total_expected,
            "name_accuracy": name_accuracy,
            "full_accuracy": full_accuracy,
        }

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute AST Accuracy for all samples"""
        valid_samples = [s for s in samples if s.tool_calls is not None or s.expected_tool_calls is not None]

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples with tool calls"},
            )

        correct_count = 0
        name_match_count = 0
        per_sample_scores = []

        for sample in valid_samples:
            result = self._compute_single(sample)
            if result["correct"]:
                correct_count += 1
                per_sample_scores.append(1.0)
            else:
                per_sample_scores.append(result.get("full_accuracy", 0.0))
            if result["name_match"]:
                name_match_count += 1

        accuracy = correct_count / len(valid_samples)
        name_accuracy = name_match_count / len(valid_samples)

        return MetricResult(
            name=self.name,
            score=accuracy,
            details={
                "full_match_accuracy": accuracy,
                "name_match_accuracy": name_accuracy,
                "correct_count": correct_count,
                "num_samples": len(valid_samples),
            },
            per_sample_scores=per_sample_scores,
        )


class TaskSuccessRateMetric(BaseMetric):
    """
    Task Success Rate.
    Measures if a multi-step task was completed successfully (all steps correct).
    """

    name = "task_success_rate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.partial_credit = self.config.get("partial_credit", False)

    def _evaluate_task(self, sample: EvalSample) -> Dict[str, Any]:
        """Evaluate task completion"""
        # Check if all required tool calls were made
        actual_calls = sample.tool_calls or []
        expected_calls = sample.expected_tool_calls or []

        if not expected_calls:
            # Single-turn task - check output matches reference
            if sample.reference:
                output_normalized = self.normalize_text(sample.output)
                ref_normalized = self.normalize_text(sample.reference)
                success = output_normalized == ref_normalized or ref_normalized in output_normalized
                return {"success": success, "steps_completed": 1 if success else 0, "total_steps": 1}
            return {"success": True, "steps_completed": 1, "total_steps": 1}

        # Multi-step task
        steps_completed = 0
        for expected in expected_calls:
            expected_name = expected.get("name") or expected.get("function", {}).get("name")
            for actual in actual_calls:
                actual_name = actual.get("name") or actual.get("function", {}).get("name")
                if actual_name == expected_name:
                    steps_completed += 1
                    break

        total_steps = len(expected_calls)
        success = steps_completed == total_steps

        return {
            "success": success,
            "steps_completed": steps_completed,
            "total_steps": total_steps,
            "completion_rate": steps_completed / total_steps if total_steps > 0 else 0.0
        }

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Task Success Rate for all samples"""
        if not samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples provided"},
            )

        success_count = 0
        total_steps = 0
        completed_steps = 0
        per_sample_scores = []

        for sample in samples:
            result = self._evaluate_task(sample)
            if result["success"]:
                success_count += 1
                per_sample_scores.append(1.0)
            else:
                if self.partial_credit:
                    per_sample_scores.append(result.get("completion_rate", 0.0))
                else:
                    per_sample_scores.append(0.0)

            total_steps += result["total_steps"]
            completed_steps += result["steps_completed"]

        success_rate = success_count / len(samples)

        return MetricResult(
            name=self.name,
            score=success_rate,
            details={
                "success_rate": success_rate,
                "success_count": success_count,
                "total_tasks": len(samples),
                "step_completion_rate": completed_steps / total_steps if total_steps > 0 else 0.0,
            },
            per_sample_scores=per_sample_scores,
        )
