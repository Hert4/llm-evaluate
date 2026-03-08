"""
Coding Metrics: Pass@K
"""
from typing import List, Dict, Any, Optional, Callable
import subprocess
import tempfile
import os
import sys
import math
from pathlib import Path
import itertools

from .base import BaseMetric, MetricResult
from ..data_parsers.base import EvalSample


class PassAtKMetric(BaseMetric):
    """
    Pass@K.
    Measures if generated code passes test cases.
    """

    name = "pass_at_k"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k_values = self.config.get("k_values", [1, 5, 10])
        self.timeout = self.config.get("timeout", 30)
        self.language = self.config.get("language", "python")

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks or plain text"""
        import re

        # Try to find code blocks
        code_block_pattern = r"```(?:python|py|javascript|js|java|cpp|c\+\+|c|go|rust)?\n?(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Return as is
        return text.strip()

    def _run_python_code(self, code: str, test_code: str) -> Dict[str, Any]:
        """Run Python code with tests"""
        full_code = f"{code}\n\n{test_code}"

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            passed = result.returncode == 0
            return {
                "passed": passed,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": None if passed else result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stdout": "",
                "stderr": "",
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "passed": False,
                "stdout": "",
                "stderr": "",
                "error": str(e),
            }
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def _run_tests(self, sample: EvalSample) -> Dict[str, Any]:
        """Run tests for a sample"""
        code = self._extract_code(sample.output)

        if not sample.test_cases:
            # No test cases - just check if code is syntactically valid
            try:
                compile(code, '<string>', 'exec')
                return {"passed": True, "num_passed": 1, "num_total": 1}
            except SyntaxError as e:
                return {"passed": False, "num_passed": 0, "num_total": 1, "error": str(e)}

        # Run each test case
        num_passed = 0
        errors = []

        for test_case in sample.test_cases:
            if isinstance(test_case, dict):
                test_code = test_case.get("test", test_case.get("code", ""))
            else:
                test_code = str(test_case)

            result = self._run_python_code(code, test_code)
            if result["passed"]:
                num_passed += 1
            else:
                errors.append(result.get("error", "Unknown error"))

        all_passed = num_passed == len(sample.test_cases)

        return {
            "passed": all_passed,
            "num_passed": num_passed,
            "num_total": len(sample.test_cases),
            "errors": errors if not all_passed else None,
        }

    def _pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        Compute Pass@K.
        n: total samples
        c: number of correct samples
        k: k value
        """
        if n - c < k:
            return 1.0
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute Pass@K for all samples"""
        if not samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No samples provided"},
            )

        # Run tests for each sample
        results = []
        for sample in samples:
            result = self._run_tests(sample)
            results.append(result)

        # Compute metrics
        n = len(samples)
        c = sum(1 for r in results if r["passed"])

        pass_at_k = {}
        for k in self.k_values:
            if k <= n:
                pass_at_k[f"pass@{k}"] = self._pass_at_k(n, c, k)
            else:
                pass_at_k[f"pass@{k}"] = c / n if n > 0 else 0.0

        main_k = self.k_values[0] if self.k_values else 1
        main_score = pass_at_k.get(f"pass@{main_k}", 0.0)

        per_sample_scores = [1.0 if r["passed"] else 0.0 for r in results]

        return MetricResult(
            name=self.name,
            score=main_score,
            details={
                "pass_at_k": pass_at_k,
                "correct_count": c,
                "total_samples": n,
                "k_values": self.k_values,
            },
            per_sample_scores=per_sample_scores,
        )
