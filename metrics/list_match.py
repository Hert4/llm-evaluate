"""
List Match Metrics: Precision@K, Recall@K, NDCG@K for JSON list output vs reference.

Designed for tasks where both output and reference are JSON lists of items
(e.g., top-K product recommendations, top-K forecasted items).

Automatically parses JSON from output/reference strings, extracts item values,
and computes set-based and rank-aware metrics.
"""
import json
import math
import re
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm

from .base import BaseMetric, MetricResult
from ..data_parsers.base import EvalSample


def _extract_items(text: str) -> List[str]:
    """
    Extract ordered list of item values from JSON string.

    Handles formats like:
      - [{"ProductCode": "A"}, {"ProductCode": "B"}]
      - {"A": [{"B": "X"}, {"B": "Y"}]}
      - ["item1", "item2"]
      - Plain text fallback: split by newlines

    Returns list of string values in order.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Strip markdown code block wrappers (```json ... ``` or ``` ... ```)
    md_match = re.match(r'^```(?:\w+)?\s*\n(.*?)\n\s*```$', text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    # Try parsing as JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract from non-standard JSON
        # e.g., pretty-printed or with trailing commas
        try:
            # Remove trailing commas before ] or }
            cleaned = re.sub(r',\s*([}\]])', r'\1', text)
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: split by newlines, strip
            return [line.strip() for line in text.split('\n') if line.strip()]

    return _flatten_values(data)


def _flatten_values(data: Any) -> List[str]:
    """
    Recursively extract leaf string/number values from nested JSON,
    preserving order.

    For a list of dicts each with one value, extracts that value.
    For a dict with one key pointing to a list, recurses into the list.
    """
    if isinstance(data, str):
        return [data]

    if isinstance(data, (int, float)):
        return [str(data)]

    if isinstance(data, list):
        # Check if it's a list of dicts (each with one meaningful value)
        if all(isinstance(item, dict) for item in data):
            results = []
            for item in data:
                vals = list(item.values())
                if len(vals) == 1:
                    results.append(str(vals[0]))
                else:
                    # Multiple values per dict — take all
                    results.extend(str(v) for v in vals)
            return results

        # List of plain values
        if all(isinstance(item, (str, int, float)) for item in data):
            return [str(item) for item in data]

        # Mixed list — recurse
        results = []
        for item in data:
            results.extend(_flatten_values(item))
        return results

    if isinstance(data, dict):
        # Single-key dict wrapping a list (e.g., {"A": [...]})
        if len(data) == 1:
            return _flatten_values(list(data.values())[0])

        # Multi-key dict — take all values
        results = []
        for v in data.values():
            results.extend(_flatten_values(v))
        return results

    return []


class ListMatchMetric(BaseMetric):
    """
    List Match metric for comparing ordered lists of items.

    Parses JSON list from output and reference strings, then computes:
    - precision@k: fraction of predicted top-k items that are relevant
    - recall@k: fraction of reference items found in predicted top-k
    - ndcg@k: rank-aware score (items matched higher = better)
    - hit_rate: fraction of samples with at least 1 correct item

    Works with any JSON list format — auto-extracts item values.
    """

    name = "list_match"
    requires_reference = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k_values = self.config.get("k_values", [1, 3, 5])
        self.ignore_case = self.config.get("ignore_case", False)

    def _parse_items(self, text: str) -> List[str]:
        """Parse and optionally normalize items from text."""
        items = _extract_items(text)
        if self.ignore_case:
            items = [item.lower() for item in items]
        return items

    def _precision_at_k(self, pred: List[str], ref_set: set, k: int) -> float:
        """Precision@K: of the top-k predictions, how many are in reference?"""
        top_k = pred[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for item in top_k if item in ref_set)
        return hits / len(top_k)

    def _recall_at_k(self, pred: List[str], ref_set: set, k: int) -> float:
        """Recall@K: of all reference items, how many appear in top-k predictions?"""
        if not ref_set:
            return 0.0
        top_k = pred[:k]
        hits = sum(1 for item in top_k if item in ref_set)
        return hits / len(ref_set)

    def _ndcg_at_k(self, pred: List[str], ref_set: set, k: int) -> float:
        """NDCG@K: rank-aware metric — correct items ranked higher = better score."""
        top_k = pred[:k]
        # Binary relevance: 1 if in reference set, 0 otherwise
        relevances = [1.0 if item in ref_set else 0.0 for item in top_k]

        # DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

        # Ideal DCG: all relevant items at top
        n_relevant = min(len(ref_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_single(self, sample: EvalSample) -> Dict[str, Any]:
        """Compute all metrics for a single sample."""
        pred_items = self._parse_items(sample.output)
        ref_items = self._parse_items(sample.reference)
        ref_set = set(ref_items)

        result = {}
        for k in self.k_values:
            result[f"precision@{k}"] = self._precision_at_k(pred_items, ref_set, k)
            result[f"recall@{k}"] = self._recall_at_k(pred_items, ref_set, k)
            result[f"ndcg@{k}"] = self._ndcg_at_k(pred_items, ref_set, k)

        # Hit rate: at least 1 correct in all predictions
        result["hit"] = 1.0 if ref_set & set(pred_items) else 0.0

        # Count parsed items (useful for debugging)
        result["n_pred"] = len(pred_items)
        result["n_ref"] = len(ref_items)

        return result

    def compute(self, samples: List[EvalSample], **kwargs) -> MetricResult:
        """Compute List Match metrics for all samples."""
        valid_samples = self.validate_samples(samples)

        if not valid_samples:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No valid samples with references"},
            )

        # Collect all per-sample scores
        all_metrics: Dict[str, List[float]] = {}
        per_sample_scores = []

        for sample in tqdm(valid_samples, desc="List Match", unit="sample"):
            scores = self._compute_single(sample)

            # Use max-k recall as the per-sample summary score
            max_k = max(self.k_values)
            per_sample_scores.append(scores.get(f"recall@{max_k}", 0.0))

            for key, val in scores.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(val)

        # Aggregate
        avg_metrics = {
            key: sum(vals) / len(vals)
            for key, vals in all_metrics.items()
            if key not in ("n_pred", "n_ref")
        }

        # Main score = recall@max_k (most representative for "did we find the right items?")
        max_k = max(self.k_values)
        main_score = avg_metrics.get(f"recall@{max_k}", 0.0)

        return MetricResult(
            name=self.name,
            score=main_score,
            details={
                **avg_metrics,
                "k_values": self.k_values,
                "num_samples": len(valid_samples),
                "avg_pred_items": sum(all_metrics.get("n_pred", [])) / len(valid_samples),
                "avg_ref_items": sum(all_metrics.get("n_ref", [])) / len(valid_samples),
            },
            per_sample_scores=per_sample_scores,
        )
