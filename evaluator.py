"""
Main LLM Evaluator Class
========================
Orchestrates data parsing, ground truth generation, and metric computation.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from .config import EvalConfig, MetricType, TaskType, TASK_METRICS_MAP, OpenAIConfig
from .data_parsers import AutoParser, EvalSample
from .ground_truth import GroundTruthGenerator
from .metrics import (
    BaseMetric, MetricResult,
    BLEUMetric, COMETMetric,
    ROUGEMetric, GEvalMetric,
    ExactMatchMetric, TokenF1Metric,
    NDCGMetric, RecallAtKMetric, PrecisionAtKMetric, MRRMetric,
    ASTAccuracyMetric, TaskSuccessRateMetric,
    PassAtKMetric,
    AccuracyMetric,
    FaithfulnessMetric, ContextPrecisionMetric, ContextRecallMetric, AnswerRelevancyMetric,
    FactScoreMetric, IFEvalMetric,
    WinRateMetric, PairwiseComparisonMetric,
)

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """
    Main evaluation orchestrator.
    Handles data loading, ground truth generation, and metric computation.
    """

    # Metric registry
    METRIC_REGISTRY = {
        MetricType.BLEU: BLEUMetric,
        MetricType.COMET: COMETMetric,
        MetricType.ROUGE: ROUGEMetric,
        MetricType.G_EVAL: GEvalMetric,
        MetricType.EXACT_MATCH: ExactMatchMetric,
        MetricType.TOKEN_F1: TokenF1Metric,
        MetricType.NDCG: NDCGMetric,
        MetricType.RECALL_AT_K: RecallAtKMetric,
        MetricType.PRECISION_AT_K: PrecisionAtKMetric,
        MetricType.MRR: MRRMetric,
        MetricType.AST_ACCURACY: ASTAccuracyMetric,
        MetricType.TASK_SUCCESS_RATE: TaskSuccessRateMetric,
        MetricType.PASS_AT_K: PassAtKMetric,
        MetricType.ACCURACY: AccuracyMetric,
        MetricType.FAITHFULNESS: FaithfulnessMetric,
        MetricType.CONTEXT_PRECISION: ContextPrecisionMetric,
        MetricType.CONTEXT_RECALL: ContextRecallMetric,
        MetricType.ANSWER_RELEVANCY: AnswerRelevancyMetric,
        MetricType.FACTSCORE: FactScoreMetric,
        MetricType.IFEVAL: IFEvalMetric,
        MetricType.WIN_RATE: WinRateMetric,
        MetricType.PAIRWISE_COMPARISON: PairwiseComparisonMetric,
    }

    # LLM-based metrics that need OpenAI config
    LLM_METRICS = {
        MetricType.G_EVAL,
        MetricType.FAITHFULNESS,
        MetricType.CONTEXT_PRECISION,
        MetricType.CONTEXT_RECALL,
        MetricType.ANSWER_RELEVANCY,
        MetricType.FACTSCORE,
        MetricType.WIN_RATE,
        MetricType.PAIRWISE_COMPARISON,
    }

    def __init__(self, config: Optional[EvalConfig] = None):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvalConfig()
        self.parser = AutoParser()
        self.gt_generator = None
        self._metrics_cache = {}

        if self.config.verbose:
            logging.basicConfig(level=getattr(logging, self.config.log_level))

    def _get_metric(self, metric_type: MetricType) -> BaseMetric:
        """Get or create metric instance"""
        if metric_type not in self._metrics_cache:
            metric_class = self.METRIC_REGISTRY.get(metric_type)
            if not metric_class:
                raise ValueError(f"Unknown metric type: {metric_type}")

            # Check if metric needs OpenAI config
            if metric_type in self.LLM_METRICS:
                self._metrics_cache[metric_type] = metric_class(
                    self.config.openai,
                    self.config.metrics.__dict__
                )
            else:
                self._metrics_cache[metric_type] = metric_class(
                    self.config.metrics.__dict__
                )

        return self._metrics_cache[metric_type]

    def _detect_task_type(self, samples: List[EvalSample]) -> TaskType:
        """Auto-detect task type from samples"""
        if not samples:
            return TaskType.CUSTOM

        sample = samples[0]

        # Check for specific task indicators
        if sample.tool_calls or sample.expected_tool_calls:
            return TaskType.TOOL_CALLING

        if sample.test_cases:
            return TaskType.CODING

        if sample.choices:
            return TaskType.REASONING

        if sample.relevance_scores or sample.candidates:
            return TaskType.RANKING

        if sample.context:
            return TaskType.RAG

        if sample.conversation_history:
            return TaskType.CHAT

        # Check reference length for QA vs summarization
        if sample.reference:
            ref_words = len(sample.reference.split())
            if ref_words < 20:
                return TaskType.QA
            else:
                return TaskType.SUMMARIZATION

        return TaskType.CUSTOM

    def _get_metrics_for_task(self, task_type: TaskType) -> List[MetricType]:
        """Get recommended metrics for task type"""
        if self.config.enabled_metrics:
            return self.config.enabled_metrics

        return TASK_METRICS_MAP.get(task_type, [
            MetricType.EXACT_MATCH,
            MetricType.TOKEN_F1,
        ])

    def load_data(
        self,
        data_source: Union[str, Path, Dict, List],
        **parser_kwargs
    ) -> List[EvalSample]:
        """
        Load and parse data from any source.

        Args:
            data_source: File path, dict, or list of data
            **parser_kwargs: Additional parser arguments

        Returns:
            List of parsed EvalSample
        """
        if parser_kwargs:
            self.parser = AutoParser(**parser_kwargs)

        samples = self.parser.parse(data_source)
        logger.info(f"Loaded {len(samples)} samples")
        return samples

    def generate_ground_truth(
        self,
        samples: List[EvalSample],
        task_type: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        max_concurrent: int = 10,
    ) -> List[EvalSample]:
        """
        Generate ground truth for samples using LLM.

        Args:
            samples: Samples without references
            task_type: Type of task for prompt selection
            custom_prompt: Custom prompt template
            max_concurrent: Max concurrent API calls

        Returns:
            Samples with generated references
        """
        if not self.gt_generator:
            self.gt_generator = GroundTruthGenerator(self.config.openai)

        # Filter samples that need GT
        samples_needing_gt = [s for s in samples if not s.reference]

        if not samples_needing_gt:
            logger.info("All samples already have references")
            return samples

        logger.info(f"Generating ground truth for {len(samples_needing_gt)} samples")

        results = self.gt_generator.generate_batch(
            samples_needing_gt,
            task_type=task_type or "qa",
            custom_prompt=custom_prompt,
            max_concurrent=max_concurrent,
        )

        # Update samples with GT
        samples = self.gt_generator.update_samples_with_gt(samples, results)

        success_count = sum(1 for r in results if r.success)
        logger.info(f"Generated GT for {success_count}/{len(samples_needing_gt)} samples")

        return samples

    def evaluate(
        self,
        samples: List[EvalSample],
        metrics: Optional[List[MetricType]] = None,
        task_type: Optional[TaskType] = None,
    ) -> Dict[str, MetricResult]:
        """
        Run evaluation on samples.

        Args:
            samples: Evaluation samples
            metrics: Specific metrics to run (auto-detect if None)
            task_type: Task type (auto-detect if None)

        Returns:
            Dict mapping metric name to MetricResult
        """
        if not samples:
            logger.warning("No samples to evaluate")
            return {}

        # Auto-detect task type if not specified
        if not task_type:
            task_type = self.config.task_type or self._detect_task_type(samples)
            logger.info(f"Detected task type: {task_type.value}")

        # Get metrics to run
        if not metrics:
            metrics = self._get_metrics_for_task(task_type)
            logger.info(f"Selected metrics: {[m.value for m in metrics]}")

        # Run each metric
        results = {}
        for metric_type in metrics:
            try:
                metric = self._get_metric(metric_type)
                logger.info(f"Computing {metric_type.value}...")
                result = metric.compute(samples)
                results[metric_type.value] = result
                logger.info(f"{metric_type.value}: {result.score:.4f}")
            except Exception as e:
                logger.error(f"Error computing {metric_type.value}: {e}")
                results[metric_type.value] = MetricResult(
                    name=metric_type.value,
                    score=0.0,
                    details={"error": str(e)},
                )

        return results

    def evaluate_from_file(
        self,
        file_path: Union[str, Path],
        metrics: Optional[List[MetricType]] = None,
        generate_gt: bool = False,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Convenience method to load data and evaluate in one step.

        Args:
            file_path: Path to data file
            metrics: Metrics to run
            generate_gt: Whether to generate ground truth
            **kwargs: Additional arguments for parsing/GT generation

        Returns:
            Evaluation results
        """
        samples = self.load_data(file_path)

        if generate_gt:
            samples = self.generate_ground_truth(
                samples,
                task_type=kwargs.get("task_type"),
                custom_prompt=kwargs.get("gt_prompt"),
            )

        return self.evaluate(samples, metrics=metrics)

    def save_results(
        self,
        results: Dict[str, MetricResult],
        output_path: Union[str, Path],
        samples: Optional[List[EvalSample]] = None,
        format: str = "json",
    ) -> None:
        """
        Save evaluation results to file.

        Args:
            results: Evaluation results
            output_path: Output file path
            samples: Optional samples for detailed output
            format: Output format (json, csv, html)
        """
        output_path = Path(output_path)

        if format == "json":
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "model": self.config.openai.model,
                    "task_type": self.config.task_type.value if self.config.task_type else "auto",
                },
                "results": {name: result.to_dict() for name, result in results.items()},
                "summary": {
                    "num_samples": len(samples) if samples else None,
                    "metrics": {name: result.score for name, result in results.items()},
                },
            }

            if samples and self.config.save_detailed_results:
                output_data["samples"] = [s.to_dict() for s in samples]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            import csv
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Score", "Details"])
                for name, result in results.items():
                    writer.writerow([name, result.score, json.dumps(result.details)])

        logger.info(f"Results saved to {output_path}")

    def get_summary(self, results: Dict[str, MetricResult]) -> str:
        """Get human-readable summary of results"""
        lines = ["=" * 50, "EVALUATION SUMMARY", "=" * 50]

        for name, result in results.items():
            score_pct = result.score * 100 if result.score <= 1 else result.score
            lines.append(f"{name}: {score_pct:.2f}%")

            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"  - {key}: {value}")

        lines.append("=" * 50)
        return "\n".join(lines)


def quick_eval(
    data: Union[str, Path, List[Dict]],
    api_key: str,
    metrics: Optional[List[str]] = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    generate_gt: bool = False,
) -> Dict[str, MetricResult]:
    """
    Quick evaluation function for simple use cases.

    Args:
        data: Data file path or list of dicts with input/output
        api_key: OpenAI API key
        metrics: List of metric names to run
        base_url: API base URL (for OpenAI-compatible APIs)
        model: Model name for GT generation
        generate_gt: Whether to generate ground truth

    Returns:
        Evaluation results

    Example:
        >>> results = quick_eval(
        ...     "data.json",
        ...     api_key="sk-...",
        ...     metrics=["exact_match", "rouge"],
        ... )
    """
    config = EvalConfig(
        openai=OpenAIConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
        ),
        enabled_metrics=[MetricType(m) for m in metrics] if metrics else None,
    )

    evaluator = LLMEvaluator(config)
    samples = evaluator.load_data(data)

    if generate_gt:
        samples = evaluator.generate_ground_truth(samples)

    results = evaluator.evaluate(samples)
    print(evaluator.get_summary(results))

    return results
