#!/usr/bin/env python3
"""
Run evaluation on benchmark datasets using the LLM Evaluation Framework.

Reads task YAML configs (gt_metrics) to know which metrics to run per task.
For LLM-based metrics (faithfulness), uses the same OpenAI-compatible API.

Results are saved to data/eval_results/{model_name}/{task_name}.json

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --task crm_recommendation
    python scripts/run_evaluation.py --model misa-ai-1.1 --task crmmisa_dashboard
    python scripts/run_evaluation.py --list-tasks
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TASKS_DIR = SCRIPT_DIR / "tasks"
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"

# Import the llm-evaluate package (hyphenated dir name needs importlib)
import importlib
sys.path.insert(0, str(PROJECT_ROOT.parent))
_pkg = importlib.import_module("llm-evaluate")

from importlib import import_module
_config = import_module("llm-evaluate.config")
_evaluator = import_module("llm-evaluate.evaluator")
_data_parsers_base = import_module("llm-evaluate.data_parsers.base")

EvalConfig = _config.EvalConfig
OpenAIConfig = _config.OpenAIConfig
MetricType = _config.MetricType
MetricConfig = _config.MetricConfig
TaskType = _config.TaskType
LLMEvaluator = _evaluator.LLMEvaluator
EvalSample = _data_parsers_base.EvalSample

# ---------------------------------------------------------------------------
# API Configuration (same endpoint as generate_ground_truth.py)
# ---------------------------------------------------------------------------
API_URL = "https://numerous-catch-uploaded-compile.trycloudflare.com/v1"
API_KEY = "misa_misa_00t07fh7_ZFRMf6rOUaVHTv6CZH0uOzAx_LDP1IeWM"
DEFAULT_JUDGE_MODEL = "gpt-5.2"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from httpx/openai so tqdm bars stay clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Metric name → MetricType mapping
# ---------------------------------------------------------------------------
METRIC_NAME_MAP = {
    "exact_match": "EXACT_MATCH",
    "token_f1": "TOKEN_F1",
    "bleu": "BLEU",
    "faithfulness": "FAITHFULNESS",
    "rouge": "ROUGE",
    "accuracy": "ACCURACY",
    "comet": "COMET",
    "g_eval": "G_EVAL",
    "context_precision": "CONTEXT_PRECISION",
    "context_recall": "CONTEXT_RECALL",
    "answer_relevancy": "ANSWER_RELEVANCY",
    "list_match": "LIST_MATCH",
}

# LLM-based metrics that need the API
LLM_BASED_METRICS = {"faithfulness", "g_eval", "context_precision", "context_recall", "answer_relevancy"}


# ---------------------------------------------------------------------------
# Task config (reuse from generate_ground_truth.py)
# ---------------------------------------------------------------------------
@dataclass
class EvalTaskConfig:
    """Evaluation task config from YAML."""
    task_name: str = ""
    task_type: str = "qa"
    description: str = ""
    gt_task_type: str = "qa"
    gt_metrics: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, filepath: Path) -> "EvalTaskConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            raise ValueError(f"Empty YAML: {filepath}")
        cfg = cls()
        cfg.task_name = raw.get("task_name", filepath.stem)
        cfg.task_type = raw.get("task_type", "qa")
        cfg.description = raw.get("description", "")
        cfg.gt_task_type = raw.get("gt_task_type", raw.get("task_type", "qa"))
        cfg.gt_metrics = raw.get("gt_metrics", [])
        return cfg

    @property
    def has_eval_config(self) -> bool:
        return bool(self.gt_metrics)


def discover_eval_tasks(tasks_dir: Path) -> List[EvalTaskConfig]:
    """Load all task YAMLs that have evaluation metrics configured."""
    if not tasks_dir.is_dir():
        return []
    configs = []
    for yaml_file in sorted(tasks_dir.glob("*.yaml")) + sorted(tasks_dir.glob("*.yml")):
        if yaml_file.stem.startswith("_") or yaml_file.stem == "language_mapping":
            continue
        try:
            cfg = EvalTaskConfig.from_yaml(yaml_file)
            if cfg.has_eval_config:
                configs.append(cfg)
        except Exception as e:
            logger.warning(f"Failed to load {yaml_file.name}: {e}")
    return configs


def find_benchmark_file(task_name: str, benchmarks_dir: Path) -> Optional[Path]:
    """Find benchmark JSON file, excluding backups."""
    matches = [
        p for p in benchmarks_dir.glob(f"{task_name}_*.json")
        if "_backup" not in p.stem
    ]
    if not matches:
        return None
    return sorted(matches)[-1]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
def run_single_task(
    task_config: EvalTaskConfig,
    benchmarks_dir: Path,
    results_dir: Path,
    model_name: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Optional[Dict[str, Any]]:
    """
    Run evaluation for a single task.

    Returns dict with results or None on failure.
    """
    # Find benchmark file
    benchmark_path = find_benchmark_file(task_config.task_name, benchmarks_dir)
    if not benchmark_path:
        logger.error(f"[{task_config.task_name}] No benchmark file found")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"Task: {task_config.task_name}")
    logger.info(f"File: {benchmark_path.name}")
    logger.info(f"Metrics: {task_config.gt_metrics}")
    logger.info(f"Model under evaluation: {model_name}")
    logger.info(f"{'='*60}")

    # Load benchmark data
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    samples_data = benchmark_data.get("data", [])

    # Check references exist
    ref_count = sum(1 for s in samples_data if s.get("reference"))
    if ref_count == 0:
        logger.error(f"[{task_config.task_name}] No samples have reference! Run generate_ground_truth.py first.")
        return None

    logger.info(f"Loaded {len(samples_data)} samples ({ref_count} with reference)")

    # Filter samples to only those from the target model
    model_samples = [
        s for s in samples_data
        if s.get("metadata", {}).get("model") == model_name and s.get("reference")
    ]

    # If no samples match model filter, use all samples with reference
    # (benchmark output IS from the production model)
    if not model_samples:
        model_samples = [s for s in samples_data if s.get("reference")]
        logger.info(f"Using all {len(model_samples)} samples with reference (output is from production model)")
    else:
        logger.info(f"Found {len(model_samples)} samples from model '{model_name}'")

    # Convert to EvalSample objects
    eval_samples = []
    for s in model_samples:
        sample = EvalSample.from_dict(s)
        eval_samples.append(sample)

    # Determine which metrics need LLM
    has_llm_metrics = any(m in LLM_BASED_METRICS for m in task_config.gt_metrics)

    # Build config
    openai_config = OpenAIConfig(
        api_key=API_KEY,
        base_url=API_URL,
        model=judge_model,
        temperature=0.0,
        max_tokens=4096,
        timeout=120,
        max_retries=5,
    )

    metric_types = []
    for metric_name in task_config.gt_metrics:
        enum_name = METRIC_NAME_MAP.get(metric_name)
        if enum_name:
            try:
                metric_types.append(MetricType(metric_name))
            except ValueError:
                logger.warning(f"Unknown metric: {metric_name}, skipping")
        else:
            logger.warning(f"Unknown metric: {metric_name}, skipping")

    if not metric_types:
        logger.error(f"[{task_config.task_name}] No valid metrics to run")
        return None

    # Build evaluator config
    eval_config = EvalConfig(
        openai=openai_config if has_llm_metrics else OpenAIConfig(),
        enabled_metrics=metric_types,
    )

    evaluator = LLMEvaluator(eval_config)

    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate(eval_samples, metrics=metric_types)
    elapsed = time.time() - start_time

    # Print results
    summary = evaluator.get_summary(results)
    print(summary)

    # Build output
    result_output = {
        "task_name": task_config.task_name,
        "task_type": task_config.gt_task_type,
        "description": task_config.description,
        "model_evaluated": model_name,
        "judge_model": judge_model,
        "benchmark_file": benchmark_path.name,
        "num_samples": len(eval_samples),
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "metrics": {},
    }

    for name, result in results.items():
        result_output["metrics"][name] = {
            "score": round(result.score, 6),
            "score_pct": round(result.score * 100 if result.score <= 1 else result.score, 2),
            "details": result.details,
        }

    # Save results
    model_results_dir = results_dir / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_results_dir / f"{task_config.task_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_output, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_config.task_name}] Results saved → {output_path}")
    logger.info(f"[{task_config.task_name}] Done in {elapsed:.1f}s")

    return result_output


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_summary_table(all_results: List[Dict[str, Any]], model_name: str):
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print(f"  EVALUATION REPORT — Model: {model_name}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    # Header
    print(f"\n  {'Task':<25} {'Metric':<18} {'Score':>8} {'Samples':>8}")
    print(f"  {'─'*25} {'─'*18} {'─'*8} {'─'*8}")

    for result in all_results:
        task = result["task_name"]
        n_samples = result["num_samples"]
        first = True
        for metric_name, metric_data in result["metrics"].items():
            score_pct = metric_data["score_pct"]
            if first:
                print(f"  {task:<25} {metric_name:<18} {score_pct:>7.2f}% {n_samples:>8}")
                first = False
            else:
                print(f"  {'':<25} {metric_name:<18} {score_pct:>7.2f}%")
        print()

    print(f"{'='*80}")

    # Save summary
    return {
        "model": model_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": {
            r["task_name"]: {
                m: d["score_pct"] for m, d in r["metrics"].items()
            }
            for r in all_results
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                      # Evaluate all tasks
  %(prog)s --task crm_recommendation            # Single task
  %(prog)s --model misa-ai-1.1                  # Specify model name
  %(prog)s --list-tasks                         # Show available tasks
  %(prog)s --judge-model gpt-5.2                # LLM judge model
        """,
    )
    parser.add_argument(
        "--task", action="append", dest="tasks",
        help="Task(s) to evaluate (can repeat). Default: all.",
    )
    parser.add_argument(
        "--model", default="misa-ai-1.1",
        help="Model name being evaluated (default: misa-ai-1.1).",
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help=f"LLM judge model for faithfulness etc. (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--benchmarks-dir", default=None,
        help="Benchmarks directory.",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Results output directory.",
    )
    parser.add_argument(
        "--tasks-dir", default=None,
        help="YAML tasks directory.",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List tasks with eval config and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else TASKS_DIR
    benchmarks_dir = Path(args.benchmarks_dir) if args.benchmarks_dir else BENCHMARKS_DIR
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    # Discover tasks
    eval_tasks = discover_eval_tasks(tasks_dir)
    if not eval_tasks:
        logger.error(f"No tasks with gt_metrics found in {tasks_dir}")
        sys.exit(1)

    # List tasks
    if args.list_tasks:
        print(f"\nTasks with evaluation config (from {tasks_dir}):")
        print("-" * 80)
        for cfg in eval_tasks:
            benchmark = find_benchmark_file(cfg.task_name, benchmarks_dir)
            bname = benchmark.name if benchmark else "(not found)"
            print(f"  {cfg.task_name:<30} metrics={cfg.gt_metrics}")
            print(f"    Benchmark: {bname}")
            if cfg.description:
                print(f"    {cfg.description}")
        print()
        return

    # Filter tasks
    if args.tasks:
        filtered = [cfg for cfg in eval_tasks if cfg.task_name in args.tasks]
        if not filtered:
            available = ", ".join(cfg.task_name for cfg in eval_tasks)
            logger.error(f"No matching tasks. Available: {available}")
            sys.exit(1)
        eval_tasks = filtered

    # Run evaluations
    model_name = args.model
    logger.info(f"Evaluating model: {model_name}")
    logger.info(f"Tasks: {len(eval_tasks)}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Results dir: {results_dir / model_name}")

    all_results = []
    for task_config in tqdm(eval_tasks, desc="Tasks", unit="task"):
        try:
            result = run_single_task(
                task_config=task_config,
                benchmarks_dir=benchmarks_dir,
                results_dir=results_dir,
                model_name=model_name,
                judge_model=args.judge_model,
            )
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"[{task_config.task_name}] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary — merge with existing summary so partial runs don't overwrite
    if all_results:
        summary = print_summary_table(all_results, model_name)

        # Save summary (merge with existing)
        summary_dir = results_dir / model_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "_summary.json"

        # Load existing summary if present
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing.get("tasks", {}).update(summary.get("tasks", {}))
                summary["tasks"] = existing["tasks"]
            except Exception:
                pass  # Overwrite on error

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved → {summary_path}")
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()
