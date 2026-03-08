#!/usr/bin/env python3
"""
LLM Evaluation Framework CLI
============================
Command-line interface for evaluating LLM outputs.

Usage:
    python -m llm_eval_framework.cli evaluate data.json --metrics rouge bleu
    python -m llm_eval_framework.cli evaluate logs.json --generate-gt --api-key sk-...
"""
import argparse
import os
import sys
import json
from pathlib import Path

from .evaluator import LLMEvaluator
from .config import EvalConfig, OpenAIConfig, MetricType, TaskType


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate LLM outputs")
    eval_parser.add_argument("input", help="Input data file (JSON, CSV, JSONL)")
    eval_parser.add_argument(
        "--metrics", "-m",
        nargs="+",
        help="Metrics to compute (e.g., rouge bleu exact_match)",
    )
    eval_parser.add_argument(
        "--task", "-t",
        choices=["qa", "summarization", "translation", "rag", "chat", "coding", "tool_calling", "reasoning"],
        help="Task type (auto-detected if not specified)",
    )
    eval_parser.add_argument(
        "--output", "-o",
        help="Output file for results",
    )
    eval_parser.add_argument(
        "--format", "-f",
        choices=["json", "csv"],
        default="json",
        help="Output format",
    )
    eval_parser.add_argument(
        "--generate-gt",
        action="store_true",
        help="Generate ground truth using LLM",
    )
    eval_parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    eval_parser.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="API base URL (for OpenAI-compatible APIs)",
    )
    eval_parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model for GT generation and LLM-based metrics",
    )
    eval_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    # List metrics command
    list_parser = subparsers.add_parser("list-metrics", help="List available metrics")

    return parser.parse_args()


def list_metrics():
    """List all available metrics"""
    print("\nAvailable Metrics:")
    print("=" * 60)

    metric_groups = {
        "Translation": ["bleu", "comet"],
        "Summarization": ["rouge", "g_eval"],
        "QA": ["exact_match", "token_f1"],
        "Ranking": ["ndcg", "recall_at_k", "precision_at_k", "mrr"],
        "Tool Calling": ["ast_accuracy", "task_success_rate"],
        "Coding": ["pass_at_k"],
        "Reasoning": ["accuracy"],
        "RAG": ["faithfulness", "context_precision", "context_recall", "answer_relevancy"],
        "Safety": ["factscore", "ifeval"],
        "Chat": ["win_rate", "pairwise_comparison"],
    }

    for group, metrics in metric_groups.items():
        print(f"\n{group}:")
        for metric in metrics:
            print(f"  - {metric}")


def evaluate(args):
    """Run evaluation"""
    # Build config
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    config = EvalConfig(
        openai=OpenAIConfig(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
        ),
        task_type=TaskType(args.task) if args.task else None,
        enabled_metrics=[MetricType(m) for m in args.metrics] if args.metrics else None,
        verbose=args.verbose,
    )

    # Create evaluator
    evaluator = LLMEvaluator(config)

    # Load data
    print(f"Loading data from: {args.input}")
    samples = evaluator.load_data(args.input)
    print(f"Loaded {len(samples)} samples")

    # Generate GT if requested
    if args.generate_gt:
        if not api_key:
            print("Error: --api-key or OPENAI_API_KEY required for GT generation")
            sys.exit(1)
        print("Generating ground truth...")
        samples = evaluator.generate_ground_truth(samples)

    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate(samples)

    # Print summary
    print(evaluator.get_summary(results))

    # Save results
    if args.output:
        evaluator.save_results(
            results,
            args.output,
            samples=samples,
            format=args.format,
        )
        print(f"Results saved to: {args.output}")


def main():
    args = parse_args()

    if args.command == "evaluate":
        evaluate(args)
    elif args.command == "list-metrics":
        list_metrics()
    else:
        print("Use --help for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
