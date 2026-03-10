"""
Example: Evaluate API Logs (like all_logs.json)
===============================================
This example shows how to evaluate LLM API call logs.
"""
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType
from llm_eval_framework.data_parsers import LogParser


def evaluate_api_logs(
    log_file: str,
    api_key: str = None,
    generate_gt: bool = True,
    filter_model: str = None,
):
    """
    Evaluate API call logs.

    Args:
        log_file: Path to log file (e.g., all_logs.json)
        api_key: OpenAI API key for GT generation
        generate_gt: Whether to generate ground truth
        filter_model: Only evaluate logs from specific model
    """
    print(f"Loading logs from: {log_file}")

    # Create config
    config = EvalConfig(
        openai=OpenAIConfig(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        ),
        verbose=True,
    )

    # Create evaluator
    evaluator = LLMEvaluator(config)

    # Load logs with optional filtering
    parser = LogParser(filter_by_model=filter_model)
    samples = parser.parse(log_file)

    print(f"Loaded {len(samples)} samples")

    # Show sample info
    if samples:
        print("\nSample info:")
        for i, sample in enumerate(samples[:3]):
            print(f"  [{i}] Input: {sample.input[:80]}...")
            print(f"      Output: {sample.output[:80]}..." if sample.output else "      Output: (empty)")
            print(f"      Model: {sample.metadata.get('model', 'N/A')}")
            print()

    # Generate ground truth if needed and API key available
    if generate_gt and config.openai.api_key:
        print("Generating ground truth...")
        # Detect task type from first sample
        task_type = "qa"  # Default
        if samples and samples[0].context:
            task_type = "rag"

        samples = evaluator.generate_ground_truth(samples, task_type=task_type)

        # Show generated GT
        print("\nGenerated ground truth samples:")
        for i, sample in enumerate(samples[:3]):
            if sample.reference:
                print(f"  [{i}] Reference: {sample.reference[:100]}...")

    # Run evaluation
    print("\nRunning evaluation...")

    # Select appropriate metrics
    metrics = [MetricType.TOKEN_F1]

    # Add context-based metrics if context available
    if any(s.context for s in samples) and config.openai.api_key:
        metrics.extend([MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY])

    # Add tool calling metrics if tool calls present
    if any(s.tool_calls for s in samples):
        metrics.append(MetricType.AST_ACCURACY)

    results = evaluator.evaluate(samples, metrics=metrics)

    # Print summary
    print(evaluator.get_summary(results))

    # Save results
    output_file = Path(log_file).stem + "_eval_results.json"
    evaluator.save_results(results, output_file, samples=samples)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate API logs")
    parser.add_argument("log_file", help="Path to log file (JSON)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--no-gt", action="store_true", help="Skip GT generation")
    parser.add_argument("--model", help="Filter by model name")
    args = parser.parse_args()

    evaluate_api_logs(
        args.log_file,
        api_key=args.api_key,
        generate_gt=not args.no_gt,
        filter_model=args.model,
    )


if __name__ == "__main__":
    # Example usage with all_logs.json
    log_path = Path(__file__).parent.parent.parent / "data" / "all_logs.json"

    if log_path.exists():
        print(f"Found log file: {log_path}")
        print("Run with: python evaluate_logs.py <path_to_logs>")
        print("Or set OPENAI_API_KEY and run this script directly")

        # Uncomment to run:
        # evaluate_api_logs(str(log_path))
    else:
        print("Usage: python evaluate_logs.py <path_to_logs.json> [--api-key KEY]")
