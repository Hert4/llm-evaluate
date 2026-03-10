"""
Ví Dụ: Đánh Giá BẤT KỲ Dataset Nào
===================================
Script này hướng dẫn cách đánh giá dữ liệu với bất kỳ format nào.

Chạy: python evaluate_any_data.py <path_to_data.json>
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType, TaskType
from llm_eval_framework.data_parsers import AutoParser, EvalSample


def evaluate_any_dataset(
    data_path: str,
    api_key: Optional[str] = None,
    task_type: Optional[str] = None,
    metrics: Optional[list] = None,
    field_mapping: Optional[dict] = None,
    generate_gt: bool = False,
    max_samples: Optional[int] = None,
):
    """
    Đánh giá bất kỳ dataset nào.

    Args:
        data_path: Đường dẫn file (JSON, CSV, JSONL)
        api_key: OpenAI API key
        task_type: Loại task (qa, summarization, rag, chat, coding, tool_calling, reasoning)
        metrics: Danh sách metrics (auto nếu không chỉ định)
        field_mapping: Map field names (vd: {"question": "input", "answer": "output"})
        generate_gt: Có generate ground truth không
        max_samples: Giới hạn số samples
    """
    print("=" * 70)
    print(" LLM EVALUATION FRAMEWORK")
    print("=" * 70)

    # 1. Config
    config = EvalConfig(
        openai=OpenAIConfig(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        ),
        task_type=TaskType(task_type) if task_type else None,
        verbose=True,
    )

    evaluator = LLMEvaluator(config)

    # 2. Load data
    print(f"\n📂 Loading: {data_path}")

    if field_mapping:
        samples = evaluator.load_data(data_path, field_mapping=field_mapping)
    else:
        samples = evaluator.load_data(data_path)

    if max_samples:
        samples = samples[:max_samples]

    print(f"   Loaded {len(samples)} samples")

    # 3. Hiển thị cấu trúc data
    print("\n📋 Data structure:")
    if samples:
        sample = samples[0]
        print(f"   - id: {sample.id}")
        print(f"   - input: {sample.input[:100]}..." if sample.input else "   - input: (empty)")
        print(f"   - output: {sample.output[:100]}..." if sample.output else "   - output: (empty)")
        print(f"   - reference: {sample.reference[:100]}..." if sample.reference else "   - reference: (none)")
        print(f"   - context: {sample.context[:100]}..." if sample.context else "   - context: (none)")
        print(f"   - tool_calls: {sample.tool_calls}" if sample.tool_calls else "")
        print(f"   - choices: {sample.choices}" if sample.choices else "")
        print(f"   - metadata keys: {list(sample.metadata.keys())[:5]}")

    # 4. Auto-detect task type
    detected_task = evaluator._detect_task_type(samples)
    print(f"\n🔍 Detected task type: {detected_task.value}")

    # 5. Generate GT nếu cần
    if generate_gt and config.openai.api_key:
        samples_without_ref = [s for s in samples if not s.reference]
        if samples_without_ref:
            print(f"\n🤖 Generating ground truth for {len(samples_without_ref)} samples...")
            samples = evaluator.generate_ground_truth(samples)
            print("   Done!")

    # 6. Chọn metrics
    if metrics:
        selected_metrics = [MetricType(m) for m in metrics]
    else:
        # Auto-select based on task type
        from llm_eval_framework.config import TASK_METRICS_MAP
        selected_metrics = TASK_METRICS_MAP.get(detected_task, [MetricType.TOKEN_F1])

        # Filter out LLM-based metrics if no API key
        if not config.openai.api_key:
            llm_metrics = {
                MetricType.G_EVAL, MetricType.FAITHFULNESS,
                MetricType.CONTEXT_PRECISION, MetricType.CONTEXT_RECALL,
                MetricType.ANSWER_RELEVANCY, MetricType.FACTSCORE,
                MetricType.WIN_RATE, MetricType.PAIRWISE_COMPARISON,
            }
            selected_metrics = [m for m in selected_metrics if m not in llm_metrics]

    print(f"\n📊 Selected metrics: {[m.value for m in selected_metrics]}")

    # 7. Evaluate
    print("\n🔄 Evaluating...")
    results = evaluator.evaluate(samples, metrics=selected_metrics)

    # 8. Print results
    print("\n" + evaluator.get_summary(results))

    # 9. Save results
    output_file = Path(data_path).stem + "_results.json"
    evaluator.save_results(results, output_file, samples=samples[:100])
    print(f"\n💾 Results saved to: {output_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Đánh giá bất kỳ dataset nào",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Đánh giá cơ bản
  python evaluate_any_data.py data.json

  # Với metrics cụ thể
  python evaluate_any_data.py data.json --metrics rouge exact_match

  # Với field mapping
  python evaluate_any_data.py data.json --field-mapping '{"question":"input","answer":"output"}'

  # Generate ground truth
  python evaluate_any_data.py data.json --generate-gt --api-key sk-...

Task types: qa, summarization, translation, rag, chat, coding, tool_calling, reasoning

Metrics: bleu, comet, rouge, g_eval, exact_match, token_f1, ndcg, recall_at_k,
         ast_accuracy, task_success_rate, pass_at_k, accuracy, faithfulness,
         context_precision, context_recall, factscore, ifeval, win_rate
        """,
    )

    parser.add_argument("data_path", help="Đường dẫn file data (JSON, CSV, JSONL)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--task", choices=[
        "qa", "summarization", "translation", "rag",
        "chat", "coding", "tool_calling", "reasoning"
    ], help="Task type (auto-detect nếu không chỉ định)")
    parser.add_argument("--metrics", nargs="+", help="Metrics to compute")
    parser.add_argument("--field-mapping", type=json.loads,
                        help='JSON mapping field names, vd: {"question":"input"}')
    parser.add_argument("--generate-gt", action="store_true", help="Generate ground truth")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")

    args = parser.parse_args()

    evaluate_any_dataset(
        args.data_path,
        api_key=args.api_key,
        task_type=args.task,
        metrics=args.metrics,
        field_mapping=args.field_mapping,
        generate_gt=args.generate_gt,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    # Demo với CRM data
    crm_path = Path(__file__).parent.parent / "data" / "examples" / "crm.json"

    if crm_path.exists() and len(sys.argv) == 1:
        print("Demo với CRM data...")
        print(f"File: {crm_path}")
        print("\nChạy: python evaluate_any_data.py <path> [options]")
        print("      python evaluate_any_data.py --help")

        # Quick demo
        evaluate_any_dataset(
            str(crm_path),
            max_samples=10,
            metrics=["ifeval"],
        )
    else:
        main()
