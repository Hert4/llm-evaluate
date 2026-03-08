"""
Basic Usage Examples for LLM Evaluation Framework
================================================
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType, TaskType
from llm_eval_framework.data_parsers import EvalSample


def example_1_basic_evaluation():
    """Basic evaluation with pre-defined samples"""
    print("\n" + "="*60)
    print("Example 1: Basic Evaluation")
    print("="*60)

    # Create samples directly
    samples = [
        EvalSample(
            id="1",
            input="What is the capital of France?",
            output="The capital of France is Paris.",
            reference="Paris",
        ),
        EvalSample(
            id="2",
            input="What is 2 + 2?",
            output="2 + 2 equals 4.",
            reference="4",
        ),
        EvalSample(
            id="3",
            input="Who wrote Romeo and Juliet?",
            output="Romeo and Juliet was written by Shakespeare.",
            reference="William Shakespeare",
        ),
    ]

    # Create evaluator (no API needed for basic metrics)
    evaluator = LLMEvaluator()

    # Run QA metrics
    results = evaluator.evaluate(
        samples,
        metrics=[MetricType.EXACT_MATCH, MetricType.TOKEN_F1],
    )

    # Print results
    print(evaluator.get_summary(results))


def example_2_evaluate_from_json():
    """Evaluate from JSON file"""
    print("\n" + "="*60)
    print("Example 2: Evaluate from JSON File")
    print("="*60)

    # Sample data in JSON format
    sample_data = {
        "data": [
            {
                "id": 1,
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "expected": "Python is a high-level programming language.",
            },
            {
                "id": 2,
                "question": "What is AI?",
                "answer": "AI stands for Artificial Intelligence.",
                "expected": "Artificial Intelligence",
            },
        ]
    }

    # Create evaluator with field mapping
    evaluator = LLMEvaluator()
    samples = evaluator.load_data(
        sample_data,
        field_mapping={
            "question": "input",
            "answer": "output",
            "expected": "reference",
        }
    )

    results = evaluator.evaluate(samples)
    print(evaluator.get_summary(results))


def example_3_evaluate_api_logs():
    """Evaluate API call logs (like all_logs.json)"""
    print("\n" + "="*60)
    print("Example 3: Evaluate API Logs")
    print("="*60)

    # Sample API log format
    log_data = {
        "data": [
            {
                "id": 194597,
                "model": "gpt-4",
                "requestPayload": '{"messages":[{"role":"user","content":"What is 2+2?"}]}',
                "responsePayload": '{"choices":[{"message":{"content":"2+2 equals 4."}}]}',
            }
        ]
    }

    evaluator = LLMEvaluator()
    samples = evaluator.load_data(log_data)

    print(f"Loaded {len(samples)} samples from logs")
    for sample in samples:
        print(f"  Input: {sample.input[:50]}...")
        print(f"  Output: {sample.output[:50]}...")


def example_4_with_ground_truth_generation():
    """Generate ground truth using OpenAI API"""
    print("\n" + "="*60)
    print("Example 4: Generate Ground Truth")
    print("="*60)

    # Requires API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Create config with OpenAI settings
    config = EvalConfig(
        openai=OpenAIConfig(
            api_key=api_key,
            model="gpt-4o",  # Or any OpenAI-compatible model
            base_url="https://api.openai.com/v1",  # Can change for other APIs
        )
    )

    evaluator = LLMEvaluator(config)

    # Samples without reference
    samples = [
        EvalSample(
            id="1",
            input="What is the capital of Vietnam?",
            output="The capital of Vietnam is Hanoi.",
        ),
    ]

    # Generate ground truth
    samples_with_gt = evaluator.generate_ground_truth(samples, task_type="qa")

    print(f"Generated reference: {samples_with_gt[0].reference}")

    # Now evaluate
    results = evaluator.evaluate(samples_with_gt)
    print(evaluator.get_summary(results))


def example_5_summarization_evaluation():
    """Evaluate summarization with ROUGE"""
    print("\n" + "="*60)
    print("Example 5: Summarization Evaluation (ROUGE)")
    print("="*60)

    samples = [
        EvalSample(
            id="1",
            input="""Apple Inc. announced its quarterly earnings today,
            reporting revenue of $89.5 billion, a 5% increase from last year.
            CEO Tim Cook highlighted strong iPhone sales and growth in services.
            The company also announced a new stock buyback program worth $90 billion.""",
            output="Apple reported $89.5B revenue with strong iPhone sales and a $90B buyback.",
            reference="Apple's Q3 revenue was $89.5 billion, up 5%. Strong iPhone sales and $90 billion buyback announced.",
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(
        samples,
        metrics=[MetricType.ROUGE],
        task_type=TaskType.SUMMARIZATION,
    )
    print(evaluator.get_summary(results))


def example_6_rag_evaluation():
    """Evaluate RAG with context"""
    print("\n" + "="*60)
    print("Example 6: RAG Evaluation (requires API key)")
    print("="*60)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Skipping - OPENAI_API_KEY not set")
        return

    config = EvalConfig(
        openai=OpenAIConfig(api_key=api_key, model="gpt-4o")
    )

    samples = [
        EvalSample(
            id="1",
            input="What was Apple's revenue?",
            output="Apple's revenue was $89.5 billion, and Tim Cook said it was a record.",
            context="Apple Inc. reported quarterly revenue of $89.5 billion, a 5% increase.",
            reference="$89.5 billion",
        ),
    ]

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(
        samples,
        metrics=[MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY],
        task_type=TaskType.RAG,
    )
    print(evaluator.get_summary(results))


def example_7_tool_calling_evaluation():
    """Evaluate tool/function calling"""
    print("\n" + "="*60)
    print("Example 7: Tool Calling Evaluation")
    print("="*60)

    samples = [
        EvalSample(
            id="1",
            input="What's the weather in Hanoi?",
            output="",
            tool_calls=[{"name": "get_weather", "arguments": {"city": "Hanoi"}}],
            expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "Hanoi"}}],
        ),
        EvalSample(
            id="2",
            input="Book a flight to Paris",
            output="",
            tool_calls=[{"name": "search_flights", "arguments": {"destination": "Paris"}}],
            expected_tool_calls=[{"name": "book_flight", "arguments": {"destination": "Paris"}}],
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(
        samples,
        metrics=[MetricType.AST_ACCURACY],
        task_type=TaskType.TOOL_CALLING,
    )
    print(evaluator.get_summary(results))


def example_8_multiple_choice():
    """Evaluate multiple choice questions"""
    print("\n" + "="*60)
    print("Example 8: Multiple Choice Evaluation")
    print("="*60)

    samples = [
        EvalSample(
            id="1",
            input="What is the largest planet in our solar system?",
            output="The answer is B) Jupiter",
            choices=["Mars", "Jupiter", "Saturn", "Neptune"],
            correct_answer="B",
        ),
        EvalSample(
            id="2",
            input="Which element has the symbol 'O'?",
            output="A",
            choices=["Oxygen", "Gold", "Silver", "Iron"],
            correct_answer="A",
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(
        samples,
        metrics=[MetricType.ACCURACY],
        task_type=TaskType.REASONING,
    )
    print(evaluator.get_summary(results))


def example_9_code_evaluation():
    """Evaluate code generation with Pass@K"""
    print("\n" + "="*60)
    print("Example 9: Code Evaluation (Pass@K)")
    print("="*60)

    samples = [
        EvalSample(
            id="1",
            input="Write a function to check if a number is prime",
            output="""
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
            test_cases=[
                {"test": "assert is_prime(7) == True"},
                {"test": "assert is_prime(4) == False"},
                {"test": "assert is_prime(1) == False"},
            ],
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(
        samples,
        metrics=[MetricType.PASS_AT_K],
        task_type=TaskType.CODING,
    )
    print(evaluator.get_summary(results))


def example_10_instruction_following():
    """Evaluate instruction following (IFEval)"""
    print("\n" + "="*60)
    print("Example 10: Instruction Following (IFEval)")
    print("="*60)

    samples = [
        EvalSample(
            id="1",
            input="Write about spring. Use exactly 3 sentences. Do not use the word 'flower'.",
            output="Spring is a beautiful season. The weather becomes warmer. Birds return from migration.",
        ),
        EvalSample(
            id="2",
            input="Write a haiku about the ocean. Use exactly 3 sentences.",
            output="The ocean waves crash. Seagulls fly above the shore. Peace fills my heart now.",
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(
        samples,
        metrics=[MetricType.IFEVAL],
    )
    print(evaluator.get_summary(results))


if __name__ == "__main__":
    # Run examples that don't require API keys
    example_1_basic_evaluation()
    example_2_evaluate_from_json()
    example_3_evaluate_api_logs()
    example_5_summarization_evaluation()
    example_7_tool_calling_evaluation()
    example_8_multiple_choice()
    example_9_code_evaluation()
    example_10_instruction_following()

    # These require OPENAI_API_KEY environment variable
    example_4_with_ground_truth_generation()
    example_6_rag_evaluation()
