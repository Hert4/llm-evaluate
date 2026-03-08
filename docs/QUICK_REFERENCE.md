# Quick Reference - Cheat Sheet

## Cài Đặt

```bash
pip install openai numpy
# Optional: pip install unbabel-comet  # Cho COMET metric
```

## Import

```python
from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType, MetricConfig
from llm_eval_framework.data_parsers import EvalSample
```

## Tạo Sample

```python
sample = EvalSample(
    id="1",
    input="Câu hỏi/Input",
    output="Output của model",
    reference="Ground truth/Đáp án chuẩn",     # Optional
    context="Context cho RAG",                  # Optional
    tool_calls=[{"name": "func", "arguments": {}}],  # Optional
    expected_tool_calls=[...],                  # Optional
    choices=["A", "B", "C", "D"],               # Optional (multiple choice)
    correct_answer="A",                         # Optional
    relevance_scores=[1.0, 0.5, 0.0],          # Optional (ranking)
    test_cases=[{"test": "assert ..."}],       # Optional (coding)
)
```

## Quick Evaluation

```python
# Không cần API key
evaluator = LLMEvaluator()
samples = evaluator.load_data("data.json")
results = evaluator.evaluate(samples, metrics=[MetricType.ROUGE, MetricType.EXACT_MATCH])
print(evaluator.get_summary(results))

# Với API key (cho LLM-based metrics)
config = EvalConfig(
    openai=OpenAIConfig(api_key="sk-...", model="gpt-4o")
)
evaluator = LLMEvaluator(config)
```

## Danh Sách Metrics

| Metric | API Key | Use Case |
|--------|---------|----------|
| `BLEU` | ❌ | Dịch thuật |
| `COMET` | ❌ (cần GPU) | Dịch thuật (chính xác hơn) |
| `ROUGE` | ❌ | Tóm tắt |
| `G_EVAL` | ✅ | Tóm tắt (LLM judge) |
| `EXACT_MATCH` | ❌ | QA ngắn |
| `TOKEN_F1` | ❌ | QA |
| `NDCG` | ❌ | Ranking |
| `RECALL_AT_K` | ❌ | Retrieval |
| `PRECISION_AT_K` | ❌ | Retrieval |
| `MRR` | ❌ | Ranking |
| `AST_ACCURACY` | ❌ | Tool calling |
| `TASK_SUCCESS_RATE` | ❌ | Agent |
| `PASS_AT_K` | ❌ | Coding |
| `ACCURACY` | ❌ | Multiple choice |
| `FAITHFULNESS` | ✅ | RAG (hallucination) |
| `CONTEXT_PRECISION` | ✅ | RAG |
| `CONTEXT_RECALL` | ✅ | RAG |
| `ANSWER_RELEVANCY` | ✅ | RAG |
| `FACTSCORE` | ✅ | Safety |
| `IFEVAL` | ❌ | Instruction following |
| `WIN_RATE` | ✅ | Chat comparison |

## Code Mẫu Theo Task

### 1. QA

```python
samples = [
    EvalSample(id="1", input="Câu hỏi?", output="Trả lời", reference="Đáp án")
]
results = evaluator.evaluate(samples, metrics=[MetricType.EXACT_MATCH, MetricType.TOKEN_F1])
```

### 2. Tóm Tắt

```python
samples = [
    EvalSample(id="1", input="Văn bản dài...", output="Tóm tắt", reference="Tóm tắt chuẩn")
]
results = evaluator.evaluate(samples, metrics=[MetricType.ROUGE])
```

### 3. RAG

```python
samples = [
    EvalSample(
        id="1",
        input="Câu hỏi?",
        context="Tài liệu retrieved...",
        output="Câu trả lời",
        reference="Đáp án chuẩn"
    )
]
results = evaluator.evaluate(samples, metrics=[MetricType.FAITHFULNESS, MetricType.TOKEN_F1])
```

### 4. Tool Calling

```python
samples = [
    EvalSample(
        id="1",
        input="Thời tiết HN?",
        output="",
        tool_calls=[{"name": "get_weather", "arguments": {"city": "HN"}}],
        expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "HN"}}]
    )
]
results = evaluator.evaluate(samples, metrics=[MetricType.AST_ACCURACY])
```

### 5. Multiple Choice

```python
samples = [
    EvalSample(
        id="1",
        input="Câu hỏi?",
        output="B",
        choices=["A", "B", "C", "D"],
        correct_answer="B"
    )
]
results = evaluator.evaluate(samples, metrics=[MetricType.ACCURACY])
```

### 6. Coding

```python
samples = [
    EvalSample(
        id="1",
        input="Viết hàm...",
        output="def func(): ...",
        test_cases=[{"test": "assert func() == expected"}]
    )
]
results = evaluator.evaluate(samples, metrics=[MetricType.PASS_AT_K])
```

### 7. Ranking

```python
samples = [
    EvalSample(
        id="1",
        input="Query",
        output="",
        relevance_scores=[3, 2, 1, 0, 0]  # Điểm relevance của top-K items
    )
]
results = evaluator.evaluate(samples, metrics=[MetricType.NDCG, MetricType.RECALL_AT_K])
```

## Load Data

```python
# JSON file
samples = evaluator.load_data("data.json")

# CSV file
samples = evaluator.load_data("data.csv")

# API logs (auto-detect)
samples = evaluator.load_data("all_logs.json")

# Custom field mapping
samples = evaluator.load_data(
    "data.json",
    field_mapping={
        "question": "input",
        "answer": "output",
        "expected": "reference"
    }
)
```

## Generate Ground Truth

```python
config = EvalConfig(
    openai=OpenAIConfig(api_key="sk-...", model="gpt-4o")
)
evaluator = LLMEvaluator(config)

samples = evaluator.load_data("data.json")
samples = evaluator.generate_ground_truth(samples, task_type="qa")
```

## Save Results

```python
evaluator.save_results(results, "results.json", samples=samples)
```

## CLI

```bash
# Evaluate
python -m llm_eval_framework.cli evaluate data.json --metrics rouge exact_match

# With GT generation
python -m llm_eval_framework.cli evaluate data.json --generate-gt --api-key sk-...

# List metrics
python -m llm_eval_framework.cli list-metrics
```

## OpenAI-Compatible APIs

```python
config = EvalConfig(
    openai=OpenAIConfig(
        api_key="your-key",
        base_url="https://your-api.com/v1",  # Custom endpoint
        model="your-model",
    )
)
```

Hỗ trợ: OpenAI, Azure OpenAI, Ollama, vLLM, MISA AI, và các API tương thích khác.
