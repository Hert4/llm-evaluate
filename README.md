# LLM Evaluation Framework

A comprehensive framework for evaluating LLM outputs using multiple metrics. Supports any data format and uses OpenAI-compatible API for ground truth generation.

## Features

- **Universal Data Parser**: Automatically handles JSON, CSV, TSV, JSONL, and API log formats
- **Ground Truth Generation**: Uses OpenAI-compatible APIs to generate reference answers
- **20+ Evaluation Metrics**: Covering translation, summarization, QA, RAG, coding, and more
- **Flexible Configuration**: YAML/JSON config files or programmatic configuration
- **CLI & Python API**: Use from command line or integrate into your code

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType

# Create evaluator
config = EvalConfig(
    openai=OpenAIConfig(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",  # Or any OpenAI-compatible API
        model="gpt-4o",
    )
)
evaluator = LLMEvaluator(config)

# Load data (auto-detects format)
samples = evaluator.load_data("your_data.json")

# Optional: Generate ground truth
samples = evaluator.generate_ground_truth(samples)

# Run evaluation
results = evaluator.evaluate(samples)
print(evaluator.get_summary(results))
```

### CLI

```bash
# Basic evaluation
python -m llm_eval_framework.cli evaluate data.json --metrics rouge exact_match

# With ground truth generation
python -m llm_eval_framework.cli evaluate logs.json --generate-gt --api-key sk-...

# List available metrics
python -m llm_eval_framework.cli list-metrics
```

## Supported Metrics

### Translation
| Metric | Description |
|--------|-------------|
| `bleu` | N-gram overlap score |
| `comet` | Neural MT evaluation (requires GPU) |

### Summarization
| Metric | Description |
|--------|-------------|
| `rouge` | ROUGE-1, ROUGE-2, ROUGE-L scores |
| `g_eval` | LLM-based evaluation on coherence, fluency, relevance |

### Question Answering
| Metric | Description |
|--------|-------------|
| `exact_match` | Exact string match (normalized) |
| `token_f1` | Token-level F1 score |

### Ranking / Retrieval
| Metric | Description |
|--------|-------------|
| `ndcg` | Normalized Discounted Cumulative Gain |
| `recall_at_k` | Recall at K |
| `precision_at_k` | Precision at K |
| `mrr` | Mean Reciprocal Rank |

### Tool/Function Calling
| Metric | Description |
|--------|-------------|
| `ast_accuracy` | Correct function name and arguments |
| `task_success_rate` | Multi-step task completion rate |

### Coding
| Metric | Description |
|--------|-------------|
| `pass_at_k` | Code passes test cases |

### Reasoning
| Metric | Description |
|--------|-------------|
| `accuracy` | Multiple choice accuracy |

### RAG (Retrieval Augmented Generation)
| Metric | Description |
|--------|-------------|
| `faithfulness` | Answer grounded in context (detects hallucination) |
| `context_precision` | Retrieved context relevance |
| `context_recall` | Context contains needed information |
| `answer_relevancy` | Answer addresses the question |

### Safety
| Metric | Description |
|--------|-------------|
| `factscore` | Factual accuracy of atomic claims |
| `ifeval` | Instruction following (word count, format, etc.) |

### Chat / Chatbot
| Metric | Description |
|--------|-------------|
| `win_rate` | LLM-as-judge comparison vs baseline |
| `pairwise_comparison` | Compare two models directly |

## Data Formats

### Standard Format
```json
{
  "data": [
    {
      "id": "1",
      "input": "What is the capital of France?",
      "output": "The capital is Paris.",
      "reference": "Paris"
    }
  ]
}
```

### API Log Format (auto-detected)
```json
{
  "data": [
    {
      "id": 123,
      "requestPayload": "{\"messages\":[{\"role\":\"user\",\"content\":\"...\"}]}",
      "responsePayload": "{\"choices\":[{\"message\":{\"content\":\"...\"}}]}"
    }
  ]
}
```

### Field Mapping
If your data uses different field names, use field mapping:

```python
samples = evaluator.load_data(
    "data.json",
    field_mapping={
        "question": "input",
        "answer": "output",
        "expected": "reference",
    }
)
```

## Configuration

### YAML Config
```yaml
openai:
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"

task_type: "qa"

enabled_metrics:
  - exact_match
  - token_f1

metrics:
  k_values: [1, 3, 5, 10]
  rouge_types: ["rouge1", "rouge2", "rougeL"]
```

### Load Config
```python
config = EvalConfig.from_yaml("config.yaml")
evaluator = LLMEvaluator(config)
```

## Examples

See `examples/basic_usage.py` for comprehensive examples including:
- Basic evaluation
- JSON/CSV file evaluation
- API log evaluation
- Ground truth generation
- RAG evaluation
- Tool calling evaluation
- Code evaluation
- And more...

## Using with OpenAI-Compatible APIs

The framework works with any OpenAI-compatible API:

```python
config = EvalConfig(
    openai=OpenAIConfig(
        api_key="your-api-key",
        base_url="https://your-api.example.com/v1",  # Your API endpoint
        model="your-model-name",
    )
)
```

Compatible providers:
- OpenAI
- Azure OpenAI
- Anthropic (via proxy)
- Local models (Ollama, vLLM, etc.)
- MISA AI API
- And more...

## Benchmark Results

Ground truth: `claude-sonnet-4-5` | Evaluated: `misa-ai-1.0-plus`, `mercury-2`, `misa-ai-1.1` | Date: 03/2026

> **[Xem chi tiết đầy đủ tại leaderboard/](leaderboard/)**

### Leaderboard tổng hợp

| App | Task | Metric | misa-ai-1.0-plus | mercury-2 | misa-ai-1.1 |
|-----|------|--------|:-----------------:|:---------:|:-----------:|
| `crmkh` | Product Rec. | ROUGE | **42.95%** | 49.06% ¹ | 35.86% |
| `crmkh` | Product Rec. | Token F1 | **80.61%** | 78.66% ¹ | 64.81% |
| `crmkh` | Product Rec. | JSON Valid | **100%** | 98% ¹ | **100%** |
| `crmkh` | Product Rec. | Latency | ~2.0s | ~2.1s ¹ | **~0.5s** |
| `crmmisa` | Biz Analysis | ROUGE | 30.70% | **31.72%** ¹ | **31.26%** |
| `crmmisa` | Biz Analysis | G-Eval | 50.00% | 50.00% ¹ | 50.00% |
| `crmmisa` | Biz Analysis | Relevancy | **94.40%** | 92.45% ¹ | **94.40%** |
| `crmmisa` | Biz Analysis | Latency | ~11.9s | ~2.9s ¹ | **~0.3s** |

> ¹ mercury-2 đánh giá ở Eval Run 1 (GT khác). Chỉ nên so sánh cùng superscript.
>
> Chi tiết: [crmkh](leaderboard/crmkh.md) | [crmmisa](leaderboard/crmmisa.md) | [Leaderboard đầy đủ](leaderboard/)

<!-- Thêm dự án mới:
1. Tạo file leaderboard/<app_code>.md theo template trong leaderboard/README.md
2. Thêm hàng vào bảng tổng hợp ở đây
3. Cập nhật leaderboard/README.md
-->

---

## License

MIT License
