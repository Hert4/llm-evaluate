"""
Configuration for LLM Evaluation Framework
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class MetricType(Enum):
    # Translation
    BLEU = "bleu"
    COMET = "comet"

    # Summarization
    ROUGE = "rouge"
    G_EVAL = "g_eval"

    # QA
    EXACT_MATCH = "exact_match"
    TOKEN_F1 = "token_f1"

    # Ranking
    NDCG = "ndcg"
    RECALL_AT_K = "recall_at_k"
    PRECISION_AT_K = "precision_at_k"
    MRR = "mrr"

    # Tool/Agent
    AST_ACCURACY = "ast_accuracy"
    TASK_SUCCESS_RATE = "task_success_rate"

    # Coding
    PASS_AT_K = "pass_at_k"

    # Reasoning
    ACCURACY = "accuracy"

    # RAG
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    ANSWER_RELEVANCY = "answer_relevancy"

    # Safety
    FACTSCORE = "factscore"
    IFEVAL = "ifeval"

    # Chat
    WIN_RATE = "win_rate"
    PAIRWISE_COMPARISON = "pairwise_comparison"

    # List Match (for recommendation / forecast)
    LIST_MATCH = "list_match"


class TaskType(Enum):
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QA = "qa"
    RANKING = "ranking"
    TOOL_CALLING = "tool_calling"
    AGENT = "agent"
    CODING = "coding"
    REASONING = "reasoning"
    RAG = "rag"
    SAFETY = "safety"
    CHAT = "chat"
    CUSTOM = "custom"


# Mapping from task type to recommended metrics
TASK_METRICS_MAP: Dict[TaskType, List[MetricType]] = {
    TaskType.TRANSLATION: [MetricType.BLEU, MetricType.COMET],
    TaskType.SUMMARIZATION: [MetricType.ROUGE, MetricType.G_EVAL],
    TaskType.QA: [MetricType.EXACT_MATCH, MetricType.TOKEN_F1],
    TaskType.RANKING: [MetricType.NDCG, MetricType.RECALL_AT_K, MetricType.MRR],
    TaskType.TOOL_CALLING: [MetricType.AST_ACCURACY],
    TaskType.AGENT: [MetricType.TASK_SUCCESS_RATE],
    TaskType.CODING: [MetricType.PASS_AT_K],
    TaskType.REASONING: [MetricType.ACCURACY],
    TaskType.RAG: [MetricType.FAITHFULNESS, MetricType.CONTEXT_PRECISION, MetricType.CONTEXT_RECALL],
    TaskType.SAFETY: [MetricType.FACTSCORE, MetricType.IFEVAL],
    TaskType.CHAT: [MetricType.WIN_RATE, MetricType.G_EVAL],
}


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI-compatible API"""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class MetricConfig:
    """Configuration for individual metrics"""
    # BLEU
    bleu_max_ngram: int = 4
    bleu_smoothing: bool = True

    # ROUGE
    rouge_types: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])

    # NDCG/Recall
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    # Pass@K
    pass_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    code_timeout: int = 30

    # G-Eval criteria
    g_eval_criteria: List[str] = field(default_factory=lambda: [
        "coherence", "fluency", "relevance", "consistency"
    ])

    # Faithfulness threshold
    faithfulness_threshold: float = 0.5

    # IFEval
    ifeval_strict: bool = True


@dataclass
class EvalConfig:
    """Main configuration for evaluation"""
    # OpenAI-compatible API config
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)

    # Metric-specific configs
    metrics: MetricConfig = field(default_factory=MetricConfig)

    # Task type (auto-detected if not specified)
    task_type: Optional[TaskType] = None

    # Specific metrics to run (if None, auto-select based on task_type)
    enabled_metrics: Optional[List[MetricType]] = None

    # Data parsing
    auto_detect_format: bool = True

    # Input/Output field names (for flexible data formats)
    input_field: str = "input"
    output_field: str = "output"
    reference_field: str = "reference"
    context_field: str = "context"

    # Batch processing
    batch_size: int = 10
    num_workers: int = 4

    # Output
    output_format: str = "json"  # json, csv, html
    save_detailed_results: bool = True

    # Caching
    enable_cache: bool = True
    cache_dir: str = ".eval_cache"

    # Logging
    verbose: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvalConfig":
        """Create config from dictionary"""
        openai_config = OpenAIConfig(**config_dict.get("openai", {}))
        metric_config = MetricConfig(**config_dict.get("metrics", {}))

        return cls(
            openai=openai_config,
            metrics=metric_config,
            task_type=TaskType(config_dict["task_type"]) if config_dict.get("task_type") else None,
            enabled_metrics=[MetricType(m) for m in config_dict.get("enabled_metrics", [])] if config_dict.get("enabled_metrics") else None,
            **{k: v for k, v in config_dict.items() if k not in ["openai", "metrics", "task_type", "enabled_metrics"]}
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvalConfig":
        """Load config from YAML file"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "EvalConfig":
        """Load config from JSON file"""
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
