"""
LLM Evaluation Framework
========================
A comprehensive framework for evaluating LLM outputs using multiple metrics.
Supports any data format and uses OpenAI-compatible API for ground truth generation.

Metrics supported:
- Translation: BLEU, COMET
- Summarization: ROUGE, G-Eval
- QA: Exact Match, Token F1
- Ranking: NDCG@K, Recall@K
- Tool/Agent: AST Accuracy, Task Success Rate
- Coding: Pass@K
- Reasoning: Accuracy (Multiple Choice)
- RAG: Faithfulness, Context Precision, Context Recall
- Safety: FactScore, IFEval
- Chat: Win Rate (LLM-as-Judge)
"""

from .evaluator import LLMEvaluator
from .config import EvalConfig
from .data_parsers import DataParser, AutoParser

__version__ = "1.0.0"
__all__ = ["LLMEvaluator", "EvalConfig", "DataParser", "AutoParser"]
