"""
Base Data Parser
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
import json


@dataclass
class EvalSample:
    """
    Standardized evaluation sample format.
    All parsers convert data to this format.
    """
    id: str
    input: str  # The input/prompt/question
    output: str  # The model's actual output/prediction
    reference: Optional[str] = None  # Ground truth / expected output
    context: Optional[str] = None  # Context for RAG evaluation
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For tool calling evaluation
    tool_calls: Optional[List[Dict[str, Any]]] = None
    expected_tool_calls: Optional[List[Dict[str, Any]]] = None

    # For multi-turn conversations
    conversation_history: Optional[List[Dict[str, str]]] = None

    # For ranking evaluation
    candidates: Optional[List[str]] = None
    relevance_scores: Optional[List[float]] = None

    # For code evaluation
    test_cases: Optional[List[Dict[str, Any]]] = None

    # For multiple choice
    choices: Optional[List[str]] = None
    correct_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "reference": self.reference,
            "context": self.context,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "expected_tool_calls": self.expected_tool_calls,
            "conversation_history": self.conversation_history,
            "candidates": self.candidates,
            "relevance_scores": self.relevance_scores,
            "test_cases": self.test_cases,
            "choices": self.choices,
            "correct_answer": self.correct_answer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalSample":
        """Create from dictionary"""
        return cls(
            id=str(data.get("id", "")),
            input=str(data.get("input", "")),
            output=str(data.get("output", "")),
            reference=data.get("reference"),
            context=data.get("context"),
            metadata=data.get("metadata", {}),
            tool_calls=data.get("tool_calls"),
            expected_tool_calls=data.get("expected_tool_calls"),
            conversation_history=data.get("conversation_history"),
            candidates=data.get("candidates"),
            relevance_scores=data.get("relevance_scores"),
            test_cases=data.get("test_cases"),
            choices=data.get("choices"),
            correct_answer=data.get("correct_answer"),
        )


class DataParser(ABC):
    """
    Abstract base class for data parsers.
    All parsers must implement the parse method.
    """

    def __init__(self, field_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize parser with optional field mapping.

        Args:
            field_mapping: Map from your data field names to standard names.
                           e.g., {"question": "input", "answer": "output"}
        """
        self.field_mapping = field_mapping or {}

    @abstractmethod
    def parse(self, data_source: Any) -> List[EvalSample]:
        """
        Parse data source into list of EvalSample.

        Args:
            data_source: Can be file path, raw data, or any supported format

        Returns:
            List of standardized EvalSample objects
        """
        pass

    def parse_stream(self, data_source: Any) -> Iterator[EvalSample]:
        """
        Stream parse for large files. Override for streaming support.
        """
        for sample in self.parse(data_source):
            yield sample

    def map_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply field mapping to data dict"""
        mapped = {}
        for key, value in data.items():
            new_key = self.field_mapping.get(key, key)
            mapped[new_key] = value
        return mapped

    def detect_task_type(self, samples: List[EvalSample]) -> str:
        """
        Detect task type from samples.
        Override for custom detection logic.
        """
        if not samples:
            return "custom"

        sample = samples[0]

        # Check for tool calling
        if sample.tool_calls or sample.expected_tool_calls:
            return "tool_calling"

        # Check for code
        if sample.test_cases:
            return "coding"

        # Check for multiple choice
        if sample.choices:
            return "reasoning"

        # Check for ranking
        if sample.candidates or sample.relevance_scores:
            return "ranking"

        # Check for RAG (has context)
        if sample.context:
            return "rag"

        # Check for QA (short reference)
        if sample.reference and len(sample.reference.split()) < 20:
            return "qa"

        # Check for summarization (longer reference)
        if sample.reference and len(sample.reference.split()) >= 20:
            return "summarization"

        # Check for chat (conversation history)
        if sample.conversation_history:
            return "chat"

        return "custom"
