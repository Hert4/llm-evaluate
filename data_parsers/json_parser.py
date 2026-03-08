"""
JSON Data Parser
"""
import json
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from .base import DataParser, EvalSample


class JSONParser(DataParser):
    """
    Parser for JSON data files.
    Supports various JSON structures.
    """

    def __init__(
        self,
        field_mapping: Optional[Dict[str, str]] = None,
        data_key: Optional[str] = None,
        nested_input_path: Optional[str] = None,
        nested_output_path: Optional[str] = None,
    ):
        """
        Initialize JSON parser.

        Args:
            field_mapping: Map field names to standard names
            data_key: Key to access data array (e.g., "data", "items", "results")
            nested_input_path: Dot-notation path to input (e.g., "messages.0.content")
            nested_output_path: Dot-notation path to output
        """
        super().__init__(field_mapping)
        self.data_key = data_key
        self.nested_input_path = nested_input_path
        self.nested_output_path = nested_output_path

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dict using dot notation"""
        if not path:
            return None

        keys = path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list):
                try:
                    idx = int(key)
                    current = current[idx] if idx < len(current) else None
                except ValueError:
                    current = None
            else:
                return None
            if current is None:
                return None
        return current

    def _extract_from_payload(self, payload_str: str, is_request: bool = True) -> Dict[str, Any]:
        """Extract data from JSON payload string (for API logs)"""
        try:
            payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
        except json.JSONDecodeError:
            return {}

        result = {}

        if is_request:
            # Extract input from messages
            messages = payload.get("messages", [])
            if messages:
                # Get the last user message as input
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    result["input"] = user_messages[-1].get("content", "")

                # Get system message as context
                system_messages = [m for m in messages if m.get("role") == "system"]
                if system_messages:
                    result["context"] = system_messages[0].get("content", "")

                # Store full conversation history
                result["conversation_history"] = messages

            # Extract tools/functions if present
            if "tools" in payload:
                result["expected_tool_calls"] = payload["tools"]
            if "functions" in payload:
                result["expected_tool_calls"] = payload["functions"]
        else:
            # Extract output from response
            choices = payload.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                result["output"] = message.get("content", "")

                # Extract tool calls if present
                if "tool_calls" in message:
                    result["tool_calls"] = message["tool_calls"]
                if "function_call" in message:
                    result["tool_calls"] = [message["function_call"]]

            # Also check for direct content
            if "content" in payload:
                result["output"] = payload["content"]

        return result

    def parse(self, data_source: Any) -> List[EvalSample]:
        """Parse JSON data into EvalSamples"""
        # Load data
        if isinstance(data_source, (str, Path)):
            with open(data_source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(data_source, dict):
            data = data_source
        elif isinstance(data_source, list):
            data = data_source
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

        # Extract data array
        if isinstance(data, dict):
            if self.data_key:
                items = data.get(self.data_key, [])
            else:
                # Auto-detect data key
                for key in ["data", "items", "results", "samples", "records", "logs"]:
                    if key in data and isinstance(data[key], list):
                        items = data[key]
                        break
                else:
                    # Treat as single item
                    items = [data]
        else:
            items = data

        samples = []
        for i, item in enumerate(items):
            sample = self._parse_item(item, i)
            if sample:
                samples.append(sample)

        return samples

    def _parse_item(self, item: Dict[str, Any], index: int) -> Optional[EvalSample]:
        """Parse single item into EvalSample"""
        # Apply field mapping
        mapped = self.map_fields(item)

        # Handle API log format (requestPayload/responsePayload)
        if "requestPayload" in item or "responsePayload" in item:
            request_data = self._extract_from_payload(
                item.get("requestPayload", "{}"), is_request=True
            )
            response_data = self._extract_from_payload(
                item.get("responsePayload", "{}"), is_request=False
            )
            mapped.update(request_data)
            mapped.update(response_data)

        # Handle nested paths
        if self.nested_input_path:
            nested_input = self._get_nested_value(item, self.nested_input_path)
            if nested_input:
                mapped["input"] = nested_input

        if self.nested_output_path:
            nested_output = self._get_nested_value(item, self.nested_output_path)
            if nested_output:
                mapped["output"] = nested_output

        # Extract standard fields
        sample_id = str(mapped.get("id", item.get("id", index)))
        input_text = str(mapped.get("input", mapped.get("prompt", mapped.get("question", ""))))
        output_text = str(mapped.get("output", mapped.get("response", mapped.get("answer", ""))))
        reference = mapped.get("reference", mapped.get("expected", mapped.get("ground_truth")))
        context = mapped.get("context", mapped.get("source", mapped.get("document")))

        # Skip if no input or output
        if not input_text and not output_text:
            return None

        # Build metadata
        metadata = {
            k: v for k, v in item.items()
            if k not in ["id", "input", "output", "reference", "context",
                        "requestPayload", "responsePayload"]
        }

        return EvalSample(
            id=sample_id,
            input=input_text,
            output=output_text,
            reference=str(reference) if reference else None,
            context=str(context) if context else None,
            metadata=metadata,
            tool_calls=mapped.get("tool_calls"),
            expected_tool_calls=mapped.get("expected_tool_calls"),
            conversation_history=mapped.get("conversation_history"),
            candidates=mapped.get("candidates"),
            relevance_scores=mapped.get("relevance_scores"),
            test_cases=mapped.get("test_cases"),
            choices=mapped.get("choices", mapped.get("options")),
            correct_answer=mapped.get("correct_answer", mapped.get("label")),
        )

    def parse_stream(self, data_source: Any) -> Iterator[EvalSample]:
        """Stream parse for large JSON files"""
        if isinstance(data_source, (str, Path)):
            # Try to stream parse for large files
            try:
                import ijson

                with open(data_source, "rb") as f:
                    # Try to find the data array
                    items = ijson.items(f, "data.item")
                    for i, item in enumerate(items):
                        sample = self._parse_item(item, i)
                        if sample:
                            yield sample
            except ImportError:
                # Fallback to regular parsing
                for sample in self.parse(data_source):
                    yield sample
        else:
            for sample in self.parse(data_source):
                yield sample
