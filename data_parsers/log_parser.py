"""
Log Parser for API call logs
"""
import json
import re
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from datetime import datetime
from .base import DataParser, EvalSample


class LogParser(DataParser):
    """
    Parser for API call logs (like OpenAI/MISA API logs).
    Handles requestPayload/responsePayload format.
    """

    def __init__(
        self,
        field_mapping: Optional[Dict[str, str]] = None,
        extract_tool_calls: bool = True,
        extract_conversation: bool = True,
        filter_by_model: Optional[str] = None,
        filter_by_app: Optional[str] = None,
    ):
        """
        Initialize log parser.

        Args:
            field_mapping: Map field names to standard names
            extract_tool_calls: Extract tool/function calls
            extract_conversation: Extract conversation history
            filter_by_model: Only include logs from specific model
            filter_by_app: Only include logs from specific application
        """
        super().__init__(field_mapping)
        self.extract_tool_calls = extract_tool_calls
        self.extract_conversation = extract_conversation
        self.filter_by_model = filter_by_model
        self.filter_by_app = filter_by_app

    def _parse_payload(self, payload: Any) -> Dict:
        """Parse JSON payload (string or dict)"""
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return {}
        elif isinstance(payload, dict):
            return payload
        return {}

    def _extract_input_from_request(self, request: Dict) -> Dict[str, Any]:
        """Extract input data from request payload"""
        result = {}
        messages = request.get("messages", [])

        if messages:
            # Get user messages
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                result["input"] = user_messages[-1].get("content", "")

            # Get system prompt as context
            system_messages = [m for m in messages if m.get("role") == "system"]
            if system_messages:
                result["context"] = system_messages[0].get("content", "")

            # Full conversation
            if self.extract_conversation:
                result["conversation_history"] = messages

        # Extract expected tools
        if self.extract_tool_calls:
            if "tools" in request:
                result["expected_tool_calls"] = request["tools"]
            elif "functions" in request:
                result["expected_tool_calls"] = request["functions"]

        return result

    def _extract_output_from_response(self, response: Dict) -> Dict[str, Any]:
        """Extract output data from response payload"""
        result = {}

        # Standard OpenAI format
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            result["output"] = content if content else ""

            # Extract tool calls
            if self.extract_tool_calls:
                if "tool_calls" in message:
                    result["tool_calls"] = message["tool_calls"]
                elif "function_call" in message:
                    result["tool_calls"] = [message["function_call"]]

        # Direct content (some APIs)
        elif "content" in response:
            result["output"] = response["content"]

        # Text field (some APIs)
        elif "text" in response:
            result["output"] = response["text"]

        return result

    def parse(self, data_source: Any) -> List[EvalSample]:
        """Parse log data into EvalSamples"""
        # Load data
        if isinstance(data_source, (str, Path)):
            with open(data_source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(data_source, dict):
            data = data_source
        elif isinstance(data_source, list):
            data = {"data": data_source}
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

        # Extract data array
        if isinstance(data, dict):
            for key in ["data", "logs", "items", "records"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            else:
                items = [data]
        else:
            items = data

        samples = []
        for i, item in enumerate(items):
            # Apply filters
            if self.filter_by_model and item.get("model") != self.filter_by_model:
                continue
            if self.filter_by_app and item.get("applicationCode") != self.filter_by_app:
                continue

            sample = self._parse_log_item(item, i)
            if sample:
                samples.append(sample)

        return samples

    def _parse_log_item(self, item: Dict[str, Any], index: int) -> Optional[EvalSample]:
        """Parse single log entry into EvalSample"""
        # Parse request and response
        request = self._parse_payload(item.get("requestPayload", {}))
        response = self._parse_payload(item.get("responsePayload", {}))

        # Extract input and output
        input_data = self._extract_input_from_request(request)
        output_data = self._extract_output_from_response(response)

        input_text = input_data.get("input", "")
        output_text = output_data.get("output", "")

        if not input_text and not output_text:
            return None

        # Build metadata
        metadata = {
            "model": item.get("model"),
            "application": item.get("applicationCode"),
            "consumer": item.get("consumerName"),
            "endpoint": item.get("endpointApi"),
            "trace_id": item.get("traceId"),
        }

        return EvalSample(
            id=str(item.get("id", index)),
            input=input_text,
            output=output_text,
            reference=None,  # Logs don't have ground truth
            context=input_data.get("context"),
            metadata=metadata,
            tool_calls=output_data.get("tool_calls"),
            expected_tool_calls=input_data.get("expected_tool_calls"),
            conversation_history=input_data.get("conversation_history"),
        )

    def parse_stream(self, data_source: Any) -> Iterator[EvalSample]:
        """Stream parse for large log files"""
        if isinstance(data_source, (str, Path)):
            try:
                import ijson

                with open(data_source, "rb") as f:
                    items = ijson.items(f, "data.item")
                    for i, item in enumerate(items):
                        if self.filter_by_model and item.get("model") != self.filter_by_model:
                            continue
                        if self.filter_by_app and item.get("applicationCode") != self.filter_by_app:
                            continue

                        sample = self._parse_log_item(item, i)
                        if sample:
                            yield sample
            except ImportError:
                for sample in self.parse(data_source):
                    yield sample
        else:
            for sample in self.parse(data_source):
                yield sample
