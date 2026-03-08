"""
Auto Parser - Automatically detects data format
"""
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from .base import DataParser, EvalSample
from .json_parser import JSONParser
from .csv_parser import CSVParser
from .log_parser import LogParser


class AutoParser(DataParser):
    """
    Automatically detects data format and uses appropriate parser.
    Supports JSON, CSV, TSV, JSONL, and API log formats.
    """

    def __init__(
        self,
        field_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize auto parser.

        Args:
            field_mapping: Map field names to standard names
            **kwargs: Additional arguments passed to specific parsers
        """
        super().__init__(field_mapping)
        self.kwargs = kwargs

    def _detect_format(self, data_source: Any) -> str:
        """Detect data format from source"""
        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            suffix = path.suffix.lower()

            if suffix == ".json":
                # Check if it's API logs or regular JSON
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if self._is_api_log(data):
                            return "log"
                        return "json"
                    except json.JSONDecodeError:
                        return "json"

            elif suffix == ".jsonl":
                return "jsonl"

            elif suffix == ".csv":
                return "csv"

            elif suffix == ".tsv":
                return "tsv"

            else:
                # Try to detect from content
                with open(path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    if first_line.strip().startswith("{"):
                        return "json"
                    elif "\t" in first_line:
                        return "tsv"
                    elif "," in first_line:
                        return "csv"

        elif isinstance(data_source, dict):
            if self._is_api_log(data_source):
                return "log"
            return "json"

        elif isinstance(data_source, list):
            if data_source and self._is_api_log({"data": data_source}):
                return "log"
            return "json"

        return "json"

    def _is_api_log(self, data: Union[Dict, List]) -> bool:
        """Check if data looks like API logs"""
        items = []
        if isinstance(data, dict):
            for key in ["data", "logs", "items"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
        elif isinstance(data, list):
            items = data

        if not items:
            return False

        # Check first item for log indicators
        first_item = items[0] if items else {}
        log_fields = ["requestPayload", "responsePayload", "traceId", "consumerName", "endpointApi"]
        return any(field in first_item for field in log_fields)

    def _get_parser(self, format_type: str) -> DataParser:
        """Get appropriate parser for format"""
        if format_type == "log":
            return LogParser(self.field_mapping, **self.kwargs)
        elif format_type == "csv":
            return CSVParser(self.field_mapping, delimiter=",", **self.kwargs)
        elif format_type == "tsv":
            return CSVParser(self.field_mapping, delimiter="\t", **self.kwargs)
        elif format_type == "jsonl":
            return JSONParser(self.field_mapping, **self.kwargs)
        else:
            return JSONParser(self.field_mapping, **self.kwargs)

    def parse(self, data_source: Any) -> List[EvalSample]:
        """Auto-detect format and parse data"""
        format_type = self._detect_format(data_source)
        parser = self._get_parser(format_type)
        return parser.parse(data_source)

    def parse_with_format(self, data_source: Any) -> tuple[List[EvalSample], str]:
        """Parse and return detected format"""
        format_type = self._detect_format(data_source)
        parser = self._get_parser(format_type)
        samples = parser.parse(data_source)
        return samples, format_type


class FlexibleFieldParser(DataParser):
    """
    Parser that can handle any field names by trying multiple common patterns.
    """

    # Common field name patterns for each standard field
    FIELD_PATTERNS = {
        "input": ["input", "prompt", "question", "query", "text", "message", "content", "instruction"],
        "output": ["output", "response", "answer", "reply", "completion", "generated", "prediction"],
        "reference": ["reference", "expected", "ground_truth", "gt", "label", "target", "gold"],
        "context": ["context", "source", "document", "passage", "retrieved", "knowledge"],
        "id": ["id", "idx", "index", "sample_id", "example_id"],
    }

    def __init__(self, base_parser: Optional[DataParser] = None):
        super().__init__()
        self.base_parser = base_parser or JSONParser()

    def _find_field_mapping(self, sample_data: Dict[str, Any]) -> Dict[str, str]:
        """Automatically find field mapping from sample data"""
        mapping = {}
        used_fields = set()

        for standard_field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                # Exact match
                if pattern in sample_data and pattern not in used_fields:
                    mapping[pattern] = standard_field
                    used_fields.add(pattern)
                    break

                # Case-insensitive match
                for field in sample_data:
                    if field.lower() == pattern.lower() and field not in used_fields:
                        mapping[field] = standard_field
                        used_fields.add(field)
                        break

        return mapping

    def parse(self, data_source: Any) -> List[EvalSample]:
        """Parse with auto-detected field mapping"""
        # First, try to get a sample to detect fields
        if isinstance(data_source, (str, Path)):
            with open(data_source, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = data_source

        # Get first sample for field detection
        if isinstance(data, dict):
            for key in ["data", "items", "results", "samples"]:
                if key in data and isinstance(data[key], list):
                    sample_data = data[key][0] if data[key] else {}
                    break
            else:
                sample_data = data
        elif isinstance(data, list):
            sample_data = data[0] if data else {}
        else:
            sample_data = {}

        # Detect field mapping
        field_mapping = self._find_field_mapping(sample_data)

        # Create parser with mapping
        self.base_parser.field_mapping = field_mapping
        return self.base_parser.parse(data_source)
