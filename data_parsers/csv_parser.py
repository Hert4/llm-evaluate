"""
CSV Data Parser
"""
import csv
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from .base import DataParser, EvalSample


class CSVParser(DataParser):
    """
    Parser for CSV/TSV data files.
    """

    def __init__(
        self,
        field_mapping: Optional[Dict[str, str]] = None,
        delimiter: str = ",",
        has_header: bool = True,
    ):
        """
        Initialize CSV parser.

        Args:
            field_mapping: Map field names to standard names
            delimiter: CSV delimiter (default: comma)
            has_header: Whether first row is header
        """
        super().__init__(field_mapping)
        self.delimiter = delimiter
        self.has_header = has_header

    def parse(self, data_source: Any) -> List[EvalSample]:
        """Parse CSV data into EvalSamples"""
        if isinstance(data_source, (str, Path)):
            with open(data_source, "r", encoding="utf-8") as f:
                return self._parse_file(f)
        elif hasattr(data_source, "read"):
            return self._parse_file(data_source)
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

    def _parse_file(self, file_obj) -> List[EvalSample]:
        """Parse CSV file object"""
        reader = csv.reader(file_obj, delimiter=self.delimiter)

        if self.has_header:
            headers = next(reader)
            headers = [self.field_mapping.get(h, h) for h in headers]
        else:
            # Use default headers
            headers = ["input", "output", "reference"]

        samples = []
        for i, row in enumerate(reader):
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))

            item = dict(zip(headers, row))
            sample = self._parse_item(item, i)
            if sample:
                samples.append(sample)

        return samples

    def _parse_item(self, item: Dict[str, Any], index: int) -> Optional[EvalSample]:
        """Parse single row into EvalSample"""
        sample_id = str(item.get("id", index))
        input_text = str(item.get("input", item.get("prompt", item.get("question", ""))))
        output_text = str(item.get("output", item.get("response", item.get("answer", ""))))
        reference = item.get("reference", item.get("expected", item.get("ground_truth")))
        context = item.get("context", item.get("source", item.get("document")))

        if not input_text and not output_text:
            return None

        metadata = {k: v for k, v in item.items() if k not in ["id", "input", "output", "reference", "context"]}

        return EvalSample(
            id=sample_id,
            input=input_text,
            output=output_text,
            reference=str(reference) if reference else None,
            context=str(context) if context else None,
            metadata=metadata,
            choices=item.get("choices", "").split("|") if item.get("choices") else None,
            correct_answer=item.get("correct_answer", item.get("label")),
        )

    def parse_stream(self, data_source: Any) -> Iterator[EvalSample]:
        """Stream parse for large CSV files"""
        if isinstance(data_source, (str, Path)):
            with open(data_source, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=self.delimiter)

                if self.has_header:
                    headers = next(reader)
                    headers = [self.field_mapping.get(h, h) for h in headers]
                else:
                    headers = ["input", "output", "reference"]

                for i, row in enumerate(reader):
                    if len(row) < len(headers):
                        row.extend([""] * (len(headers) - len(row)))

                    item = dict(zip(headers, row))
                    sample = self._parse_item(item, i)
                    if sample:
                        yield sample
        else:
            for sample in self.parse(data_source):
                yield sample
