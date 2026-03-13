#!/usr/bin/env python3
"""
Convert raw API log data to benchmark datasets for LLM Evaluation Framework.

All task definitions are loaded from YAML config files in tasks/ directory.
The script is fully data-driven — add a new YAML file = add a new benchmark.

Usage:
    python scripts/convert_raw_to_benchmark.py
    python scripts/convert_raw_to_benchmark.py --task crm_recommendation
    python scripts/convert_raw_to_benchmark.py --samples 300 --seed 123
    python scripts/convert_raw_to_benchmark.py --list-tasks
    python scripts/convert_raw_to_benchmark.py --config tasks/my_task.yaml
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TASKS_DIR = SCRIPT_DIR / "tasks"
LANGUAGE_MAPPING_FILE = TASKS_DIR / "language_mapping.yaml"

# ---------------------------------------------------------------------------
# Language mapping cache
# ---------------------------------------------------------------------------
_LANG_MAP: Optional[Dict[str, str]] = None


def get_language_mapping() -> Dict[str, str]:
    """Load Vietnamese language name → locale code mapping (cached)."""
    global _LANG_MAP
    if _LANG_MAP is None:
        if LANGUAGE_MAPPING_FILE.is_file():
            with open(LANGUAGE_MAPPING_FILE, "r", encoding="utf-8") as f:
                _LANG_MAP = yaml.safe_load(f) or {}
            logger.info(f"Loaded language mapping: {len(_LANG_MAP)} entries")
        else:
            _LANG_MAP = {}
    return _LANG_MAP


# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
@dataclass
class GlobalConfig:
    """Pipeline-level configuration (from CLI args)."""

    raw_data_dir: str = str(PROJECT_ROOT.parent / "data")
    output_dir: str = str(PROJECT_ROOT / "data" / "benchmarks")
    random_seed: int = 42
    sampling_strategy: str = "random"  # random | stratified
    skip_errors: bool = True
    skip_null_responses: bool = True
    deduplicate: bool = True
    sample_size_override: Optional[int] = None  # CLI --samples


# ---------------------------------------------------------------------------
# Task Definition (loaded from YAML)
# ---------------------------------------------------------------------------
@dataclass
class TaskDefinition:
    """A single benchmark task definition — everything from the YAML file."""

    # Required
    task_name: str = ""
    task_type: str = "qa"
    source_file: str = ""

    # Sampling
    max_samples: int = 500  # 0 = take all
    sampling_strategy: str = ""  # override global; "" = use global

    # Filtering
    filters: Dict[str, Any] = field(default_factory=dict)
    # filters.application_code: str or list[str]
    # filters.system_prompt_contains: str or list[str]  (ANY match)
    # filters.system_prompt_not_contains: str or list[str]
    # filters.model: str
    # filters.exclude_application_code: str or list[str]

    # Extra metadata extraction
    extra_metadata_fields: List[str] = field(default_factory=list)
    # e.g. ["consumerName"]  →  metadata["consumerName"] = record["consumerName"]

    metadata_regex: Dict[str, str] = field(default_factory=dict)
    # e.g. {"target_language": "sang ngôn ngữ (.+?)(?:\\s+một cách|\\s*[.\\n])"}
    # Extracts from user_message, adds to metadata

    # Stratified sampling key (metadata field name)
    stratified_key: str = ""
    # e.g. "target_language" → group by this metadata key for stratified sampling

    # Language mapping: normalize Vietnamese lang names → locale codes
    # If set, applies language_mapping.yaml to the specified metadata_regex field
    language_mapping_field: str = ""
    # e.g. "target_language" → after regex extraction, map "tiếng Séc" → "cs"

    # Description (for --list-tasks)
    description: str = ""

    @classmethod
    def from_yaml(cls, filepath: Path) -> "TaskDefinition":
        """Load task definition from a YAML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            raise ValueError(f"Empty YAML: {filepath}")

        td = cls()
        td.task_name = raw.get("task_name", filepath.stem)
        td.task_type = raw.get("task_type", "qa")
        td.source_file = raw.get("source_file", "")
        td.max_samples = raw.get("max_samples", 500)
        td.sampling_strategy = raw.get("sampling_strategy", "")
        td.filters = raw.get("filters", {})
        td.extra_metadata_fields = raw.get("extra_metadata_fields", [])
        td.metadata_regex = raw.get("metadata_regex", {})
        td.stratified_key = raw.get("stratified_key", "")
        td.language_mapping_field = raw.get("language_mapping_field", "")
        td.description = raw.get("description", "")
        return td

    def to_yaml(self) -> str:
        """Serialize back to YAML string."""
        d = {
            "task_name": self.task_name,
            "task_type": self.task_type,
            "source_file": self.source_file,
            "max_samples": self.max_samples,
            "description": self.description,
        }
        if self.sampling_strategy:
            d["sampling_strategy"] = self.sampling_strategy
        if self.filters:
            d["filters"] = self.filters
        if self.extra_metadata_fields:
            d["extra_metadata_fields"] = self.extra_metadata_fields
        if self.metadata_regex:
            d["metadata_regex"] = self.metadata_regex
        if self.stratified_key:
            d["stratified_key"] = self.stratified_key
        return yaml.dump(d, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Statistics tracker
# ---------------------------------------------------------------------------
@dataclass
class ConversionStats:
    """Track conversion statistics for a single task."""

    task_name: str = ""
    source_file: str = ""
    total_raw: int = 0
    after_task_filter: int = 0
    after_error_filter: int = 0
    after_dedup: int = 0
    parse_errors: int = 0
    final_samples: int = 0

    def summary(self) -> str:
        return (
            f"  {self.task_name}:\n"
            f"    Source: {self.source_file}\n"
            f"    Raw records:      {self.total_raw:>6}\n"
            f"    After task filter: {self.after_task_filter:>6}\n"
            f"    After error filter:{self.after_error_filter:>6}\n"
            f"    After dedup:       {self.after_dedup:>6}\n"
            f"    Parse errors:      {self.parse_errors:>6}\n"
            f"    Final samples:     {self.final_samples:>6}"
        )


# ---------------------------------------------------------------------------
# Data loading cache
# ---------------------------------------------------------------------------
_FILE_CACHE: Dict[str, List[Dict]] = {}


def load_raw_data(filepath: str) -> List[Dict]:
    """Load raw data with file-level caching."""
    if filepath not in _FILE_CACHE:
        logger.info(f"Loading {Path(filepath).name} ...")
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _FILE_CACHE[filepath] = raw.get("data", raw if isinstance(raw, list) else [])
        logger.info(f"  Loaded {len(_FILE_CACHE[filepath])} records")
    return _FILE_CACHE[filepath]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_payload(payload: Any) -> Dict:
    """Safely parse a JSON payload (string or dict)."""
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(payload, dict):
        return payload
    return {}


def content_hash(text: str) -> str:
    """Generate a short hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def extract_messages(request: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract (system_prompt, user_message) from request payload."""
    messages = request.get("messages", [])
    system_msg = None
    user_msg = None
    for m in messages:
        role = m.get("role", "")
        if role == "system" and system_msg is None:
            system_msg = m.get("content", "")
        elif role == "user":
            user_msg = m.get("content", "")
    return system_msg, user_msg


def extract_response_content(response: Dict) -> Optional[str]:
    """Extract assistant response text from response payload."""
    choices = response.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if content:
            return content
    return None


def extract_usage(response: Dict) -> Dict[str, Any]:
    """Extract token usage info from response payload."""
    usage = response.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _to_list(value: Any) -> List[str]:
    """Normalize a string-or-list config value to list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


# ---------------------------------------------------------------------------
# Generic Converter (data-driven by TaskDefinition)
# ---------------------------------------------------------------------------
class GenericConverter:
    """
    A single converter driven entirely by a TaskDefinition (from YAML).
    No subclassing needed — all behaviour comes from config.
    """

    def __init__(self, task_def: TaskDefinition, global_config: GlobalConfig):
        self.task_def = task_def
        self.global_config = global_config
        self.stats = ConversionStats()

        # Pre-compile regex extractors
        self._regex_extractors: Dict[str, re.Pattern] = {}
        for key, pattern in task_def.metadata_regex.items():
            self._regex_extractors[key] = re.compile(pattern)

    # --- Filtering -----------------------------------------------------------

    def _matches_task(self, record: Dict) -> bool:
        """Check if record passes all filters defined in YAML."""
        filters = self.task_def.filters
        if not filters:
            return True

        # application_code filter
        app_codes = _to_list(filters.get("application_code"))
        if app_codes and record.get("applicationCode") not in app_codes:
            return False

        # exclude_application_code
        exclude_apps = _to_list(filters.get("exclude_application_code"))
        if exclude_apps and record.get("applicationCode") in exclude_apps:
            return False

        # model filter
        model_filter = filters.get("model")
        if model_filter and record.get("model") != model_filter:
            return False

        # system_prompt_contains (ANY marker must match)
        prompt_contains = _to_list(filters.get("system_prompt_contains"))
        if prompt_contains:
            request = parse_payload(record.get("requestPayload", ""))
            system_msg, _ = extract_messages(request)
            if not system_msg:
                return False
            if not any(marker in system_msg for marker in prompt_contains):
                return False

        # system_prompt_not_contains (NONE of the markers should match)
        prompt_not_contains = _to_list(filters.get("system_prompt_not_contains"))
        if prompt_not_contains:
            request = parse_payload(record.get("requestPayload", ""))
            system_msg, _ = extract_messages(request)
            if system_msg and any(marker in system_msg for marker in prompt_not_contains):
                return False

        return True

    # --- Extraction ----------------------------------------------------------

    def _extract_sample(self, record: Dict, index: int) -> Optional[Dict]:
        """Extract an EvalSample dict from a raw record."""
        request = parse_payload(record.get("requestPayload", ""))
        response = parse_payload(record.get("responsePayload", ""))

        system_msg, user_msg = extract_messages(request)
        output_text = extract_response_content(response)

        if not user_msg or not output_text:
            return None

        usage = extract_usage(response)

        # Base metadata
        metadata = {
            "model": record.get("model"),
            "trace_id": record.get("traceId"),
            "application": record.get("applicationCode"),
            "processing_time_ms": record.get("processingTimeMs"),
            "created_at": record.get("createdAt"),
            **usage,
        }

        # Extra metadata fields from record (e.g. consumerName)
        for field_name in self.task_def.extra_metadata_fields:
            metadata[field_name] = record.get(field_name)

        # Regex-based metadata extraction from user message
        for key, pattern in self._regex_extractors.items():
            match = pattern.search(user_msg)
            metadata[key] = match.group(1).strip() if match else None

        # Language mapping: convert Vietnamese name → locale code
        if self.task_def.language_mapping_field:
            lm_field = self.task_def.language_mapping_field
            raw_lang = metadata.get(lm_field)
            if raw_lang:
                lang_map = get_language_mapping()
                locale_code = lang_map.get(raw_lang)
                if locale_code:
                    metadata[f"{lm_field}_original"] = raw_lang
                    metadata[lm_field] = locale_code
                else:
                    # Keep original, log once
                    metadata[f"{lm_field}_original"] = raw_lang

        return {
            "id": "",
            "input": user_msg,
            "output": output_text,
            "reference": None,
            "context": system_msg,
            "metadata": metadata,
        }

    # --- Sampling ------------------------------------------------------------

    def _effective_strategy(self) -> str:
        """Get sampling strategy: task-level override > global."""
        if self.task_def.sampling_strategy:
            return self.task_def.sampling_strategy
        return self.global_config.sampling_strategy

    def _effective_max_samples(self) -> int:
        """Get max samples: CLI override > task-level."""
        if self.global_config.sample_size_override is not None:
            return self.global_config.sample_size_override
        return self.task_def.max_samples

    def _sample(self, samples: List[Dict], rng: random.Random) -> List[Dict]:
        """Sample down to max_samples."""
        max_n = self._effective_max_samples()
        if max_n == 0 or len(samples) <= max_n:
            return samples

        strategy = self._effective_strategy()

        if strategy == "stratified" and self.task_def.stratified_key:
            return self._stratified_sample(samples, max_n, rng)
        else:
            return rng.sample(samples, max_n)

    def _stratified_sample(
        self, samples: List[Dict], max_n: int, rng: random.Random
    ) -> List[Dict]:
        """Stratified sampling by the configured metadata key."""
        key = self.task_def.stratified_key
        groups: Dict[str, List[Dict]] = {}
        for s in samples:
            group_val = s.get("metadata", {}).get(key, "unknown")
            groups.setdefault(str(group_val), []).append(s)

        n_groups = len(groups)
        if n_groups == 0:
            return rng.sample(samples, min(max_n, len(samples)))

        base_per_group = max(1, max_n // n_groups)
        result = []

        for _, group_samples in sorted(groups.items()):
            quota = min(base_per_group, len(group_samples))
            result.extend(rng.sample(group_samples, quota))

        # Fill remaining slots
        if len(result) < max_n:
            used_ids = {id(s) for s in result}
            remaining = [s for s in samples if id(s) not in used_ids]
            rng.shuffle(remaining)
            result.extend(remaining[: max_n - len(result)])

        if len(result) > max_n:
            result = rng.sample(result, max_n)

        return result

    # --- Main pipeline -------------------------------------------------------

    def convert(self) -> List[Dict]:
        """Run the full conversion pipeline."""
        filepath = str(
            Path(self.global_config.raw_data_dir) / self.task_def.source_file
        )
        all_records = load_raw_data(filepath)

        self.stats.task_name = self.task_def.task_name
        self.stats.source_file = self.task_def.source_file
        self.stats.total_raw = len(all_records)

        # Step 1: Filter by task
        task_records = [r for r in all_records if self._matches_task(r)]
        self.stats.after_task_filter = len(task_records)
        logger.info(
            f"[{self.task_def.task_name}] Task filter: "
            f"{len(all_records)} → {len(task_records)}"
        )

        # Step 2: Filter errors and null responses
        valid_records = []
        for r in task_records:
            if self.global_config.skip_errors and r.get("isError"):
                continue
            if self.global_config.skip_null_responses and r.get("responsePayload") is None:
                continue
            valid_records.append(r)
        self.stats.after_error_filter = len(valid_records)
        logger.info(
            f"[{self.task_def.task_name}] Error filter: "
            f"{len(task_records)} → {len(valid_records)}"
        )

        # Step 3: Extract samples
        samples = []
        for i, record in enumerate(valid_records):
            try:
                sample = self._extract_sample(record, i)
                if sample:
                    samples.append(sample)
            except Exception as e:
                self.stats.parse_errors += 1
                logger.debug(
                    f"[{self.task_def.task_name}] Parse error at index {i}: {e}"
                )

        if self.stats.parse_errors:
            logger.warning(
                f"[{self.task_def.task_name}] {self.stats.parse_errors} parse errors"
            )

        # Step 4: Deduplicate
        if self.global_config.deduplicate:
            seen_hashes = set()
            deduped = []
            for s in samples:
                h = content_hash(s["input"])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    deduped.append(s)
            logger.info(
                f"[{self.task_def.task_name}] Dedup: {len(samples)} → {len(deduped)}"
            )
            samples = deduped
        self.stats.after_dedup = len(samples)

        # Step 5: Sample
        rng = random.Random(self.global_config.random_seed)
        samples = self._sample(samples, rng)
        self.stats.final_samples = len(samples)

        # Step 6: Re-index IDs
        for i, s in enumerate(samples):
            s["id"] = f"{self.task_def.task_name}_{i + 1:04d}"

        logger.info(f"[{self.task_def.task_name}] Final: {len(samples)} samples")
        return samples

    def save(self, samples: List[Dict]) -> Path:
        """Save benchmark dataset to JSON file."""
        output_dir = Path(self.global_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.task_def.task_name}_{len(samples)}.json"
        output_path = output_dir / filename

        output = {
            "metadata": {
                "task_name": self.task_def.task_name,
                "task_type": self.task_def.task_type,
                "description": self.task_def.description,
                "source_file": self.task_def.source_file,
                "total_raw_records": self.stats.total_raw,
                "filtered_records": self.stats.after_task_filter,
                "valid_records": self.stats.after_error_filter,
                "deduped_records": self.stats.after_dedup,
                "sampled_records": len(samples),
                "sampling_strategy": self._effective_strategy(),
                "random_seed": self.global_config.random_seed,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "script": "convert_raw_to_benchmark.py",
            },
            "data": samples,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[{self.task_def.task_name}] Saved → {output_path} ({len(samples)} samples)"
        )
        return output_path


# ---------------------------------------------------------------------------
# HTKH Pre-processed Dataset Converter
# ---------------------------------------------------------------------------
# HTKH datasets ở /home/dev/Develop_2026/data/htkh_eval_datasets/ đã được
# tách sẵn theo nghiệp vụ với format EvalSample gần đầy đủ.
# Converter này đọc trực tiếp các file đó và chuẩn hoá sang benchmark format.

HTKH_TASK_NAMES = {
    "htkh_intent_classification",
    "htkh_intent_routing",
    "htkh_rag_qa",
    "htkh_response_evaluation",
}

# Vị trí mặc định của HTKH pre-processed files
HTKH_DATA_DIR = PROJECT_ROOT.parent / "data" / "htkh_eval_datasets"


class HTKHConverter:
    """
    Convert pre-processed HTKH eval datasets thành benchmark format chuẩn.

    Khác GenericConverter, các file này ĐÃ ở dạng EvalSample nên chỉ cần:
    - Đọc file JSON từ htkh_eval_datasets/
    - Chuẩn hoá các field đặc thù theo từng task
    - Áp dụng sampling nếu cần
    - Save với đầy đủ benchmark metadata
    """

    def __init__(
        self,
        task_def: TaskDefinition,
        global_config: GlobalConfig,
        htkh_data_dir: Optional[Path] = None,
    ):
        self.task_def = task_def
        self.global_config = global_config
        self.htkh_data_dir = htkh_data_dir or HTKH_DATA_DIR
        self.stats = ConversionStats()

    def _load_htkh_file(self) -> List[Dict]:
        """Load pre-processed HTKH JSON file."""
        # Source file có thể là tên file hoặc path đầy đủ
        source = self.task_def.source_file
        filepath = self.htkh_data_dir / source
        if not filepath.exists():
            raise FileNotFoundError(f"HTKH source file not found: {filepath}")

        logger.info(f"[{self.task_def.task_name}] Loading HTKH file: {filepath.name}")
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records = raw.get("data", raw if isinstance(raw, list) else [])
        logger.info(f"[{self.task_def.task_name}] Loaded {len(records)} records")
        return records

    def _normalize_sample(self, record: Dict, index: int) -> Optional[Dict]:
        """
        Chuẩn hoá 1 record HTKH sang benchmark EvalSample format.

        Xử lý đặc thù theo từng task_name.
        """
        task = self.task_def.task_name
        meta = record.get("metadata", {}) or {}

        # --- htkh_intent_classification ---
        # input: tin nhắn user, output: trống, không cần context
        if task == "htkh_intent_classification":
            input_text = (record.get("input") or "").strip()
            if not input_text:
                return None
            return {
                "id": "",
                "input": input_text,
                "output": record.get("output") or "",
                "reference": None,
                "context": None,
                "metadata": {
                    "task": meta.get("task", "intent_classification"),
                    "label_schema": meta.get("label_schema", []),
                    "is_error": meta.get("is_error", False),
                    "processing_ms": meta.get("processing_ms"),
                    "source": "htkh",
                },
            }

        # --- htkh_intent_routing ---
        # input: tin nhắn mới nhất, conversation_history: lịch sử hội thoại
        elif task == "htkh_intent_routing":
            input_text = (record.get("input") or "").strip()
            if not input_text:
                return None
            conv_history = record.get("conversation_history") or []
            return {
                "id": "",
                "input": input_text,
                "output": record.get("output") or "",
                "reference": None,
                "context": None,
                "conversation_history": conv_history,
                "metadata": {
                    "task": meta.get("task", "intent_routing"),
                    "is_error": meta.get("is_error", False),
                    "processing_ms": meta.get("processing_ms"),
                    "source": "htkh",
                },
            }

        # --- htkh_rag_qa ---
        # input: câu hỏi user (có thể trống, sẽ generate sau từ metadata.intent)
        # context: nội dung bài viết KB
        elif task == "htkh_rag_qa":
            context = (record.get("context") or "").strip()
            if not context:
                return None  # Bắt buộc phải có context
            input_text = (record.get("input") or "").strip()
            # input trống vẫn giữ, sẽ generate_ground_truth --step generate_input sau
            return {
                "id": "",
                "input": input_text,
                "output": record.get("output") or "",
                "reference": None,
                "context": context,
                "metadata": {
                    "task": meta.get("task", "rag_qa"),
                    "intent": meta.get("intent", ""),
                    "article_title": meta.get("article_title", ""),
                    "article_url": meta.get("article_url", ""),
                    "product": meta.get("product", ""),
                    "is_error": meta.get("is_error", False),
                    "processing_ms": meta.get("processing_ms"),
                    "source": "htkh",
                },
            }

        # --- htkh_response_evaluation ---
        # input: câu hỏi user, context: câu trả lời chatbot cần đánh giá
        elif task == "htkh_response_evaluation":
            input_text = (record.get("input") or "").strip()
            context = (record.get("context") or "").strip()
            if not input_text or not context:
                return None
            return {
                "id": "",
                "input": input_text,
                "output": record.get("output") or "",
                "reference": None,
                "context": context,
                "metadata": {
                    "task": meta.get("task", "response_evaluation"),
                    "eval_tasks": meta.get("eval_tasks", []),
                    "is_error": meta.get("is_error", False),
                    "processing_ms": meta.get("processing_ms"),
                    "source": "htkh",
                },
            }

        return None

    def convert(self) -> List[Dict]:
        """Run HTKH conversion pipeline."""
        records = self._load_htkh_file()

        self.stats.task_name = self.task_def.task_name
        self.stats.source_file = self.task_def.source_file
        self.stats.total_raw = len(records)
        self.stats.after_task_filter = len(records)  # No extra filtering needed

        # Filter errors if configured
        valid_records = []
        for r in records:
            if self.global_config.skip_errors and r.get("metadata", {}).get("is_error"):
                continue
            valid_records.append(r)
        self.stats.after_error_filter = len(valid_records)
        logger.info(
            f"[{self.task_def.task_name}] Error filter: "
            f"{len(records)} → {len(valid_records)}"
        )

        # Extract samples
        samples = []
        for i, record in enumerate(valid_records):
            try:
                sample = self._normalize_sample(record, i)
                if sample:
                    samples.append(sample)
            except Exception as e:
                self.stats.parse_errors += 1
                logger.debug(f"[{self.task_def.task_name}] Parse error at {i}: {e}")

        if self.stats.parse_errors:
            logger.warning(
                f"[{self.task_def.task_name}] {self.stats.parse_errors} parse errors"
            )

        # Deduplicate on input (skip if input trống, e.g. rag_qa)
        if self.global_config.deduplicate:
            seen_hashes: set = set()
            deduped = []
            for s in samples:
                inp = s.get("input") or ""
                ctx = s.get("context") or ""
                h = content_hash(inp + ctx)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    deduped.append(s)
            logger.info(
                f"[{self.task_def.task_name}] Dedup: {len(samples)} → {len(deduped)}"
            )
            samples = deduped
        self.stats.after_dedup = len(samples)

        # Sampling
        max_n = self.global_config.sample_size_override or self.task_def.max_samples
        if max_n and max_n > 0 and len(samples) > max_n:
            rng = random.Random(self.global_config.random_seed)
            samples = rng.sample(samples, max_n)
        self.stats.final_samples = len(samples)

        # Re-index IDs
        for i, s in enumerate(samples):
            s["id"] = f"{self.task_def.task_name}_{i + 1:04d}"

        logger.info(f"[{self.task_def.task_name}] Final: {len(samples)} samples")
        return samples

    def save(self, samples: List[Dict]) -> Path:
        """Save benchmark dataset to JSON file."""
        output_dir = Path(self.global_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.task_def.task_name}_{len(samples)}.json"
        output_path = output_dir / filename

        output = {
            "metadata": {
                "task_name": self.task_def.task_name,
                "task_type": self.task_def.task_type,
                "description": self.task_def.description,
                "source_file": self.task_def.source_file,
                "total_raw_records": self.stats.total_raw,
                "filtered_records": self.stats.after_task_filter,
                "valid_records": self.stats.after_error_filter,
                "deduped_records": self.stats.after_dedup,
                "sampled_records": len(samples),
                "sampling_strategy": self.global_config.sampling_strategy,
                "random_seed": self.global_config.random_seed,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "script": "convert_raw_to_benchmark.py (HTKH)",
            },
            "data": samples,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[{self.task_def.task_name}] Saved → {output_path} ({len(samples)} samples)"
        )
        return output_path


# ---------------------------------------------------------------------------
# Task Discovery
# ---------------------------------------------------------------------------
def discover_tasks(tasks_dir: Path) -> List[TaskDefinition]:
    """Load all YAML task definitions from the tasks/ directory."""
    if not tasks_dir.is_dir():
        logger.warning(f"Tasks directory not found: {tasks_dir}")
        return []

    definitions = []
    for yaml_file in sorted(tasks_dir.glob("*.yaml")) + sorted(tasks_dir.glob("*.yml")):
        # Skip non-task YAML files (e.g. language_mapping.yaml)
        if yaml_file.stem.startswith("_") or yaml_file.stem == "language_mapping":
            continue
        try:
            td = TaskDefinition.from_yaml(yaml_file)
            if not td.source_file:
                logger.debug(f"Skipping {yaml_file.name}: no source_file defined")
                continue
            definitions.append(td)
            logger.debug(f"Loaded task: {td.task_name} from {yaml_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {yaml_file.name}: {e}")

    return definitions


# ---------------------------------------------------------------------------
# Conversion Pipeline
# ---------------------------------------------------------------------------
class ConversionPipeline:
    """Orchestrate all converters."""

    def __init__(
        self,
        global_config: GlobalConfig,
        task_definitions: List[TaskDefinition],
        task_filter: Optional[List[str]] = None,
    ):
        self.global_config = global_config

        if task_filter:
            filtered = [td for td in task_definitions if td.task_name in task_filter]
            if not filtered:
                available = ", ".join(td.task_name for td in task_definitions)
                raise ValueError(
                    f"No matching tasks. Available: {available}"
                )
            task_definitions = filtered

        # Chọn converter phù hợp: HTKHConverter cho HTKH tasks, GenericConverter cho còn lại
        htkh_data_dir = Path(global_config.raw_data_dir) / "htkh_eval_datasets"
        self.converters = [
            HTKHConverter(td, global_config, htkh_data_dir=htkh_data_dir)
            if td.task_name in HTKH_TASK_NAMES
            else GenericConverter(td, global_config)
            for td in task_definitions
        ]

    def run(self) -> List[Tuple[str, Path, int]]:
        """Run all converters."""
        results = []
        for converter in self.converters:
            try:
                samples = converter.convert()
                if samples:
                    output_path = converter.save(samples)
                    results.append(
                        (converter.task_def.task_name, output_path, len(samples))
                    )
                else:
                    logger.warning(
                        f"[{converter.task_def.task_name}] No samples produced!"
                    )
            except FileNotFoundError as e:
                logger.error(
                    f"[{converter.task_def.task_name}] Source file not found: {e}"
                )
            except Exception as e:
                logger.error(
                    f"[{converter.task_def.task_name}] Conversion failed: {e}"
                )
                raise

        return results

    def print_summary(self, results: List[Tuple[str, Path, int]]):
        """Print conversion summary."""
        print("\n" + "=" * 70)
        print("  CONVERSION SUMMARY")
        print("=" * 70)

        for converter in self.converters:
            print(converter.stats.summary())
            print()

        print("-" * 70)
        print(f"  {'Task':<30} {'Samples':>8}  {'Output File'}")
        print("-" * 70)
        total = 0
        for task_name, output_path, count in results:
            print(f"  {task_name:<30} {count:>8}  {output_path.name}")
            total += count
        print("-" * 70)
        print(f"  {'TOTAL':<30} {total:>8}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw API logs to benchmark datasets (YAML-driven)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # Convert all tasks in tasks/
  %(prog)s --task crm_recommendation        # Convert specific task
  %(prog)s --task crm_recommendation --task makt_forecast
  %(prog)s --samples 300                    # Override all sample sizes
  %(prog)s --seed 123                       # Different random seed
  %(prog)s --strategy stratified            # Stratified sampling
  %(prog)s --list-tasks                     # List available tasks
  %(prog)s --tasks-dir /path/to/yamls       # Custom YAML directory
  %(prog)s --raw-dir /path/to/data          # Custom raw data directory
  %(prog)s --output-dir /path/to/output     # Custom output directory
        """,
    )
    parser.add_argument(
        "--task", action="append", dest="tasks",
        help="Task(s) to convert (can be repeated). Default: all.",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Override sample size for all tasks (0 = take all).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--strategy", choices=["random", "stratified"], default="random",
        help="Sampling strategy (default: random).",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable deduplication.",
    )
    parser.add_argument(
        "--raw-dir", default=None,
        help="Raw data directory path.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory path.",
    )
    parser.add_argument(
        "--tasks-dir", default=None,
        help="YAML task definitions directory.",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List available tasks and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Discover tasks
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else TASKS_DIR
    task_definitions = discover_tasks(tasks_dir)
    if not task_definitions:
        logger.error(f"No task YAML files found in: {tasks_dir}")
        logger.info("Create YAML files in scripts/tasks/ — see existing ones as examples.")
        sys.exit(1)

    # List tasks
    if args.list_tasks:
        print(f"\nAvailable tasks (from {tasks_dir}):")
        print("-" * 75)
        for td in task_definitions:
            max_label = str(td.max_samples) if td.max_samples > 0 else "all"
            strat = td.sampling_strategy or "global"
            print(
                f"  {td.task_name:<30} type={td.task_type:<15} "
                f"max={max_label:<5} src={td.source_file}"
            )
            if td.description:
                print(f"    {td.description}")
        print()
        return

    # Build global config
    config = GlobalConfig()
    config.random_seed = args.seed
    config.sampling_strategy = args.strategy
    config.deduplicate = not args.no_dedup
    config.sample_size_override = args.samples

    if args.raw_dir:
        config.raw_data_dir = args.raw_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    # Run
    logger.info("Starting conversion pipeline...")
    logger.info(f"  Tasks dir:     {tasks_dir}")
    logger.info(f"  Raw data dir:  {config.raw_data_dir}")
    logger.info(f"  Output dir:    {config.output_dir}")
    logger.info(f"  Seed:          {config.random_seed}")
    logger.info(f"  Strategy:      {config.sampling_strategy}")
    logger.info(f"  Deduplicate:   {config.deduplicate}")
    logger.info(f"  Tasks found:   {len(task_definitions)}")

    pipeline = ConversionPipeline(config, task_definitions, task_filter=args.tasks)
    results = pipeline.run()
    pipeline.print_summary(results)


if __name__ == "__main__":
    main()
