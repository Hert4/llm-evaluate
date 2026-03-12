#!/usr/bin/env python3
"""
Generate Ground Truth (reference) for benchmark datasets using GPT-5.2 API.

Reads benchmark JSON files (already containing input + output), calls GPT-5.2
to generate reference answers, and writes them back into the benchmark files.

YAML-driven: each task YAML defines gt_prompt, gt_task_type, and gt_metrics.

Usage:
    python scripts/generate_ground_truth.py
    python scripts/generate_ground_truth.py --task crm_recommendation
    python scripts/generate_ground_truth.py --task crm_recommendation --dry-run --limit 3
    python scripts/generate_ground_truth.py --list-tasks
    python scripts/generate_ground_truth.py --concurrency 3
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    print("aiohttp is required: pip install aiohttp")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml")
    sys.exit(1)

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm as atqdm
except ImportError:
    print("tqdm is required: pip install tqdm")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TASKS_DIR = SCRIPT_DIR / "tasks"
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
API_URL = "https://numerous-catch-uploaded-compile.trycloudflare.com/v1/chat/completions"
API_KEY = "misa_misa_00t07fh7_ZFRMf6rOUaVHTv6CZH0uOzAx_LDP1IeWM"
MODEL = "gpt-5.2"

# Defaults
DEFAULT_CONCURRENCY = 5
DEFAULT_MAX_COMPLETION_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 120  # seconds per request
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2  # exponential backoff base
CHECKPOINT_INTERVAL = 20  # save after every N samples

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GTTaskConfig:
    """Ground truth generation config extracted from task YAML."""

    task_name: str = ""
    task_type: str = "qa"
    description: str = ""

    # GT-specific fields
    gt_prompt: str = ""          # Prompt template with {input} and {context}
    gt_task_type: str = "qa"     # qa | translation | rag
    gt_metrics: List[str] = field(default_factory=list)  # e.g. [exact_match, token_f1]

    # Benchmark file info
    benchmark_file: str = ""     # e.g. crm_recommendation_150.json

    @classmethod
    def from_yaml(cls, filepath: Path) -> "GTTaskConfig":
        """Load GT config from a task YAML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            raise ValueError(f"Empty YAML: {filepath}")

        cfg = cls()
        cfg.task_name = raw.get("task_name", filepath.stem)
        cfg.task_type = raw.get("task_type", "qa")
        cfg.description = raw.get("description", "")
        cfg.gt_prompt = raw.get("gt_prompt", "")
        cfg.gt_task_type = raw.get("gt_task_type", raw.get("task_type", "qa"))
        cfg.gt_metrics = raw.get("gt_metrics", [])
        return cfg

    @property
    def has_gt_config(self) -> bool:
        """Check if this task has GT generation configured."""
        return bool(self.gt_prompt)


@dataclass
class GenerationStats:
    """Track generation statistics."""

    task_name: str = ""
    total_samples: int = 0
    already_have_reference: int = 0
    to_generate: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def success_rate(self) -> float:
        if self.to_generate == 0:
            return 100.0
        return (self.success / self.to_generate) * 100

    def summary(self) -> str:
        return (
            f"  {self.task_name}:\n"
            f"    Total samples:          {self.total_samples:>6}\n"
            f"    Already have reference: {self.already_have_reference:>6}\n"
            f"    To generate:            {self.to_generate:>6}\n"
            f"    Success:                {self.success:>6}\n"
            f"    Failed:                 {self.failed:>6}\n"
            f"    Skipped (dry-run):      {self.skipped:>6}\n"
            f"    Success rate:           {self.success_rate:.1f}%"
        )


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------
def discover_gt_tasks(tasks_dir: Path) -> List[GTTaskConfig]:
    """Load all task YAMLs that have GT config."""
    if not tasks_dir.is_dir():
        logger.warning(f"Tasks directory not found: {tasks_dir}")
        return []

    configs = []
    for yaml_file in sorted(tasks_dir.glob("*.yaml")) + sorted(tasks_dir.glob("*.yml")):
        if yaml_file.stem.startswith("_") or yaml_file.stem == "language_mapping":
            continue
        try:
            cfg = GTTaskConfig.from_yaml(yaml_file)
            if cfg.has_gt_config:
                configs.append(cfg)
            else:
                logger.debug(f"Skipping {yaml_file.name}: no gt_prompt defined")
        except Exception as e:
            logger.warning(f"Failed to load {yaml_file.name}: {e}")

    return configs


def find_benchmark_file(task_name: str, benchmarks_dir: Path) -> Optional[Path]:
    """Find the benchmark JSON file for a given task."""
    # Look for files matching pattern: {task_name}_{count}.json
    # Exclude backup files
    matches = [
        p for p in benchmarks_dir.glob(f"{task_name}_*.json")
        if "_backup" not in p.stem
    ]
    if not matches:
        return None
    # If multiple matches, pick the one with the highest count
    return sorted(matches)[-1]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(sample: Dict, gt_prompt: str) -> str:
    """
    Build the API prompt for a single sample.

    Supports {input} and {context} placeholders in gt_prompt.
    """
    input_text = sample.get("input", "")
    context_text = sample.get("context", "") or ""

    try:
        prompt = gt_prompt.format(input=input_text, context=context_text)
    except KeyError as e:
        # If there are extra placeholders, just use input replacement
        logger.warning(f"Prompt template has unknown placeholder {e}, using raw substitution")
        prompt = gt_prompt.replace("{input}", input_text).replace("{context}", context_text)

    return prompt


# ---------------------------------------------------------------------------
# API calling with retry
# ---------------------------------------------------------------------------
async def call_api(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    sample_id: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> Dict[str, Any]:
    """
    Call GPT-5.2 API with retry logic.

    Returns: {"success": bool, "content": str, "error": str|None}
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if content:
                            return {"success": True, "content": content.strip(), "error": None}
                        else:
                            return {
                                "success": False,
                                "content": "",
                                "error": "Empty response content",
                            }

                    elif resp.status == 429:
                        # Rate limited — wait and retry
                        wait = RETRY_DELAY_BASE ** attempt
                        logger.warning(
                            f"[{sample_id}] Rate limited (429), waiting {wait}s "
                            f"(attempt {attempt}/{MAX_RETRIES})"
                        )
                        await asyncio.sleep(wait)
                        continue

                    elif resp.status >= 500:
                        # Server error — retry
                        wait = RETRY_DELAY_BASE ** attempt
                        body = await resp.text()
                        logger.warning(
                            f"[{sample_id}] Server error {resp.status}, waiting {wait}s "
                            f"(attempt {attempt}/{MAX_RETRIES}): {body[:200]}"
                        )
                        await asyncio.sleep(wait)
                        continue

                    else:
                        body = await resp.text()
                        return {
                            "success": False,
                            "content": "",
                            "error": f"HTTP {resp.status}: {body[:500]}",
                        }

            except asyncio.TimeoutError:
                wait = RETRY_DELAY_BASE ** attempt
                logger.warning(
                    f"[{sample_id}] Timeout, waiting {wait}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                await asyncio.sleep(wait)
                continue

            except aiohttp.ClientError as e:
                wait = RETRY_DELAY_BASE ** attempt
                logger.warning(
                    f"[{sample_id}] Connection error: {e}, waiting {wait}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                await asyncio.sleep(wait)
                continue

        return {
            "success": False,
            "content": "",
            "error": f"Failed after {MAX_RETRIES} retries",
        }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------
async def generate_references(
    samples: List[Dict],
    gt_prompt: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Generate references for a list of samples.

    Args:
        samples: List of sample dicts (from benchmark JSON)
        gt_prompt: Prompt template with {input}/{context} placeholders
        concurrency: Max concurrent API calls
        dry_run: If True, don't actually call API — just show prompts
        limit: Only process first N samples (for testing)

    Returns:
        Updated list of samples with reference field filled
    """
    # Filter: only generate for samples without reference
    to_process = []
    for s in samples:
        if s.get("reference"):
            continue
        to_process.append(s)

    if limit:
        to_process = to_process[:limit]

    if not to_process:
        logger.info("All samples already have references. Nothing to do.")
        return samples

    logger.info(f"Generating references for {len(to_process)} samples...")

    if dry_run:
        logger.info("=== DRY RUN MODE — showing prompts without calling API ===")
        for s in to_process:
            prompt = build_prompt(s, gt_prompt)
            print(f"\n{'='*60}")
            print(f"Sample: {s['id']}")
            print(f"{'='*60}")
            print(f"Prompt ({len(prompt)} chars):")
            print(prompt[:1000])
            if len(prompt) > 1000:
                print(f"... [{len(prompt) - 1000} more chars]")
            print(f"\nCurrent output ({len(s.get('output', ''))} chars):")
            print(str(s.get("output", ""))[:300])
            print()
        return samples

    # Real API calls
    semaphore = asyncio.Semaphore(concurrency)
    results_map: Dict[str, str] = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for s in to_process:
            prompt = build_prompt(s, gt_prompt)
            task = call_api(session, prompt, semaphore, s["id"])
            tasks.append((s["id"], task))

        # Process with progress bar
        pbar = tqdm(total=len(tasks), desc="Generating GT", unit="sample")
        success_count = 0
        fail_count = 0

        # Process in order but with concurrent execution
        coroutines = [t[1] for t in tasks]
        ids = [t[0] for t in tasks]

        for i, result in enumerate(
            asyncio.as_completed([
                _wrap_with_id(sid, coro)
                for sid, coro in zip(ids, coroutines)
            ])
        ):
            sid, res = await result
            if res["success"]:
                results_map[sid] = res["content"]
                success_count += 1
            else:
                logger.error(f"[{sid}] Failed: {res['error']}")
                fail_count += 1
            pbar.update(1)
            pbar.set_postfix(ok=success_count, fail=fail_count)

        pbar.close()

    # Update samples with results
    for s in samples:
        if s["id"] in results_map:
            s["reference"] = results_map[s["id"]]

    logger.info(f"Generation complete: {success_count} success, {fail_count} failed")
    return samples


async def _wrap_with_id(sample_id: str, coro):
    """Wrap a coroutine to return (id, result) tuple."""
    result = await coro
    return (sample_id, result)


async def generate_with_checkpoints(
    benchmark_data: Dict,
    benchmark_path: Path,
    gt_prompt: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> Dict:
    """
    Generate references with checkpoint saving.

    Saves progress every CHECKPOINT_INTERVAL samples to avoid losing work.
    """
    samples = benchmark_data["data"]

    # Filter: only generate for samples without reference
    to_process = [s for s in samples if not s.get("reference")]
    if limit:
        to_process = to_process[:limit]

    if not to_process:
        logger.info("All samples already have references. Nothing to do.")
        return benchmark_data

    if dry_run:
        logger.info("=== DRY RUN MODE — showing prompts without calling API ===")
        for s in to_process:
            prompt = build_prompt(s, gt_prompt)
            print(f"\n{'='*60}")
            print(f"Sample: {s['id']}")
            print(f"{'='*60}")
            print(f"Prompt ({len(prompt)} chars):")
            print(prompt[:1000])
            if len(prompt) > 1000:
                print(f"... [{len(prompt) - 1000} more chars]")
            print(f"\nCurrent output ({len(s.get('output', ''))} chars):")
            print(str(s.get("output", ""))[:300])
            print()
        return benchmark_data

    logger.info(f"Generating references for {len(to_process)} samples (checkpoint every {CHECKPOINT_INTERVAL})...")

    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0
    fail_count = 0
    processed_count = 0

    # Build ID→sample index map for fast lookup
    id_to_idx = {s["id"]: i for i, s in enumerate(samples)}

    async with aiohttp.ClientSession() as session:
        # Process in batches for checkpointing
        batches = [
            to_process[i:i + CHECKPOINT_INTERVAL]
            for i in range(0, len(to_process), CHECKPOINT_INTERVAL)
        ]

        pbar = tqdm(total=len(to_process), desc="Generating GT", unit="sample")

        for batch_idx, batch in enumerate(batches):
            # Create tasks for this batch
            batch_coros = []
            for s in batch:
                prompt = build_prompt(s, gt_prompt)
                coro = call_api(session, prompt, semaphore, s["id"])
                batch_coros.append((s["id"], coro))

            # Execute batch concurrently
            for future in asyncio.as_completed([
                _wrap_with_id(sid, coro)
                for sid, coro in batch_coros
            ]):
                sid, res = await future
                if res["success"]:
                    # Update sample in-place
                    idx = id_to_idx[sid]
                    samples[idx]["reference"] = res["content"]
                    success_count += 1
                else:
                    logger.error(f"[{sid}] Failed: {res['error']}")
                    fail_count += 1

                processed_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=success_count, fail=fail_count)

            # Checkpoint: save after each batch
            if not dry_run and batch_idx < len(batches) - 1:
                _save_checkpoint(benchmark_data, benchmark_path)
                logger.info(
                    f"  Checkpoint saved ({processed_count}/{len(to_process)} processed)"
                )

        pbar.close()

    logger.info(f"Generation complete: {success_count} success, {fail_count} failed")
    return benchmark_data


def _save_checkpoint(benchmark_data: Dict, benchmark_path: Path):
    """Save current state as checkpoint."""
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def load_benchmark(filepath: Path) -> Dict:
    """Load a benchmark JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def backup_benchmark(filepath: Path) -> Path:
    """Create a backup of the benchmark file before modification."""
    backup_path = filepath.with_name(
        filepath.stem + "_backup" + filepath.suffix
    )
    shutil.copy2(filepath, backup_path)
    logger.info(f"Backup created: {backup_path.name}")
    return backup_path


def save_benchmark(data: Dict, filepath: Path):
    """Save benchmark data back to JSON."""
    # Update metadata
    data.setdefault("metadata", {})
    data["metadata"]["gt_generated_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["metadata"]["gt_model"] = MODEL
    data["metadata"]["gt_script"] = "generate_ground_truth.py"

    # Count references
    total = len(data.get("data", []))
    with_ref = sum(1 for s in data.get("data", []) if s.get("reference"))
    data["metadata"]["samples_with_reference"] = with_ref
    data["metadata"]["samples_total"] = total

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved: {filepath.name} ({with_ref}/{total} samples have reference)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def process_task(
    task_config: GTTaskConfig,
    benchmarks_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_backup: bool = False,
) -> Optional[GenerationStats]:
    """Process a single task: load benchmark → generate GT → save."""
    stats = GenerationStats(task_name=task_config.task_name)

    # Find benchmark file
    benchmark_path = find_benchmark_file(task_config.task_name, benchmarks_dir)
    if not benchmark_path:
        logger.error(f"[{task_config.task_name}] No benchmark file found in {benchmarks_dir}")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"Task: {task_config.task_name}")
    logger.info(f"File: {benchmark_path.name}")
    logger.info(f"GT type: {task_config.gt_task_type}")
    logger.info(f"Metrics: {', '.join(task_config.gt_metrics)}")
    logger.info(f"{'='*60}")

    # Load benchmark
    benchmark_data = load_benchmark(benchmark_path)
    samples = benchmark_data.get("data", [])
    stats.total_samples = len(samples)
    stats.already_have_reference = sum(1 for s in samples if s.get("reference"))
    stats.to_generate = stats.total_samples - stats.already_have_reference

    if limit:
        stats.to_generate = min(stats.to_generate, limit)

    logger.info(
        f"Samples: {stats.total_samples} total, "
        f"{stats.already_have_reference} already have reference, "
        f"{stats.to_generate} to generate"
    )

    if stats.to_generate == 0:
        logger.info(f"[{task_config.task_name}] All samples already have references!")
        return stats

    # Backup
    if not dry_run and not skip_backup:
        backup_benchmark(benchmark_path)

    # Generate
    start_time = time.time()

    if dry_run:
        stats.skipped = stats.to_generate
        # Just show dry-run info
        to_show = [s for s in samples if not s.get("reference")]
        if limit:
            to_show = to_show[:limit]

        logger.info("=== DRY RUN MODE ===")
        for s in to_show:
            prompt = build_prompt(s, task_config.gt_prompt)
            print(f"\n{'─'*60}")
            print(f"Sample: {s['id']}")
            print(f"{'─'*60}")
            print(f"Prompt ({len(prompt)} chars):")
            print(prompt[:1500])
            if len(prompt) > 1500:
                print(f"... [{len(prompt) - 1500} more chars]")
            print(f"\nCurrent output ({len(s.get('output', ''))} chars):")
            output_str = str(s.get("output", ""))
            print(output_str[:500])
            if len(output_str) > 500:
                print(f"... [{len(output_str) - 500} more chars]")
    else:
        benchmark_data = await generate_with_checkpoints(
            benchmark_data=benchmark_data,
            benchmark_path=benchmark_path,
            gt_prompt=task_config.gt_prompt,
            concurrency=concurrency,
            dry_run=False,
            limit=limit,
        )

        # Count results
        new_refs = sum(1 for s in benchmark_data["data"] if s.get("reference"))
        stats.success = new_refs - stats.already_have_reference
        stats.failed = stats.to_generate - stats.success

        # Update GT metrics in metadata
        benchmark_data["metadata"]["gt_task_type"] = task_config.gt_task_type
        benchmark_data["metadata"]["gt_metrics"] = task_config.gt_metrics

        # Save
        save_benchmark(benchmark_data, benchmark_path)

    elapsed = time.time() - start_time
    logger.info(
        f"[{task_config.task_name}] Done in {elapsed:.1f}s"
    )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ground truth references for benchmark datasets using GPT-5.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Generate GT for all configured tasks
  %(prog)s --task crm_recommendation                # Single task
  %(prog)s --task crm_recommendation --dry-run --limit 3   # Preview prompts
  %(prog)s --concurrency 3                          # Lower concurrency
  %(prog)s --list-tasks                             # Show tasks with GT config
  %(prog)s --skip-backup                            # Don't backup before overwriting
        """,
    )
    parser.add_argument(
        "--task", action="append", dest="tasks",
        help="Task(s) to process (can be repeated). Default: all configured.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show prompts without calling API.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N samples per task (for testing).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent API calls (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--benchmarks-dir", default=None,
        help=f"Benchmarks directory (default: {BENCHMARKS_DIR}).",
    )
    parser.add_argument(
        "--tasks-dir", default=None,
        help=f"YAML task definitions directory (default: {TASKS_DIR}).",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List tasks with GT configuration and exit.",
    )
    parser.add_argument(
        "--skip-backup", action="store_true",
        help="Skip backup before overwriting benchmark files.",
    )
    return parser.parse_args()


async def async_main():
    args = parse_args()

    # Directories
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else TASKS_DIR
    benchmarks_dir = Path(args.benchmarks_dir) if args.benchmarks_dir else BENCHMARKS_DIR

    # Discover tasks
    gt_tasks = discover_gt_tasks(tasks_dir)
    if not gt_tasks:
        logger.error(f"No tasks with GT config found in {tasks_dir}")
        logger.info("Add gt_prompt, gt_task_type, gt_metrics to your task YAML files.")
        sys.exit(1)

    # List tasks
    if args.list_tasks:
        print(f"\nTasks with GT configuration (from {tasks_dir}):")
        print("-" * 80)
        for cfg in gt_tasks:
            benchmark = find_benchmark_file(cfg.task_name, benchmarks_dir)
            benchmark_name = benchmark.name if benchmark else "(not found)"
            print(f"  {cfg.task_name:<30} type={cfg.gt_task_type:<15} metrics={cfg.gt_metrics}")
            print(f"    Benchmark: {benchmark_name}")
            if cfg.description:
                print(f"    {cfg.description}")
        print()
        return

    # Filter tasks
    if args.tasks:
        filtered = [cfg for cfg in gt_tasks if cfg.task_name in args.tasks]
        if not filtered:
            available = ", ".join(cfg.task_name for cfg in gt_tasks)
            logger.error(f"No matching tasks. Available: {available}")
            sys.exit(1)
        gt_tasks = filtered

    # Process each task
    logger.info(f"Processing {len(gt_tasks)} task(s)...")
    logger.info(f"  Benchmarks dir: {benchmarks_dir}")
    logger.info(f"  Concurrency:    {args.concurrency}")
    logger.info(f"  Dry run:        {args.dry_run}")
    if args.limit:
        logger.info(f"  Limit:          {args.limit} samples/task")

    all_stats = []
    for task_config in gt_tasks:
        stats = await process_task(
            task_config=task_config,
            benchmarks_dir=benchmarks_dir,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
            limit=args.limit,
            skip_backup=args.skip_backup,
        )
        if stats:
            all_stats.append(stats)

    # Print summary
    if all_stats:
        print(f"\n{'='*70}")
        print("  GROUND TRUTH GENERATION SUMMARY")
        print(f"{'='*70}")
        for stats in all_stats:
            print(stats.summary())
            print()

        total_success = sum(s.success for s in all_stats)
        total_failed = sum(s.failed for s in all_stats)
        total_skipped = sum(s.skipped for s in all_stats)
        total_samples = sum(s.total_samples for s in all_stats)

        print(f"{'─'*70}")
        print(f"  Total: {total_samples} samples, "
              f"{total_success} generated, "
              f"{total_failed} failed, "
              f"{total_skipped} skipped")
        print(f"{'='*70}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
