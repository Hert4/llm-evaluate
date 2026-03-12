#!/usr/bin/env python3
"""
Generate model outputs for benchmark datasets.

Takes existing benchmark files (with input + reference), calls a target model
API to generate fresh outputs, saves results for evaluation.

This fills the missing step in the pipeline:
  convert_raw → generate_ground_truth → **generate_model_output** → run_evaluation

Usage:
    python scripts/generate_model_output.py --model misa-ai-1.0
    python scripts/generate_model_output.py --model misa-ai-1.0 --task crm_recommendation
    python scripts/generate_model_output.py --model misa-ai-1.0 --dry-run --limit 3
    python scripts/generate_model_output.py --list-tasks
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
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
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"

# ---------------------------------------------------------------------------
# API Configuration (same endpoint as other scripts)
# ---------------------------------------------------------------------------
API_URL = "https://numerous-catch-uploaded-compile.trycloudflare.com/v1/chat/completions"
API_KEY = "misa_misa_00t07fh7_ZFRMf6rOUaVHTv6CZH0uOzAx_LDP1IeWM"

# Defaults
DEFAULT_CONCURRENCY = 5
DEFAULT_MAX_COMPLETION_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 120
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2
CHECKPOINT_INTERVAL = 20

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Task discovery (reuse from other scripts)
# ---------------------------------------------------------------------------
def discover_tasks(tasks_dir: Path) -> List[Dict]:
    """Load all task YAMLs that have gt_metrics (evaluatable tasks)."""
    if not tasks_dir.is_dir():
        return []
    configs = []
    for yaml_file in sorted(tasks_dir.glob("*.yaml")) + sorted(tasks_dir.glob("*.yml")):
        if yaml_file.stem.startswith("_") or yaml_file.stem == "language_mapping":
            continue
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if raw and raw.get("gt_metrics"):
                configs.append({
                    "task_name": raw.get("task_name", yaml_file.stem),
                    "description": raw.get("description", ""),
                    "gt_metrics": raw.get("gt_metrics", []),
                })
        except Exception as e:
            logger.warning(f"Failed to load {yaml_file.name}: {e}")
    return configs


def find_benchmark_file(task_name: str, benchmarks_dir: Path) -> Optional[Path]:
    """Find benchmark JSON file, excluding backups."""
    matches = [
        p for p in benchmarks_dir.glob(f"{task_name}_*.json")
        if "_backup" not in p.stem
    ]
    return sorted(matches)[-1] if matches else None


# ---------------------------------------------------------------------------
# API calling with retry (adapted from generate_ground_truth.py)
# ---------------------------------------------------------------------------
async def call_model_api(
    session: aiohttp.ClientSession,
    model: str,
    system_prompt: Optional[str],
    user_message: str,
    semaphore: asyncio.Semaphore,
    sample_id: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> Dict[str, Any]:
    """
    Call target model API with the original system prompt + user message.

    Returns: {"success": bool, "content": str, "error": str|None}
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
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
                            return {"success": False, "content": "", "error": "Empty response"}

                    elif resp.status == 429:
                        wait = RETRY_DELAY_BASE ** attempt
                        logger.warning(f"[{sample_id}] Rate limited, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                        await asyncio.sleep(wait)
                        continue

                    elif resp.status >= 500:
                        wait = RETRY_DELAY_BASE ** attempt
                        logger.warning(f"[{sample_id}] Server error {resp.status}, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                        await asyncio.sleep(wait)
                        continue

                    else:
                        body = await resp.text()
                        return {"success": False, "content": "", "error": f"HTTP {resp.status}: {body[:500]}"}

            except asyncio.TimeoutError:
                wait = RETRY_DELAY_BASE ** attempt
                logger.warning(f"[{sample_id}] Timeout, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            except aiohttp.ClientError as e:
                wait = RETRY_DELAY_BASE ** attempt
                logger.warning(f"[{sample_id}] Connection error: {e}, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                await asyncio.sleep(wait)

        return {"success": False, "content": "", "error": f"Failed after {MAX_RETRIES} retries"}


async def _wrap_with_id(sample_id: str, coro):
    """Wrap a coroutine to return (id, result)."""
    result = await coro
    return (sample_id, result)


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------
async def generate_outputs_for_task(
    task_name: str,
    model: str,
    benchmarks_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    dry_run: bool = False,
    limit: Optional[int] = None,
    force: bool = False,
) -> Optional[Dict]:
    """
    Generate model outputs for a task.

    1. Load benchmark (input + reference from existing file)
    2. Create a copy with model outputs replaced
    3. Call target model API for each sample
    4. Save new benchmark file

    Returns stats dict or None on failure.
    """
    # Find source benchmark
    benchmark_path = find_benchmark_file(task_name, benchmarks_dir)
    if not benchmark_path:
        logger.error(f"[{task_name}] No benchmark file found")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"Task:  {task_name}")
    logger.info(f"Model: {model}")
    logger.info(f"File:  {benchmark_path.name}")
    logger.info(f"{'='*60}")

    # Load benchmark
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    samples = benchmark_data.get("data", [])
    total = len(samples)

    # Check which samples need generation
    # If force=True, regenerate all. Otherwise skip samples already from this model.
    to_process = []
    for s in samples:
        if not force and s.get("metadata", {}).get("model") == model:
            continue  # Already has output from this model
        to_process.append(s)

    if not to_process:
        to_process = samples  # First run: generate all

    if limit:
        to_process = to_process[:limit]

    logger.info(f"Total samples: {total}, to generate: {len(to_process)}")

    if dry_run:
        logger.info("=== DRY RUN — showing what would be sent ===")
        for s in to_process[:3]:
            print(f"\n{'─'*60}")
            print(f"Sample: {s['id']}")
            print(f"System: {str(s.get('context', ''))[:200]}...")
            print(f"User:   {str(s.get('input', ''))[:200]}...")
        if len(to_process) > 3:
            print(f"\n... and {len(to_process) - 3} more samples")
        return {"task_name": task_name, "total": total, "generated": 0, "dry_run": True}

    # Generate outputs
    semaphore = asyncio.Semaphore(concurrency)
    results_map: Dict[str, str] = {}
    success_count = 0
    fail_count = 0

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(to_process), desc=f"Generating ({model})", unit="sample")

        # Process in batches for checkpointing
        for batch_start in range(0, len(to_process), CHECKPOINT_INTERVAL):
            batch = to_process[batch_start:batch_start + CHECKPOINT_INTERVAL]

            coros = []
            for s in batch:
                coro = call_model_api(
                    session=session,
                    model=model,
                    system_prompt=s.get("context"),
                    user_message=s.get("input", ""),
                    semaphore=semaphore,
                    sample_id=s["id"],
                )
                coros.append((s["id"], coro))

            for future in asyncio.as_completed([
                _wrap_with_id(sid, coro) for sid, coro in coros
            ]):
                sid, res = await future
                if res["success"]:
                    results_map[sid] = res["content"]
                    success_count += 1
                else:
                    logger.error(f"[{sid}] Failed: {res['error']}")
                    fail_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=success_count, fail=fail_count)

        pbar.close()

    elapsed = time.time() - start_time

    # Build new benchmark with updated outputs
    new_samples = []
    for s in samples:
        new_s = dict(s)
        if s["id"] in results_map:
            new_s["output"] = results_map[s["id"]]
            new_s.setdefault("metadata", {})
            new_s["metadata"]["model"] = model
            new_s["metadata"]["generated_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_samples.append(new_s)

    # Save new benchmark file for this model
    model_benchmarks_dir = benchmarks_dir / model
    model_benchmarks_dir.mkdir(parents=True, exist_ok=True)

    output_path = model_benchmarks_dir / benchmark_path.name
    output_data = {
        "metadata": {
            **benchmark_data.get("metadata", {}),
            "model": model,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generated_from": benchmark_path.name,
            "script": "generate_model_output.py",
        },
        "data": new_samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_name}] Saved → {output_path}")
    logger.info(f"[{task_name}] {success_count} success, {fail_count} failed, {elapsed:.1f}s")

    return {
        "task_name": task_name,
        "total": total,
        "generated": success_count,
        "failed": fail_count,
        "elapsed": round(elapsed, 1),
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Evaluation (reuse run_evaluation.py logic inline)
# ---------------------------------------------------------------------------
def run_eval_for_model(model: str, tasks: List[str], benchmarks_dir: Path):
    """Run evaluation for a model using its generated benchmark files."""
    import subprocess

    model_benchmarks_dir = benchmarks_dir / model
    if not model_benchmarks_dir.is_dir():
        logger.error(f"No benchmark files found for model {model}")
        return

    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_evaluation.py"),
        "--model", model,
        "--benchmarks-dir", str(model_benchmarks_dir),
    ]
    for task in tasks:
        cmd.extend(["--task", task])

    logger.info(f"\n{'='*60}")
    logger.info(f"Running evaluation for {model}...")
    logger.info(f"{'='*60}")

    subprocess.run(cmd, check=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model outputs and optionally evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model misa-ai-1.0                          # Generate + evaluate all tasks
  %(prog)s --model misa-ai-1.0 --task crm_recommendation  # Single task
  %(prog)s --model misa-ai-1.0 --generate-only           # Generate without eval
  %(prog)s --model misa-ai-1.0 --eval-only               # Eval only (already generated)
  %(prog)s --model misa-ai-1.0 --dry-run --limit 3       # Preview
  %(prog)s --list-tasks                                    # Show available tasks
        """,
    )
    parser.add_argument(
        "--model", required=True,
        help="Target model name to generate outputs with (e.g., misa-ai-1.0).",
    )
    parser.add_argument(
        "--task", action="append", dest="tasks",
        help="Task(s) to process (can repeat). Default: all.",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate outputs, skip evaluation.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation (outputs must already exist).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be sent without calling API.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N samples per task.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent API calls (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regenerate even if outputs exist.",
    )
    parser.add_argument(
        "--benchmarks-dir", default=None,
        help="Source benchmarks directory.",
    )
    parser.add_argument(
        "--tasks-dir", default=None,
        help="YAML tasks directory.",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List available tasks and exit.",
    )
    return parser.parse_args()


async def async_main():
    args = parse_args()

    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else TASKS_DIR
    benchmarks_dir = Path(args.benchmarks_dir) if args.benchmarks_dir else BENCHMARKS_DIR

    # Discover tasks
    all_tasks = discover_tasks(tasks_dir)
    if not all_tasks:
        logger.error(f"No evaluatable tasks found in {tasks_dir}")
        sys.exit(1)

    # List tasks
    if args.list_tasks:
        print(f"\nAvailable tasks (from {tasks_dir}):")
        print("-" * 70)
        for t in all_tasks:
            benchmark = find_benchmark_file(t["task_name"], benchmarks_dir)
            bname = benchmark.name if benchmark else "(not found)"
            print(f"  {t['task_name']:<30} metrics={t['gt_metrics']}")
            print(f"    Benchmark: {bname}")
            if t["description"]:
                print(f"    {t['description']}")
        print()
        return

    # Filter tasks
    task_names = [t["task_name"] for t in all_tasks]
    if args.tasks:
        missing = [t for t in args.tasks if t not in task_names]
        if missing:
            logger.error(f"Unknown task(s): {missing}. Available: {task_names}")
            sys.exit(1)
        task_names = args.tasks

    model = args.model
    logger.info(f"Model:  {model}")
    logger.info(f"Tasks:  {task_names}")

    # Step 1: Generate outputs
    if not args.eval_only:
        all_stats = []
        for task_name in task_names:
            stats = await generate_outputs_for_task(
                task_name=task_name,
                model=model,
                benchmarks_dir=benchmarks_dir,
                concurrency=args.concurrency,
                dry_run=args.dry_run,
                limit=args.limit,
                force=args.force,
            )
            if stats:
                all_stats.append(stats)

        # Print generation summary
        if all_stats:
            print(f"\n{'='*70}")
            print(f"  OUTPUT GENERATION SUMMARY — Model: {model}")
            print(f"{'='*70}")
            for s in all_stats:
                status = "DRY RUN" if s.get("dry_run") else f"{s['generated']} ok, {s.get('failed', 0)} fail"
                print(f"  {s['task_name']:<30} {s['total']} samples  → {status}")
            print(f"{'='*70}\n")

    # Step 2: Evaluate
    if not args.generate_only and not args.dry_run:
        run_eval_for_model(model, task_names, benchmarks_dir)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
