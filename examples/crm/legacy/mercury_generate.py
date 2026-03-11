"""
STEP 2: Generate output tu mercury-2 (Inception Labs API)
=========================================================================
Chay o MAY NGOAI (khong bi firewall chan):
    pip install openai
    python3 mercury_generate.py

Input:
    mercury_requests/crmkh_requests.json
    mercury_requests/crmmisa_requests.json

Output:
    mercury_outputs/crmkh_outputs.json
    mercury_outputs/crmmisa_outputs.json

Sau do copy folder mercury_outputs/ ve may noi bo de chay evaluate.
"""
import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ============================================================================
# CONFIG
# ============================================================================

INCEPTION_API_BASE_URL = "https://api.inceptionlabs.ai/v1"
INCEPTION_API_KEY = "sk_d494791d6cdeccb8803b6addc0675c65"
MODEL = "mercury-2"
MAX_CONCURRENT = 5
MAX_TOKENS = 2048
TEMPERATURE = 0.0
TIMEOUT = 120
MAX_RETRIES = 3

INPUT_DIR = Path(__file__).parent / "mercury_requests"
OUTPUT_DIR = Path(__file__).parent / "mercury_outputs"

# ============================================================================
# ASYNC GENERATION
# ============================================================================

async def call_model(client, messages, sample_id, semaphore):
    """Goi 1 request toi mercury-2"""
    async with semaphore:
        start = time.time()
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.time() - start
            content = response.choices[0].message.content.strip() if response.choices else ""
            usage = response.usage
            return {
                "sample_id": sample_id,
                "output": content,
                "success": True,
                "time_seconds": round(elapsed, 2),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            elapsed = time.time() - start
            return {
                "sample_id": sample_id,
                "output": "",
                "success": False,
                "error": str(e),
                "time_seconds": round(elapsed, 2),
            }


async def generate_all(entries: List[Dict], max_concurrent: int = MAX_CONCURRENT) -> List[Dict]:
    """Generate output cho tat ca entries"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=INCEPTION_API_KEY,
        base_url=INCEPTION_API_BASE_URL,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for entry in entries:
        messages = entry["messages"]
        sample_id = entry["id"]
        tasks.append(call_model(client, messages, sample_id, semaphore))

    print(f"  Sending {len(tasks)} requests (concurrency={max_concurrent})...")
    results = await asyncio.gather(*tasks)
    await client.close()
    return list(results)


# ============================================================================
# MAIN
# ============================================================================

def load_existing_output(output_file: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load output file neu da ton tai de resume cac mau bi loi."""
    if not output_file.exists():
        return [], {}

    with open(output_file, "r", encoding="utf-8") as f:
        existing_output = json.load(f)

    existing_map = {str(item["id"]): item for item in existing_output}
    return existing_output, existing_map


def build_output_record(entry: Dict, generated: Dict) -> Dict:
    """Chuan hoa 1 record output tu request + generation result."""
    return {
        "id": entry["id"],
        "messages": entry["messages"],
        "old_output": entry.get("old_output", ""),
        "new_output": generated.get("output", ""),
        "new_output_success": generated.get("success", False),
        "new_output_error": generated.get("error"),
        "new_output_time_seconds": generated.get("time_seconds"),
        "new_output_prompt_tokens": generated.get("prompt_tokens"),
        "new_output_completion_tokens": generated.get("completion_tokens"),
        "application_code": entry.get("application_code", ""),
        "consumer_name": entry.get("consumer_name", ""),
        "processing_time_ms": entry.get("processing_time_ms"),
        "available_products": entry.get("available_products", []),
    }


def process_file(input_file: Path, output_file: Path, max_concurrent: int, resume_failed_only: bool):
    """Process 1 file requests -> outputs"""
    print(f"\n{'=' * 60}")
    print(f"  Processing: {input_file.name}")
    print(f"{'=' * 60}")

    with open(input_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    _, existing_map = load_existing_output(output_file)

    print(f"  Samples: {len(entries)}")
    print(f"  Model: {MODEL}")
    print(f"  API: {INCEPTION_API_BASE_URL}")

    pending_entries = entries
    if resume_failed_only and existing_map:
        pending_entries = [
            entry for entry in entries
            if not existing_map.get(str(entry["id"]), {}).get("new_output_success", False)
        ]
        skipped_success = len(entries) - len(pending_entries)
        print(f"  Resume mode: ON")
        print(f"  Existing output: {output_file}")
        print(f"  Skip success: {skipped_success}")
        print(f"  Retry failed/pending: {len(pending_entries)}")

    if not pending_entries:
        print("  Khong con mau loi/pending. Bo qua goi API.")
        success = sum(
            1 for entry in entries
            if existing_map.get(str(entry["id"]), {}).get("new_output_success", False)
        )
        failed = len(entries) - success
        print(f"  Success: {success}/{len(entries)}")
        if failed > 0:
            print(f"  Failed: {failed}")
        return {"success": success, "failed": failed, "time": 0.0}

    start = time.time()
    results = asyncio.run(generate_all(pending_entries, max_concurrent))
    elapsed = time.time() - start

    success = sum(1 for r in results if r["success"])
    failed = len(results) - success

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Success: {success}/{len(results)}")
    if failed > 0:
        print(f"  Failed: {failed}")
        for r in results:
            if not r["success"]:
                print(f"    ID {r['sample_id']}: {r.get('error', 'Unknown')}")

    # Merge ket qua moi vao file output cu, giu nguyen cac mau da thanh cong
    result_map = {str(r["sample_id"]): r for r in results}
    output_data = []
    for entry in entries:
        sid = str(entry["id"])
        existing_record = existing_map.get(sid)
        if sid in result_map:
            output_data.append(build_output_record(entry, result_map[sid]))
            continue

        if existing_record:
            output_data.append({
                "id": entry["id"],
                "messages": entry["messages"],
                "old_output": entry.get("old_output", ""),
                "new_output": existing_record.get("new_output", ""),
                "new_output_success": existing_record.get("new_output_success", False),
                "new_output_error": existing_record.get("new_output_error"),
                "new_output_time_seconds": existing_record.get("new_output_time_seconds"),
                "new_output_prompt_tokens": existing_record.get("new_output_prompt_tokens"),
                "new_output_completion_tokens": existing_record.get("new_output_completion_tokens"),
                "application_code": entry.get("application_code", ""),
                "consumer_name": entry.get("consumer_name", ""),
                "processing_time_ms": entry.get("processing_time_ms"),
                "available_products": entry.get("available_products", []),
            })
            continue

        output_data.append(build_output_record(entry, {}))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_file}")

    total_success = sum(1 for item in output_data if item["new_output_success"])
    total_failed = len(output_data) - total_success
    print(f"  Total after merge: {total_success}/{len(output_data)} success, {total_failed} failed")

    return {"success": total_success, "failed": total_failed, "time": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Generate mercury-2 outputs")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR),
                        help="Directory chua *_requests.json files")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory luu *_outputs.json files")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT,
                        help="So request dong thoi")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Chi xu ly cac file cu the (vd: crmkh_requests.json)")
    parser.add_argument("--force-all", action="store_true",
                        help="Bo qua output cu, chay lai toan bo samples")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input dir not found: {input_dir}")
        print(f"Hay chay mercury_extract_requests.py truoc!")
        return

    # Find request files
    if args.files:
        request_files = [input_dir / f for f in args.files]
    else:
        request_files = sorted(input_dir.glob("*_requests.json"))

    if not request_files:
        print(f"Khong tim thay *_requests.json trong {input_dir}")
        return

    print(f"Mercury-2 Output Generation")
    print(f"  API: {INCEPTION_API_BASE_URL}")
    print(f"  Model: {MODEL}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Input dir: {input_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Files: {[f.name for f in request_files]}")
    print(f"  Resume failed only: {not args.force_all}")

    total_start = time.time()
    all_stats = {}

    for req_file in request_files:
        if not req_file.exists():
            print(f"  SKIP: {req_file} not found")
            continue

        # crmkh_requests.json -> crmkh_outputs.json
        out_name = req_file.name.replace("_requests.json", "_outputs.json")
        out_file = output_dir / out_name

        stats = process_file(
            req_file,
            out_file,
            args.concurrency,
            resume_failed_only=not args.force_all,
        )
        all_stats[req_file.name] = stats

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  TONG KET")
    print(f"{'=' * 60}")
    for fname, stats in all_stats.items():
        print(f"  {fname}: {stats['success']} OK, {stats['failed']} fail, {stats['time']:.1f}s")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"\nCopy folder '{output_dir}' ve may noi bo de chay evaluate!")


if __name__ == "__main__":
    main()
