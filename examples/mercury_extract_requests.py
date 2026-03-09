"""
STEP 1: Extract requests tu all_logs.json -> file nhe de mang ra may ngoai
=========================================================================
Chay o may noi bo (rd-vdi-ub-100):
    python3 mercury_extract_requests.py

Output:
    mercury_requests/crmkh_requests.json   (50 requests cho project crmkh)
    mercury_requests/crmmisa_requests.json  (50 requests cho project crmmisa)

Sau do copy folder mercury_requests/ ra may ngoai, chay mercury_generate.py
"""
import json
import re
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATH = "/home/misa/CUA/crm/raw_data/all_logs.json"
MAX_SAMPLES_PER_PROJECT = 50
OUTPUT_DIR = Path(__file__).parent / "mercury_requests"

# ============================================================================
# PARSE
# ============================================================================

def parse_payload(payload):
    """Parse JSON payload (string or dict)"""
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    elif isinstance(payload, dict):
        return payload
    return {}


def extract_product_codes(content):
    """Extract ProductCode tu content"""
    return [m.strip() for m in re.findall(r'ProductCode:\s*([^\n,]+)', content)]


def main():
    print(f"Loading {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("data", raw if isinstance(raw, list) else [])
    print(f"Total log items: {len(items)}")

    # Group by project
    projects = defaultdict(list)

    for i, item in enumerate(items):
        if item.get("isError", False):
            continue

        response_payload = item.get("responsePayload")
        if response_payload is None:
            continue

        request = parse_payload(item.get("requestPayload", {}))
        response = parse_payload(response_payload)

        messages = request.get("messages", [])
        if not messages:
            continue

        # Extract output cu (misa-ai-1.0-plus)
        choices = response.get("choices", [])
        old_output = ""
        if choices:
            old_output = choices[0].get("message", {}).get("content", "")

        user_messages = [m for m in messages if m.get("role") == "user"]
        input_text = user_messages[-1].get("content", "") if user_messages else ""

        if not input_text and not old_output:
            continue

        # Product codes (cho crmkh)
        all_content = " ".join(m.get("content", "") for m in messages)
        available_products = extract_product_codes(all_content)

        usage = response.get("usage", {})
        app_code = item.get("applicationCode", "unknown")

        entry = {
            "id": str(item.get("id", i)),
            "messages": messages,  # day la input de replay
            "old_output": old_output,  # output misa-ai-1.0-plus (de so sanh sau)
            "application_code": app_code,
            "consumer_name": item.get("consumerName", ""),
            "processing_time_ms": item.get("processingTimeMs"),
            "available_products": available_products,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

        projects[app_code].append(entry)

    # Limit va save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nProjects found:")
    for app_code, entries in projects.items():
        limited = entries[:MAX_SAMPLES_PER_PROJECT]
        print(f"  {app_code}: {len(entries)} total -> lay {len(limited)}")

        output_file = OUTPUT_DIR / f"{app_code}_requests.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(limited, f, indent=2, ensure_ascii=False)
        print(f"    -> {output_file}")

        # Print size
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"    -> Size: {size_mb:.1f} MB")

    print(f"\nDone! Copy folder '{OUTPUT_DIR}' ra may ngoai roi chay mercury_generate.py")


if __name__ == "__main__":
    main()