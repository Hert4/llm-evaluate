"""
STEP 1: Extract requests tu all_logs.json -> file nhe de mang ra may ngoai
=========================================================================
Chay o may noi bo (rd-vdi-ub-100):
    python3 mercury_extract_requests.py

Output:
    mercury_requests/crmkh_requests.json              (50 requests cho project crmkh)
    mercury_requests/crmmisa_dashboard_requests.json   (50 requests - phan tich dashboard)
    mercury_requests/crmmisa_text2sql_requests.json    (50 requests - sinh SQL query)
    mercury_requests/crmmisa_doc_extract_requests.json (<=50 requests - trich xuat tai lieu)

Sau do copy folder mercury_requests/ ra may ngoai, chay mercury_generate.py
"""
import json
import re
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATH = "/home/dev/Develop_2026/reports/09032026/data/all_logs.json"
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


def detect_crmmisa_task(messages: list) -> str:
    """Phat hien task type cua CRMMISA tu noi dung messages.

    Returns:
        "dashboard"    - phan tich KPI cho nhan vien KD
        "text2sql"     - sinh SQL query tu cau hoi tieng Viet
        "doc_extract"  - trich xuat thong tin tu hinh anh bien ban/ho so
    """
    if not messages:
        return "dashboard"

    first_content = messages[0].get("content", "")

    # Multimodal content (list with image_url) → document extraction
    if isinstance(first_content, list):
        return "doc_extract"

    content = str(first_content)

    if "DATABASE SCHEMA" in content or ("SQL" in content and "MySQL" in content):
        return "text2sql"

    if "nhà phân tích kinh doanh" in content or "dashboard" in content.lower():
        return "dashboard"

    # Fallback: check for document extraction keywords
    if "trích xuất" in content or "biên bản" in content or "hồ sơ nghiệp vụ" in content:
        return "doc_extract"

    return "dashboard"  # default


def detect_crmmisa_task_from_raw(raw_payload: str) -> str:
    """Phat hien task type tu raw payload string (ke ca khi bi truncate, khong parse duoc JSON)."""
    if "DATABASE SCHEMA" in raw_payload or ("SQL" in raw_payload and "MySQL" in raw_payload):
        return "text2sql"
    if "image_url" in raw_payload:
        return "doc_extract"
    if "nhà phân tích kinh doanh" in raw_payload or "dashboard" in raw_payload.lower():
        return "dashboard"
    if "trích xuất" in raw_payload or "biên bản" in raw_payload or "hồ sơ nghiệp vụ" in raw_payload:
        return "doc_extract"
    return "dashboard"


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

        # Xu ly requestPayload bi truncate (>20KB) — text2sql va doc_extract
        raw_request_payload = item.get("requestPayload", "")
        is_truncated = not messages and isinstance(raw_request_payload, str) and len(raw_request_payload) > 1000

        if not messages and not is_truncated:
            continue

        # Extract output cu (misa-ai-1.0-plus)
        choices = response.get("choices", [])
        old_output = ""
        if choices:
            old_output = choices[0].get("message", {}).get("content", "")

        if messages:
            user_messages = [m for m in messages if m.get("role") == "user"]
            input_text = user_messages[-1].get("content", "") if user_messages else ""
        else:
            input_text = "(truncated request)"

        if not input_text and not old_output:
            continue

        # Product codes (cho crmkh)
        if messages:
            all_content = " ".join(
                m.get("content", "") for m in messages
                if isinstance(m.get("content", ""), str)
            )
        else:
            all_content = ""
        available_products = extract_product_codes(all_content)

        usage = response.get("usage", {})
        app_code = item.get("applicationCode", "unknown")

        entry = {
            "id": str(item.get("id", i)),
            "messages": messages if messages else [],  # co the rong neu truncated
            "old_output": old_output,  # output misa-ai-1.0-plus (de so sanh sau)
            "application_code": app_code,
            "consumer_name": item.get("consumerName", ""),
            "processing_time_ms": item.get("processingTimeMs"),
            "available_products": available_products,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "is_truncated": is_truncated,
        }

        # Split crmmisa thanh 3 sub-tasks dua tren noi dung messages hoac raw payload
        if app_code == "crmmisa":
            if messages:
                task_type = detect_crmmisa_task(messages)
            else:
                task_type = detect_crmmisa_task_from_raw(raw_request_payload)
            project_key = f"crmmisa_{task_type}"
        else:
            project_key = app_code

        # Luu application_code theo sub-task (de evaluate.py group dung)
        entry["application_code"] = project_key
        projects[project_key].append(entry)

    # Limit va save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nProjects found:")
    for project_key, entries in projects.items():
        limited = entries[:MAX_SAMPLES_PER_PROJECT]
        print(f"  {project_key}: {len(entries)} total -> lay {len(limited)}")

        output_file = OUTPUT_DIR / f"{project_key}_requests.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(limited, f, indent=2, ensure_ascii=False)
        print(f"    -> {output_file}")

        # Print size
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"    -> Size: {size_mb:.1f} MB")

    print(f"\nDone! Copy folder '{OUTPUT_DIR}' ra may ngoai roi chay mercury_generate.py")


if __name__ == "__main__":
    main()