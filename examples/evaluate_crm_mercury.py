"""
So sánh misa-ai-1.0-plus vs mercury-2 (Inception Labs) trên CRM Dataset
=========================================================================
Flow:
  1. Load logs -> lấy input gốc (system + user messages) + output misa-ai-1.0-plus (có sẵn)
  2. Replay request gốc tới mercury-2 (Inception Labs API) -> lấy output mới
  3. Generate ground truth bằng claude-sonnet-4-5
  4. Đánh giá CẢ 2 model so với GT, dùng metrics phù hợp từng task:
     - crmkh  (gợi ý SP):     Product metrics + ROUGE + Token F1
     - crmmisa (phân tích KD): ROUGE + G-Eval + Answer Relevancy

Chạy: cd /home/misa/CUA && python3 llm-evaluate/examples/evaluate_crm_mercury.py
"""
import os
import sys
import json
import re
import time
import copy
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType, MetricConfig, TaskType
from llm_eval_framework.data_parsers import EvalSample, LogParser
from llm_eval_framework.ground_truth import GroundTruthGenerator


# ============================================================================
# CONSTANTS
# ============================================================================

DATA_PATH = "/home/misa/CUA/crm/raw_data/all_logs.json"
MAX_SAMPLES_PER_PROJECT = 50
OUTPUT_DIR = Path(__file__).parent / "crm_evaluation_results"

# MISA API Gateway (cho GT generation bằng claude-sonnet-4-5)
MISA_API_BASE_URL = "http://test-k8s.misa.local/llm-gateway/v1"
MISA_API_KEY = "misa_avaamis_00t3spja_10G8ejFtZGKz33l6K7zT9q2D_oBVENGtl"

# Inception Labs API (cho mercury-2)
INCEPTION_API_BASE_URL = "https://api.inceptionlabs.ai/v1"
INCEPTION_API_KEY = "sk_d494791d6cdeccb8803b6addc0675c65"

# Models
MODEL_OLD = "misa-ai-1.0-plus"   # output đã có sẵn trong logs
MODEL_NEW = "mercury-2"           # Inception Labs - cần gọi API để generate output mới
GT_MODEL  = "claude-sonnet-4-5"  # ground truth

# Metrics configuration theo từng loại dự án
PROJECT_CONFIGS = {
    "crmkh": {
        "task_description": "Gợi ý sản phẩm (Product Recommendation)",
        "task_type": "qa",
        "gt_prompt": """Bạn là hệ thống AI gợi ý sản phẩm cho CRM. Dựa vào lịch sử mua hàng và yêu cầu của nhân viên kinh doanh,
hãy gợi ý sản phẩm phù hợp nhất.

Yêu cầu:
- Gợi ý tối đa 5 sản phẩm dựa trên lịch sử mua hàng được cung cấp
- Ưu tiên sản phẩm khách hàng mua thường xuyên, gần đây
- CHỈ gợi ý sản phẩm có trong danh sách lịch sử
- Trả về JSON array với ProductCode

{context}

Yêu cầu cụ thể:
{input}

Trả lời ĐÚNG format JSON array:
[{{"ProductCode": "..."}}]""",
        "framework_metrics": [MetricType.ROUGE, MetricType.TOKEN_F1],
    },
    "crmmisa": {
        "task_description": "Phân tích kinh doanh (Business Analysis)",
        "task_type": "summarization",
        "gt_prompt": """Bạn là nhà phân tích kinh doanh chuyên nghiệp. Dựa vào dữ liệu dashboard được cung cấp,
hãy đưa ra những nhận định cốt lõi và đề xuất hành động cụ thể.

Yêu cầu:
- Phân tích dữ liệu dashboard một cách chính xác
- Đưa ra nhận định sắc bén, tập trung vào các chỉ số quan trọng
- Đề xuất hành động cụ thể cho nhân viên kinh doanh
- Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc

{context}

Yêu cầu cụ thể:
{input}""",
        "framework_metrics": [MetricType.ROUGE, MetricType.G_EVAL, MetricType.ANSWER_RELEVANCY],
    },
}


# ============================================================================
# CUSTOM PARSER - giữ lại original messages để replay
# ============================================================================

class CRMLogParser(LogParser):
    """Parser cho CRM logs - giữ nguyên messages gốc trong metadata."""

    def _extract_product_codes_from_content(self, content: str, pattern: str = r'ProductCode:\s*([^\n,]+)') -> List[str]:
        return [m.strip() for m in re.findall(pattern, content)]

    def _extract_product_codes_from_response(self, content: str) -> List[str]:
        try:
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [item["ProductCode"] for item in data if isinstance(item, dict) and "ProductCode" in item]
        except (json.JSONDecodeError, KeyError):
            pass
        return []

    def _parse_log_item(self, item: Dict[str, Any], index: int) -> Optional[EvalSample]:
        if item.get("isError", False):
            return None
        response_payload = item.get("responsePayload")
        if response_payload is None:
            return None

        request = self._parse_payload(item.get("requestPayload", {}))
        response = self._parse_payload(response_payload)

        messages = request.get("messages", [])
        if not messages:
            return None

        user_messages = [m for m in messages if m.get("role") == "user"]
        input_text = user_messages[-1].get("content", "") if user_messages else ""

        system_messages = [m for m in messages if m.get("role") == "system"]
        context = system_messages[0].get("content", "") if system_messages else ""

        choices = response.get("choices", [])
        output_text = ""
        if choices:
            output_text = choices[0].get("message", {}).get("content", "")

        if not input_text and not output_text:
            return None

        # Product codes
        all_content = " ".join(m.get("content", "") for m in messages)
        available_products = self._extract_product_codes_from_content(all_content)
        recommended_products = self._extract_product_codes_from_response(output_text)

        usage = response.get("usage", {})

        metadata = {
            "model": item.get("model"),
            "application_code": item.get("applicationCode"),
            "consumer_name": item.get("consumerName"),
            "trace_id": item.get("traceId"),
            "processing_time_ms": item.get("processingTimeMs"),
            "available_products": available_products,
            "recommended_products": recommended_products,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            # GIỮ NGUYÊN messages gốc để replay cho model mới
            "original_messages": messages,
        }

        return EvalSample(
            id=str(item.get("id", index)),
            input=input_text,
            output=output_text,
            context=context,
            metadata=metadata,
        )


# ============================================================================
# GENERATE OUTPUT TỪ MODEL MỚI (mercury-2 via Inception Labs API)
# ============================================================================

async def _call_model_async(client, model: str, messages: List[Dict], sample_id: str, semaphore) -> Dict[str, Any]:
    """Gọi 1 request tới model mới"""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip() if response.choices else ""
            usage = response.usage
            return {
                "sample_id": sample_id,
                "output": content,
                "success": True,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            return {
                "sample_id": sample_id,
                "output": "",
                "success": False,
                "error": str(e),
            }


async def generate_outputs_from_new_model_async(
    samples: List[EvalSample],
    model: str,
    api_key: str,
    base_url: str,
    max_concurrent: int = 5,
) -> List[Dict[str, Any]]:
    """Replay tất cả requests gốc tới model mới (async)"""
    import httpx
    from openai import AsyncOpenAI
    # Bypass SSL verification (Fortinet firewall intercepts HTTPS)
    http_client = httpx.AsyncClient(verify=False, timeout=120)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=3, http_client=http_client)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for sample in samples:
        messages = sample.metadata.get("original_messages", [])
        if not messages:
            # Fallback: tạo messages từ context + input
            msgs = []
            if sample.context:
                msgs.append({"role": "system", "content": sample.context})
            msgs.append({"role": "user", "content": sample.input})
            messages = msgs

        tasks.append(_call_model_async(client, model, messages, sample.id, semaphore))

    results = await asyncio.gather(*tasks)
    await client.close()
    return list(results)


def generate_outputs_from_new_model(
    samples: List[EvalSample],
    model: str,
    api_key: str,
    base_url: str,
    max_concurrent: int = 5,
) -> List[Dict[str, Any]]:
    """Sync wrapper"""
    return asyncio.run(generate_outputs_from_new_model_async(
        samples, model, api_key, base_url, max_concurrent
    ))


# ============================================================================
# CRM-SPECIFIC METRICS (chỉ cho task gợi ý sản phẩm)
# ============================================================================

def evaluate_product_recommendation(output: str, available_products: List[str]) -> Dict[str, Any]:
    """Đánh giá output gợi ý sản phẩm"""
    available = set(available_products)
    recommended = []

    # Extract product codes từ output
    try:
        json_match = re.search(r'\[.*\]', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            recommended = [item["ProductCode"] for item in data if isinstance(item, dict) and "ProductCode" in item]
    except (json.JSONDecodeError, KeyError):
        pass

    result = {
        "json_valid": False,
        "count_valid": False,
        "has_output": bool(output and output.strip()),
        "num_recommended": len(recommended),
        "valid_ratio": 0.0,
        "num_valid": 0,
    }

    # JSON valid?
    try:
        json_match = re.search(r'\[.*\]', output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                result["json_valid"] = True
    except Exception:
        pass

    # Valid products?
    if recommended and available:
        valid = [p for p in recommended if p in available]
        result["num_valid"] = len(valid)
        result["valid_ratio"] = len(valid) / len(recommended)

    result["count_valid"] = 0 < len(recommended) <= 5
    return result


def aggregate_product_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate product metrics"""
    n = len(results)
    if n == 0:
        return {"error": "No samples"}

    has_avail = [r for r in results if True]  # all samples
    valid_ratio_samples = [r for r in results if r.get("num_valid", 0) >= 0 and r.get("num_recommended", 0) > 0]

    return {
        "num_samples": n,
        "has_output_rate": sum(1 for r in results if r["has_output"]) / n,
        "json_valid_rate": sum(1 for r in results if r["json_valid"]) / n,
        "count_valid_rate": sum(1 for r in results if r["count_valid"]) / n,
        "avg_valid_ratio": (
            sum(r["valid_ratio"] for r in valid_ratio_samples) / len(valid_ratio_samples)
            if valid_ratio_samples else 0
        ),
        "avg_recommendations": sum(r["num_recommended"] for r in results) / n,
    }


# ============================================================================
# LOAD & GROUP
# ============================================================================

def load_and_group_by_project(data_path: str, max_per_project: int) -> Dict[str, List[EvalSample]]:
    print(f"\n{'=' * 70}")
    print(f" LOADING DATA")
    print(f"{'=' * 70}")
    print(f"\n📂 File: {data_path}")

    parser = CRMLogParser()
    all_samples = parser.parse(data_path)
    print(f"   Tổng samples parsed: {len(all_samples)}")

    projects = defaultdict(list)
    for sample in all_samples:
        project_name = sample.metadata.get("consumer_name", "Unknown")
        projects[project_name].append(sample)

    print(f"\n📋 Dự án:")
    for project, samples in projects.items():
        app_code = samples[0].metadata.get("application_code", "N/A") if samples else "N/A"
        print(f"   - {project} ({app_code}): {len(samples)} samples")

    limited = {}
    for project, samples in projects.items():
        limited[project] = samples[:max_per_project]
        if len(samples) > max_per_project:
            print(f"   -> Lấy {max_per_project}/{len(samples)}")

    return limited


def get_project_config(samples: List[EvalSample]) -> Dict[str, Any]:
    app_code = samples[0].metadata.get("application_code", "").strip() if samples else ""
    return PROJECT_CONFIGS.get(app_code, PROJECT_CONFIGS["crmmisa"])


# ============================================================================
# EVALUATE 1 MODEL trên 1 bộ samples (output + reference đã set sẵn)
# ============================================================================

def run_framework_metrics(
    samples: List[EvalSample],
    evaluator: LLMEvaluator,
    metrics_to_run: List[MetricType],
    label: str,
) -> Dict[str, Any]:
    """Chạy framework metrics, trả về dict {metric_name: {score, ...}}"""
    results = {}
    samples_with_gt = [s for s in samples if s.reference]
    if not samples_with_gt:
        return {"error": "No samples with GT"}

    for metric_type in metrics_to_run:
        metric_name = metric_type.value
        try:
            if metric_type in LLMEvaluator.LLM_METRICS:
                eval_samples = samples_with_gt[:20]
            else:
                eval_samples = samples_with_gt

            eval_results = evaluator.evaluate(eval_samples, metrics=[metric_type])
            metric_result = eval_results.get(metric_name)
            if metric_result:
                results[metric_name] = {
                    "score": round(metric_result.score, 4),
                    "score_pct": f"{metric_result.score:.2%}",
                    "num_samples": len(eval_samples),
                }
                print(f"      {metric_name}: {metric_result.score:.4f} ({metric_result.score:.2%})")
        except Exception as e:
            results[metric_name] = {"error": str(e)}
            print(f"      {metric_name}: ERROR - {e}")

    return results


# ============================================================================
# EVALUATE PROJECT - so sánh 2 models
# ============================================================================

def evaluate_project(
    project_name: str,
    samples: List[EvalSample],
    evaluator: LLMEvaluator,
    gt_config: OpenAIConfig,
) -> Dict[str, Any]:
    app_code = samples[0].metadata.get("application_code", "N/A").strip() if samples else "N/A"
    project_config = get_project_config(samples)

    print(f"\n\n{'=' * 70}")
    print(f" DỰ ÁN: {project_name} ({app_code})")
    print(f" Task: {project_config['task_description']}")
    print(f" Samples: {len(samples)}")
    print(f"{'=' * 70}")

    result_data = {
        "project_name": project_name,
        "application_code": app_code,
        "task": project_config["task_description"],
        "num_samples": len(samples),
        "models_compared": [MODEL_OLD, MODEL_NEW],
        "gt_model": GT_MODEL,
    }

    # ================================================================
    # STEP 1: Lấy output misa-ai-1.0-plus (đã có trong logs)
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 1: Output {MODEL_OLD} (có sẵn trong logs)")
    print(f"{'─' * 60}")

    old_outputs = {}
    for s in samples:
        old_outputs[s.id] = s.output  # output từ logs
    print(f"   ✅ {len(old_outputs)} outputs có sẵn")

    # Preview
    for i, s in enumerate(samples[:2]):
        preview = (s.output or "")[:120].replace('\n', ' ')
        print(f"   [{i+1}] ID {s.id}: {preview}...")

    # ================================================================
    # STEP 2: Generate output mercury-2 (gọi Inception Labs API)
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 2: Generate output {MODEL_NEW} (gọi Inception Labs API)")
    print(f"{'─' * 60}")
    print(f"   🔄 Replay {len(samples)} requests gốc tới {MODEL_NEW}...")
    print(f"   📡 API: {INCEPTION_API_BASE_URL}")

    start_time = time.time()
    new_model_results = generate_outputs_from_new_model(
        samples, MODEL_NEW, INCEPTION_API_KEY, INCEPTION_API_BASE_URL, max_concurrent=5
    )
    elapsed = time.time() - start_time

    new_outputs = {}
    success_count = 0
    for r in new_model_results:
        if r["success"]:
            new_outputs[r["sample_id"]] = r["output"]
            success_count += 1
        else:
            new_outputs[r["sample_id"]] = ""
            print(f"   ❌ Sample {r['sample_id']}: {r.get('error', 'Unknown')}")

    print(f"   ✅ Thành công: {success_count}/{len(samples)}")
    print(f"   ⏱️  Thời gian: {elapsed:.1f}s")

    # Preview
    for i, s in enumerate(samples[:2]):
        preview = (new_outputs.get(s.id, "") or "")[:120].replace('\n', ' ')
        print(f"   [{i+1}] ID {s.id}: {preview}...")

    result_data["new_model_generation"] = {
        "model": MODEL_NEW,
        "api_base_url": INCEPTION_API_BASE_URL,
        "success_count": success_count,
        "total": len(samples),
        "time_seconds": round(elapsed, 1),
    }

    # ================================================================
    # STEP 3: Generate Ground Truth (claude-sonnet-4-5)
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 3: Generate Ground Truth ({GT_MODEL})")
    print(f"{'─' * 60}")
    print(f"   🤖 Generating GT cho {len(samples)} samples...")

    generator = GroundTruthGenerator(gt_config)
    start_time = time.time()
    gt_results = generator.generate_batch(
        samples,
        task_type=project_config["task_type"],
        custom_prompt=project_config["gt_prompt"],
        max_concurrent=5,
    )
    elapsed = time.time() - start_time

    gt_map = {}
    gt_success = 0
    for r in gt_results:
        if r.success:
            gt_map[r.sample_id] = r.ground_truth
            gt_success += 1
        else:
            print(f"   ❌ Sample {r.sample_id}: {r.error}")

    print(f"   ✅ Thành công: {gt_success}/{len(samples)}")
    print(f"   ⏱️  Thời gian: {elapsed:.1f}s")

    # Preview so sánh 3 model outputs
    print(f"\n   📝 Preview (2 samples):")
    for i, s in enumerate(samples[:2]):
        gt = (gt_map.get(s.id, "") or "")[:100].replace('\n', ' ')
        old = (old_outputs.get(s.id, "") or "")[:100].replace('\n', ' ')
        new = (new_outputs.get(s.id, "") or "")[:100].replace('\n', ' ')
        print(f"\n   [{i+1}] ID: {s.id}")
        print(f"       GT  ({GT_MODEL}):  {gt}...")
        print(f"       OLD ({MODEL_OLD}): {old}...")
        print(f"       NEW ({MODEL_NEW}):      {new}...")

    result_data["gt_generation"] = {
        "model": GT_MODEL,
        "success_count": gt_success,
        "total": len(samples),
        "time_seconds": round(elapsed, 1),
    }

    if gt_success == 0:
        print(f"\n   ❌ Không có GT nào -> bỏ qua evaluation")
        return result_data

    # ================================================================
    # STEP 4: Product Metrics (chỉ cho crmkh)
    # ================================================================
    if app_code == "crmkh":
        print(f"\n{'─' * 60}")
        print(f" STEP 4a: PRODUCT RECOMMENDATION METRICS")
        print(f"{'─' * 60}")

        # Đánh giá cả 2 models
        for model_name, outputs in [(MODEL_OLD, old_outputs), (MODEL_NEW, new_outputs)]:
            product_results = []
            for s in samples:
                output = outputs.get(s.id, "")
                available = s.metadata.get("available_products", [])
                product_results.append(evaluate_product_recommendation(output, available))

            agg = aggregate_product_metrics(product_results)
            result_data[f"product_metrics_{model_name}"] = agg

            print(f"\n   📊 {model_name}:")
            print(f"      JSON valid: {agg['json_valid_rate']:.2%} | "
                  f"Count valid: {agg['count_valid_rate']:.2%} | "
                  f"SP hợp lệ: {agg['avg_valid_ratio']:.2%} | "
                  f"Avg gợi ý: {agg['avg_recommendations']:.1f}")

    # ================================================================
    # STEP 4b: Framework Metrics (so sánh cả 2 model vs GT)
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 4b: FRAMEWORK METRICS (vs Ground Truth)")
    print(f"{'─' * 60}")

    metrics_to_run = project_config["framework_metrics"]
    print(f"   Metrics: {[m.value for m in metrics_to_run]}")

    for model_name, outputs in [(MODEL_OLD, old_outputs), (MODEL_NEW, new_outputs)]:
        print(f"\n   ── {model_name} ──")

        # Tạo samples với output của model này + GT
        model_samples = []
        for s in samples:
            if s.id in gt_map and s.id in outputs:
                ms = copy.deepcopy(s)
                ms.output = outputs[s.id]
                ms.reference = gt_map[s.id]
                model_samples.append(ms)

        if not model_samples:
            print(f"      Không có samples để đánh giá")
            result_data[f"framework_metrics_{model_name}"] = {"error": "No samples"}
            continue

        print(f"      Samples: {len(model_samples)}")
        fm = run_framework_metrics(model_samples, evaluator, metrics_to_run, model_name)
        result_data[f"framework_metrics_{model_name}"] = fm

    # ================================================================
    # STEP 5: Performance comparison
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 5: PERFORMANCE")
    print(f"{'─' * 60}")

    # OLD model (từ logs)
    old_times = [
        s.metadata.get("processing_time_ms") for s in samples
        if s.metadata.get("processing_time_ms") is not None
    ]
    if old_times:
        old_times.sort()
        print(f"\n   ⏱️  {MODEL_OLD} (from logs):")
        print(f"      Avg: {sum(old_times)/len(old_times)/1000:.2f}s | "
              f"Median: {old_times[len(old_times)//2]/1000:.2f}s | "
              f"P95: {old_times[min(int(len(old_times)*0.95), len(old_times)-1)]/1000:.2f}s")

    # NEW model (từ generation)
    new_times = [
        r.get("completion_tokens", 0) for r in new_model_results if r["success"]
    ]
    # Estimate: chỉ có total time cho batch
    if new_model_results:
        total_new = result_data["new_model_generation"]["time_seconds"]
        avg_new = total_new / max(success_count, 1)
        print(f"\n   ⏱️  {MODEL_NEW} (from generation):")
        print(f"      Total: {total_new:.1f}s | Avg per request: {avg_new:.2f}s")

    result_data["performance"] = {
        f"{MODEL_OLD}_avg_ms": sum(old_times) / len(old_times) if old_times else None,
        f"{MODEL_NEW}_total_s": result_data["new_model_generation"]["time_seconds"],
        f"{MODEL_NEW}_avg_s": (
            result_data["new_model_generation"]["time_seconds"] / max(success_count, 1)
        ),
    }

    # Lưu chi tiết per-sample
    result_data["sample_details"] = []
    for s in samples:
        result_data["sample_details"].append({
            "id": s.id,
            "input_preview": s.input[:200],
            f"output_{MODEL_OLD}": (old_outputs.get(s.id, "") or "")[:500],
            f"output_{MODEL_NEW}": (new_outputs.get(s.id, "") or "")[:500],
            "ground_truth": (gt_map.get(s.id, "") or "")[:500],
        })

    return result_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    total_start = time.time()

    print("=" * 70)
    print(" SO SÁNH MODEL: misa-ai-1.0-plus vs mercury-2 (Inception Labs)")
    print("=" * 70)
    print(f" Data       : {DATA_PATH}")
    print(f" Limit      : {MAX_SAMPLES_PER_PROJECT} samples/dự án")
    print(f" Model cũ   : {MODEL_OLD} (output có sẵn trong logs)")
    print(f" Model mới  : {MODEL_NEW} (sẽ gọi Inception Labs API)")
    print(f" Ground truth: {GT_MODEL}")
    print(f" MISA Gateway: {MISA_API_BASE_URL}")
    print(f" Inception API: {INCEPTION_API_BASE_URL}")
    print("=" * 70)

    # 1. Load data
    project_samples = load_and_group_by_project(DATA_PATH, MAX_SAMPLES_PER_PROJECT)

    # 2. Setup evaluator (dùng Claude cho LLM-based metrics, qua MISA Gateway)
    eval_config = EvalConfig(
        openai=OpenAIConfig(
            api_key=MISA_API_KEY,
            base_url=MISA_API_BASE_URL,
            model=GT_MODEL,
            temperature=0.0,
            max_tokens=2048,
            timeout=120,
            max_retries=3,
        ),
        metrics=MetricConfig(
            g_eval_criteria=["coherence", "relevance", "consistency", "fluency"],
        ),
        verbose=True,
    )
    evaluator = LLMEvaluator(eval_config)

    gt_config = OpenAIConfig(
        api_key=MISA_API_KEY,
        base_url=MISA_API_BASE_URL,
        model=GT_MODEL,
        temperature=0.0,
        max_tokens=2048,
        timeout=120,
        max_retries=3,
    )

    # 3. Đánh giá từng project
    all_results = {}
    for project_name, samples in project_samples.items():
        result = evaluate_project(project_name, samples, evaluator, gt_config)
        all_results[project_name] = result

    # 4. BẢNG TỔNG HỢP SO SÁNH
    print(f"\n\n{'=' * 70}")
    print(f" BẢNG TỔNG HỢP: {MODEL_OLD} vs {MODEL_NEW}")
    print(f" (Ground truth: {GT_MODEL})")
    print(f"{'=' * 70}")

    for project_name, result in all_results.items():
        print(f"\n{'─' * 60}")
        print(f" 📌 {project_name} - {result.get('task', '')}")
        print(f"    Samples: {result['num_samples']}")
        print(f"{'─' * 60}")

        # Header
        print(f"\n   {'Metric':<30} | {MODEL_OLD:>20} | {MODEL_NEW:>20}")
        print(f"   {'─'*30}-+-{'─'*20}-+-{'─'*20}")

        # Product metrics (chỉ crmkh)
        old_pm = result.get(f"product_metrics_{MODEL_OLD}", {})
        new_pm = result.get(f"product_metrics_{MODEL_NEW}", {})
        if old_pm and new_pm and "error" not in old_pm:
            for metric_key, label in [
                ("json_valid_rate", "JSON Valid"),
                ("count_valid_rate", "Count Valid (1-5)"),
                ("avg_valid_ratio", "SP Hợp Lệ"),
                ("avg_recommendations", "Avg Gợi Ý"),
            ]:
                old_v = old_pm.get(metric_key, 0)
                new_v = new_pm.get(metric_key, 0)
                if metric_key == "avg_recommendations":
                    old_str = f"{old_v:.1f}"
                    new_str = f"{new_v:.1f}"
                else:
                    old_str = f"{old_v:.2%}"
                    new_str = f"{new_v:.2%}"

                # Winner indicator
                if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
                    if new_v > old_v:
                        new_str += " ✅"
                    elif old_v > new_v:
                        old_str += " ✅"
                print(f"   {label:<30} | {old_str:>20} | {new_str:>20}")

        # Framework metrics
        old_fm = result.get(f"framework_metrics_{MODEL_OLD}", {})
        new_fm = result.get(f"framework_metrics_{MODEL_NEW}", {})
        if old_fm and new_fm and "error" not in old_fm:
            for metric_name in old_fm:
                if metric_name == "error":
                    continue
                old_score = old_fm.get(metric_name, {}).get("score", None)
                new_score = new_fm.get(metric_name, {}).get("score", None)

                old_str = f"{old_score:.2%}" if old_score is not None else "N/A"
                new_str = f"{new_score:.2%}" if new_score is not None else "N/A"

                if old_score is not None and new_score is not None:
                    if new_score > old_score:
                        new_str += " ✅"
                    elif old_score > new_score:
                        old_str += " ✅"

                print(f"   {metric_name:<30} | {old_str:>20} | {new_str:>20}")

        # Performance
        perf = result.get("performance", {})
        old_avg = perf.get(f"{MODEL_OLD}_avg_ms")
        new_avg_s = perf.get(f"{MODEL_NEW}_avg_s")
        if old_avg is not None and new_avg_s is not None:
            old_str = f"{old_avg/1000:.2f}s"
            new_str = f"{new_avg_s:.2f}s"
            if new_avg_s < old_avg / 1000:
                new_str += " ✅"
            elif old_avg / 1000 < new_avg_s:
                old_str += " ✅"
            print(f"   {'Avg Response Time':<30} | {old_str:>20} | {new_str:>20}")

    # 5. Lưu kết quả
    print(f"\n\n{'=' * 70}")
    print(f" LƯU KẾT QUẢ")
    print(f"{'=' * 70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for project_name, result in all_results.items():
        safe_name = result.get("application_code", project_name.replace(" ", "_"))
        output_file = OUTPUT_DIR / f"{safe_name}_mercury2_comparison.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"   ✅ {project_name} → {output_file}")

    summary_file = OUTPUT_DIR / "mercury2_comparison_summary.json"
    summary = {
        "models_compared": [MODEL_OLD, MODEL_NEW],
        "gt_model": GT_MODEL,
        "data_source": DATA_PATH,
        "max_samples_per_project": MAX_SAMPLES_PER_PROJECT,
        "new_model_api": INCEPTION_API_BASE_URL,
        "total_time_seconds": round(time.time() - total_start, 1),
        "projects": {
            name: {k: v for k, v in result.items() if k != "sample_details"}
            for name, result in all_results.items()
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"   ✅ Summary → {summary_file}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f" ✅ HOÀN THÀNH! Tổng thời gian: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
