"""
Đánh giá model được cấu hình trong config.yaml so với baseline
=========================================================================
Chạy:
    python3 evaluate.py                          # dùng config.yaml
    python3 evaluate.py --config mercury.yaml    # dùng config khác
"""
import os
import sys
import json
import re
import time
import copy
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import yaml

# Add framework to path
_framework_dir = Path(__file__).parent.parent.parent    # .../llm-evaluate/
_parent_dir = _framework_dir.parent                    # .../Develop_2026/

sys.path.insert(0, str(_parent_dir))

# Nếu folder tên llm-evaluate (dash), tạo symlink llm_eval_framework để Python import được
_expected_pkg = _parent_dir / "llm_eval_framework"
if not _expected_pkg.exists() and _framework_dir.exists():
    try:
        _expected_pkg.symlink_to(_framework_dir)
    except OSError:
        pass

from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType, MetricConfig, TaskType
from llm_eval_framework.data_parsers import EvalSample, LogParser
from llm_eval_framework.ground_truth import GroundTruthGenerator


# ============================================================================
# CRMMISA TASK TYPE DETECTION
# ============================================================================

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
    """Phat hien task type tu raw payload string (ke ca khi bi truncate, khong parse duoc JSON).

    Dung cho truong hop requestPayload bi cat (>20KB) nhu text2sql (chua DATABASE SCHEMA dai)
    hoac doc_extract (chua image base64).
    """
    if "DATABASE SCHEMA" in raw_payload or ("SQL" in raw_payload and "MySQL" in raw_payload):
        return "text2sql"

    if "image_url" in raw_payload:
        return "doc_extract"

    if "nhà phân tích kinh doanh" in raw_payload or "dashboard" in raw_payload.lower():
        return "dashboard"

    if "trích xuất" in raw_payload or "biên bản" in raw_payload or "hồ sơ nghiệp vụ" in raw_payload:
        return "doc_extract"

    return "dashboard"


# ============================================================================
# CONFIG LOADER
# ============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Đọc config từ YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Mapping từ string metric name -> MetricType enum
METRIC_NAME_MAP = {
    "rouge": MetricType.ROUGE,
    "token_f1": MetricType.TOKEN_F1,
    "g_eval": MetricType.G_EVAL,
    "answer_relevancy": MetricType.ANSWER_RELEVANCY,
}


def resolve_metrics(metric_names: List[str]) -> List[MetricType]:
    """Chuyển đổi list string metric names -> list MetricType."""
    result = []
    for name in metric_names:
        mt = METRIC_NAME_MAP.get(name.lower())
        if mt:
            result.append(mt)
        else:
            print(f"  WARNING: Unknown metric '{name}', skipped")
    return result


def build_project_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Xây dựng PROJECT_CONFIGS từ config.yaml, chuyển metric strings -> MetricType."""
    projects = config.get("evaluation", {}).get("projects", {})
    result = {}
    for app_code, proj_cfg in projects.items():
        result[app_code] = {
            "task_description": proj_cfg["task_description"],
            "task_type": proj_cfg["task_type"],
            "gt_prompt": proj_cfg["gt_prompt"],
            "framework_metrics": resolve_metrics(proj_cfg.get("framework_metrics", [])),
            "has_product_metrics": proj_cfg.get("has_product_metrics", False),
        }
    return result


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

        # Xu ly truong hop requestPayload bi truncate (>20KB)
        # Text2SQL va Doc extraction co payload qua dai bi cat
        raw_request_payload = item.get("requestPayload", "")
        is_truncated = not messages and isinstance(raw_request_payload, str) and len(raw_request_payload) > 1000

        if not messages and not is_truncated:
            return None

        if messages:
            user_messages = [m for m in messages if m.get("role") == "user"]
            input_text = user_messages[-1].get("content", "") if user_messages else ""

            system_messages = [m for m in messages if m.get("role") == "system"]
            context = system_messages[0].get("content", "") if system_messages else ""
        else:
            # Truncated payload: dung raw string lam context, khong co input rieng
            input_text = "(truncated request)"
            context = raw_request_payload[:4000]  # giu 4KB dau lam context

        choices = response.get("choices", [])
        output_text = ""
        if choices:
            output_text = choices[0].get("message", {}).get("content", "")

        if not input_text and not output_text:
            return None

        # Product codes
        if messages:
            all_content = " ".join(
                m.get("content", "") for m in messages
                if isinstance(m.get("content", ""), str)
            )
        else:
            all_content = raw_request_payload[:4000] if is_truncated else ""
        available_products = self._extract_product_codes_from_content(all_content)
        recommended_products = self._extract_product_codes_from_response(output_text)

        usage = response.get("usage", {})

        app_code = item.get("applicationCode")

        # Split crmmisa thanh 3 sub-tasks dua tren noi dung messages hoac raw payload
        if app_code == "crmmisa":
            if messages:
                task_type = detect_crmmisa_task(messages)
            else:
                task_type = detect_crmmisa_task_from_raw(raw_request_payload)
            app_code = f"crmmisa_{task_type}"

        metadata = {
            "model": item.get("model"),
            "application_code": app_code,
            "consumer_name": item.get("consumerName"),
            "trace_id": item.get("traceId"),
            "processing_time_ms": item.get("processingTimeMs"),
            "available_products": available_products,
            "recommended_products": recommended_products,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            # GIỮ NGUYÊN messages gốc để replay cho model mới
            "original_messages": messages,
            "is_truncated": is_truncated,
        }

        return EvalSample(
            id=str(item.get("id", index)),
            input=input_text,
            output=output_text,
            context=context,
            metadata=metadata,
        )


# ============================================================================
# LOAD NEW MODEL OUTPUTS TỪ FILE (đã generate offline) — GENERIC
# ============================================================================

def load_new_model_outputs(app_code: str, output_dir: Path) -> Dict[str, str]:
    """Load outputs từ thư mục config, generic cho mọi model."""
    output_file = output_dir / f"{app_code}_outputs.json"
    if not output_file.exists():
        return {}

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    outputs = {}
    for item in data:
        sid = str(item["id"])
        if item.get("new_output_success", False):
            outputs[sid] = item.get("new_output", "")
        else:
            outputs[sid] = ""
    return outputs


def load_new_model_times(app_code: str, output_dir: Path) -> List[float]:
    """Load response times từ file outputs, generic cho mọi model."""
    output_file = output_dir / f"{app_code}_outputs.json"
    if not output_file.exists():
        return []

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        item["new_output_time_seconds"]
        for item in data
        if item.get("new_output_success") and item.get("new_output_time_seconds")
    ]


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

    valid_ratio_samples = [r for r in results if r.get("num_recommended", 0) > 0]

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
    print(f"\n   File: {data_path}")

    parser = CRMLogParser()
    all_samples = parser.parse(data_path)
    print(f"   Tong samples parsed: {len(all_samples)}")

    projects = defaultdict(list)
    for sample in all_samples:
        project_name = sample.metadata.get("application_code", "unknown")
        projects[project_name].append(sample)

    print(f"\n   Du an:")
    for project, samples in projects.items():
        print(f"   - {project}: {len(samples)} samples")

    limited = {}
    for project, samples in projects.items():
        limited[project] = samples[:max_per_project]
        if len(samples) > max_per_project:
            print(f"   -> Lay {max_per_project}/{len(samples)}")

    return limited


def get_project_config(samples: List[EvalSample], project_configs: Dict[str, Any]) -> Dict[str, Any]:
    app_code = samples[0].metadata.get("application_code", "").strip() if samples else ""
    # Fallback to first config if app_code not found
    default_key = list(project_configs.keys())[-1] if project_configs else None
    return project_configs.get(app_code, project_configs.get(default_key, {}))


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
    config: Dict[str, Any],
    project_configs: Dict[str, Any],
) -> Dict[str, Any]:
    eval_cfg = config["evaluation"]
    model_cfg = config["model"]
    gen_cfg = config["generation"]

    MODEL_OLD = eval_cfg["baseline_model"]
    MODEL_NEW = model_cfg["name"]
    GT_MODEL = eval_cfg["gt"]["model"]

    # Resolve output dir for new model outputs
    base_dir = Path(__file__).parent
    new_outputs_dir = base_dir / gen_cfg["output_dir"]

    app_code = samples[0].metadata.get("application_code", "N/A").strip() if samples else "N/A"
    project_config = get_project_config(samples, project_configs)
    display_name = project_config.get("task_description", project_name)

    print(f"\n\n{'=' * 70}")
    print(f" DU AN: {display_name} ({app_code})")
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
    # STEP 1: Lấy output baseline (đã có trong logs)
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 1: Output {MODEL_OLD} (co san trong logs)")
    print(f"{'─' * 60}")

    old_outputs = {}
    for s in samples:
        old_outputs[s.id] = s.output  # output từ logs
    print(f"   {len(old_outputs)} outputs co san")

    # Preview
    for i, s in enumerate(samples[:2]):
        preview = (s.output or "")[:120].replace('\n', ' ')
        print(f"   [{i+1}] ID {s.id}: {preview}...")

    # ================================================================
    # STEP 2: Load output model mới (từ file đã generate offline)
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 2: Load output {MODEL_NEW} (tu file offline)")
    print(f"{'─' * 60}")

    output_file = new_outputs_dir / f"{app_code}_outputs.json"
    print(f"   File: {output_file}")

    new_outputs = load_new_model_outputs(app_code, new_outputs_dir)
    # Chỉ giữ outputs cho samples hiện tại
    success_count = sum(1 for s in samples if new_outputs.get(s.id, "").strip())
    # Đảm bảo tất cả sample IDs có entry
    for s in samples:
        if s.id not in new_outputs:
            new_outputs[s.id] = ""

    print(f"   Co output: {success_count}/{len(samples)}")

    # Preview
    for i, s in enumerate(samples[:2]):
        preview = (new_outputs.get(s.id, "") or "")[:120].replace('\n', ' ')
        print(f"   [{i+1}] ID {s.id}: {preview}...")

    result_data["new_model_source"] = {
        "model": MODEL_NEW,
        "source": str(output_file),
        "success_count": success_count,
        "total": len(samples),
    }

    # ================================================================
    # STEP 3: Generate Ground Truth
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 3: Generate Ground Truth ({GT_MODEL})")
    print(f"{'─' * 60}")
    print(f"   Generating GT cho {len(samples)} samples...")

    gt_max_concurrent = eval_cfg["gt"].get("max_concurrent", 5)

    generator = GroundTruthGenerator(gt_config)
    start_time = time.time()
    gt_results = generator.generate_batch(
        samples,
        task_type=project_config["task_type"],
        custom_prompt=project_config["gt_prompt"],
        max_concurrent=gt_max_concurrent,
    )
    elapsed = time.time() - start_time

    gt_map = {}
    gt_success = 0
    for r in gt_results:
        if r.success:
            gt_map[r.sample_id] = r.ground_truth
            gt_success += 1
        else:
            print(f"   FAIL Sample {r.sample_id}: {r.error}")

    print(f"   Thanh cong: {gt_success}/{len(samples)}")
    print(f"   Thoi gian: {elapsed:.1f}s")

    # Preview so sánh 3 model outputs
    print(f"\n   Preview (2 samples):")
    for i, s in enumerate(samples[:2]):
        gt = (gt_map.get(s.id, "") or "")[:100].replace('\n', ' ')
        old = (old_outputs.get(s.id, "") or "")[:100].replace('\n', ' ')
        new = (new_outputs.get(s.id, "") or "")[:100].replace('\n', ' ')
        print(f"\n   [{i+1}] ID: {s.id}")
        print(f"       GT  ({GT_MODEL}):          {gt}...")
        print(f"       OLD ({MODEL_OLD}):  {old}...")
        print(f"       NEW ({MODEL_NEW}):      {new}...")

    result_data["gt_generation"] = {
        "model": GT_MODEL,
        "success_count": gt_success,
        "total": len(samples),
        "time_seconds": round(elapsed, 1),
    }

    if gt_success == 0:
        print(f"\n   Khong co GT nao -> bo qua evaluation")
        return result_data

    # ================================================================
    # STEP 4a: Product Metrics (chỉ cho project có has_product_metrics)
    # ================================================================
    if project_config.get("has_product_metrics", False):
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

            print(f"\n   {model_name}:")
            print(f"      JSON valid: {agg['json_valid_rate']:.2%} | "
                  f"Count valid: {agg['count_valid_rate']:.2%} | "
                  f"SP hop le: {agg['avg_valid_ratio']:.2%} | "
                  f"Avg goi y: {agg['avg_recommendations']:.1f}")

    # ================================================================
    # STEP 4b: Framework Metrics (so sánh cả 2 model vs GT)
    # ================================================================
    print(f"\n{'─' * 60}")
    print(f" STEP 4b: FRAMEWORK METRICS (vs Ground Truth)")
    print(f"{'─' * 60}")

    metrics_to_run = project_config["framework_metrics"]
    print(f"   Metrics: {[m.value for m in metrics_to_run]}")

    for model_name, outputs in [(MODEL_OLD, old_outputs), (MODEL_NEW, new_outputs)]:
        print(f"\n   -- {model_name} --")

        # Tạo samples với output của model này + GT
        model_samples = []
        for s in samples:
            if s.id in gt_map and s.id in outputs:
                ms = copy.deepcopy(s)
                ms.output = outputs[s.id]
                ms.reference = gt_map[s.id]
                model_samples.append(ms)

        if not model_samples:
            print(f"      Khong co samples de danh gia")
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
        print(f"\n   {MODEL_OLD} (from logs):")
        print(f"      Avg: {sum(old_times)/len(old_times)/1000:.2f}s | "
              f"Median: {old_times[len(old_times)//2]/1000:.2f}s | "
              f"P95: {old_times[min(int(len(old_times)*0.95), len(old_times)-1)]/1000:.2f}s")

    # NEW model (từ output file)
    new_times = load_new_model_times(app_code, new_outputs_dir)

    if new_times:
        new_times.sort()
        avg_new = sum(new_times) / len(new_times)
        print(f"\n   {MODEL_NEW} (from offline generation):")
        print(f"      Avg: {avg_new:.2f}s | "
              f"Median: {new_times[len(new_times)//2]:.2f}s | "
              f"P95: {new_times[min(int(len(new_times)*0.95), len(new_times)-1)]:.2f}s")

    result_data["performance"] = {
        f"{MODEL_OLD}_avg_ms": sum(old_times) / len(old_times) if old_times else None,
        f"{MODEL_NEW}_avg_s": sum(new_times) / len(new_times) if new_times else None,
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
    parser = argparse.ArgumentParser(description="Evaluate model (config-driven)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")
    args = parser.parse_args()

    total_start = time.time()

    # Load config
    config = load_config(args.config)
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    gen_cfg = config["generation"]

    MODEL_OLD = eval_cfg["baseline_model"]
    MODEL_NEW = model_cfg["name"]
    MODEL_NEW_DISPLAY = model_cfg.get("display_name", MODEL_NEW)
    GT_MODEL = eval_cfg["gt"]["model"]
    DATA_PATH = eval_cfg["data_path"]
    MAX_SAMPLES_PER_PROJECT = eval_cfg.get("max_samples_per_project", 50)

    base_dir = Path(__file__).parent
    OUTPUT_DIR = base_dir / eval_cfg["output_dir"]
    NEW_OUTPUTS_DIR = base_dir / gen_cfg["output_dir"]

    # Build project configs from YAML
    project_configs = build_project_configs(config)

    print("=" * 70)
    print(f" SO SANH MODEL: {MODEL_OLD} vs {MODEL_NEW}")
    print("=" * 70)
    print(f" Config      : {args.config}")
    print(f" Data        : {DATA_PATH}")
    print(f" Limit       : {MAX_SAMPLES_PER_PROJECT} samples/du an")
    print(f" Model cu    : {MODEL_OLD} (output co san trong logs)")
    print(f" Model moi   : {MODEL_NEW} (output tu file offline)")
    print(f" Ground truth: {GT_MODEL}")
    print(f" GT API      : {eval_cfg['gt']['api_base_url']}")
    print(f" New outputs : {NEW_OUTPUTS_DIR}")
    print("=" * 70)

    # 1. Load data
    project_samples = load_and_group_by_project(DATA_PATH, MAX_SAMPLES_PER_PROJECT)

    # 2. Setup evaluator (dùng LLM evaluator config cho LLM-based metrics)
    llm_eval_cfg = eval_cfg.get("llm_evaluator", eval_cfg["gt"])
    eval_config = EvalConfig(
        openai=OpenAIConfig(
            api_key=llm_eval_cfg.get("api_key", eval_cfg["gt"]["api_key"]),
            base_url=llm_eval_cfg.get("api_base_url", eval_cfg["gt"]["api_base_url"]),
            model=llm_eval_cfg.get("model", GT_MODEL),
            temperature=0.0,
            max_tokens=2048,
            timeout=eval_cfg["gt"].get("timeout", 120),
            max_retries=eval_cfg["gt"].get("max_retries", 3),
        ),
        metrics=MetricConfig(
            g_eval_criteria=llm_eval_cfg.get("g_eval_criteria", ["coherence", "relevance", "consistency", "fluency"]),
        ),
        verbose=True,
    )
    evaluator = LLMEvaluator(eval_config)

    gt_cfg = eval_cfg["gt"]
    gt_config = OpenAIConfig(
        api_key=gt_cfg["api_key"],
        base_url=gt_cfg["api_base_url"],
        model=gt_cfg["model"],
        temperature=gt_cfg.get("temperature", 0.0),
        max_tokens=gt_cfg.get("max_tokens", 2048),
        timeout=gt_cfg.get("timeout", 120),
        max_retries=gt_cfg.get("max_retries", 3),
    )

    # 3. Đánh giá từng project
    all_results = {}
    for project_name, samples in project_samples.items():
        result = evaluate_project(
            project_name, samples, evaluator, gt_config, config, project_configs
        )
        all_results[project_name] = result

    # 4. BẢNG TỔNG HỢP SO SÁNH
    print(f"\n\n{'=' * 70}")
    print(f" BANG TONG HOP: {MODEL_OLD} vs {MODEL_NEW}")
    print(f" (Ground truth: {GT_MODEL})")
    print(f"{'=' * 70}")

    for project_name, result in all_results.items():
        print(f"\n{'─' * 60}")
        print(f" {project_name} - {result.get('task', '')}")
        print(f"    Samples: {result['num_samples']}")
        print(f"{'─' * 60}")

        # Header
        print(f"\n   {'Metric':<30} | {MODEL_OLD:>20} | {MODEL_NEW:>20}")
        print(f"   {'─'*30}-+-{'─'*20}-+-{'─'*20}")

        # Product metrics (nếu có)
        old_pm = result.get(f"product_metrics_{MODEL_OLD}", {})
        new_pm = result.get(f"product_metrics_{MODEL_NEW}", {})
        if old_pm and new_pm and "error" not in old_pm:
            for metric_key, label in [
                ("json_valid_rate", "JSON Valid"),
                ("count_valid_rate", "Count Valid (1-5)"),
                ("avg_valid_ratio", "SP Hop Le"),
                ("avg_recommendations", "Avg Goi Y"),
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
                        new_str += " [WIN]"
                    elif old_v > new_v:
                        old_str += " [WIN]"
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
                        new_str += " [WIN]"
                    elif old_score > new_score:
                        old_str += " [WIN]"

                print(f"   {metric_name:<30} | {old_str:>20} | {new_str:>20}")

        # Performance
        perf = result.get("performance", {})
        old_avg = perf.get(f"{MODEL_OLD}_avg_ms")
        new_avg_s = perf.get(f"{MODEL_NEW}_avg_s")
        if old_avg is not None and new_avg_s is not None:
            old_str = f"{old_avg/1000:.2f}s"
            new_str = f"{new_avg_s:.2f}s"
            if new_avg_s < old_avg / 1000:
                new_str += " [WIN]"
            elif old_avg / 1000 < new_avg_s:
                old_str += " [WIN]"
            print(f"   {'Avg Response Time':<30} | {old_str:>20} | {new_str:>20}")

    # 5. Lưu kết quả
    print(f"\n\n{'=' * 70}")
    print(f" LUU KET QUA")
    print(f"{'=' * 70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tạo safe model name cho file output
    safe_model = MODEL_NEW.replace("/", "_").replace(" ", "_")

    for project_name, result in all_results.items():
        safe_name = result.get("application_code", project_name.replace(" ", "_"))
        output_file = OUTPUT_DIR / f"{safe_name}_{safe_model}_comparison.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"   {project_name} -> {output_file}")

    summary_file = OUTPUT_DIR / f"{safe_model}_comparison_summary.json"
    summary = {
        "config_file": args.config,
        "models_compared": [MODEL_OLD, MODEL_NEW],
        "gt_model": GT_MODEL,
        "data_source": DATA_PATH,
        "max_samples_per_project": MAX_SAMPLES_PER_PROJECT,
        "new_model_outputs_dir": str(NEW_OUTPUTS_DIR),
        "total_time_seconds": round(time.time() - total_start, 1),
        "projects": {
            name: {k: v for k, v in result.items() if k != "sample_details"}
            for name, result in all_results.items()
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"   Summary -> {summary_file}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f" HOAN THANH! Tong thoi gian: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
