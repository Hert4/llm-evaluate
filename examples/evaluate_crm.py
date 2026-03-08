"""
Ví Dụ: Đánh Giá Dataset CRM (Gợi Ý Sản Phẩm)
=============================================
Dataset này chứa logs từ hệ thống CRM, task là gợi ý sản phẩm
dựa trên lịch sử mua hàng của khách hàng.

Chạy: python evaluate_crm.py
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType, MetricConfig
from llm_eval_framework.data_parsers import EvalSample, LogParser


# ============================================================================
# CUSTOM PARSER CHO CRM DATA
# ============================================================================

class CRMLogParser(LogParser):
    """
    Parser tùy chỉnh cho CRM logs.
    Xử lý format đặc thù của dữ liệu gợi ý sản phẩm.
    """

    def _extract_product_codes_from_request(self, messages: List[Dict]) -> List[str]:
        """Trích xuất danh sách ProductCode từ request (lịch sử mua hàng)"""
        product_codes = []
        for msg in messages:
            content = msg.get("content", "")
            # Tìm pattern ProductCode: XXX
            matches = re.findall(r'ProductCode:\s*([^\n]+)', content)
            product_codes.extend([m.strip() for m in matches])
        return product_codes

    def _extract_product_codes_from_response(self, content: str) -> List[str]:
        """Trích xuất ProductCode từ response JSON"""
        product_codes = []
        try:
            # Tìm JSON array trong response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    if isinstance(item, dict) and "ProductCode" in item:
                        product_codes.append(item["ProductCode"])
        except json.JSONDecodeError:
            pass
        return product_codes

    def _parse_log_item(self, item: Dict[str, Any], index: int) -> Optional[EvalSample]:
        """Override để xử lý CRM data"""
        # Parse request và response
        request = self._parse_payload(item.get("requestPayload", {}))
        response = self._parse_payload(item.get("responsePayload", {}))

        messages = request.get("messages", [])
        if not messages:
            return None

        # Lấy input (user message)
        user_messages = [m for m in messages if m.get("role") == "user"]
        input_text = user_messages[-1].get("content", "") if user_messages else ""

        # Lấy context (system message)
        system_messages = [m for m in messages if m.get("role") == "system"]
        context = system_messages[0].get("content", "") if system_messages else ""

        # Lấy output
        choices = response.get("choices", [])
        output_text = ""
        if choices:
            message = choices[0].get("message", {})
            output_text = message.get("content", "")

        if not input_text and not output_text:
            return None

        # Trích xuất product codes
        available_products = self._extract_product_codes_from_request(messages)
        recommended_products = self._extract_product_codes_from_response(output_text)

        # Metadata
        metadata = {
            "model": item.get("model"),
            "application": item.get("applicationCode"),
            "consumer": item.get("consumerName"),
            "trace_id": item.get("traceId"),
            "available_products": available_products,
            "recommended_products": recommended_products,
        }

        return EvalSample(
            id=str(item.get("id", index)),
            input=input_text,
            output=output_text,
            context=context,
            metadata=metadata,
        )


# ============================================================================
# CUSTOM METRICS CHO CRM
# ============================================================================

def evaluate_product_recommendation(sample: EvalSample) -> Dict[str, Any]:
    """
    Đánh giá chất lượng gợi ý sản phẩm.

    Metrics:
    - valid_ratio: % sản phẩm gợi ý có trong danh sách cho phép
    - json_valid: Output có đúng format JSON không
    - count_match: Số lượng gợi ý có đúng yêu cầu (max 5)
    """
    metadata = sample.metadata
    available = set(metadata.get("available_products", []))
    recommended = metadata.get("recommended_products", [])

    results = {
        "valid_ratio": 0.0,
        "json_valid": False,
        "count_valid": False,
        "num_recommended": len(recommended),
        "num_valid": 0,
        "invalid_products": [],
    }

    # Check JSON format
    try:
        json_match = re.search(r'\[.*\]', sample.output, re.DOTALL)
        if json_match:
            json.loads(json_match.group())
            results["json_valid"] = True
    except:
        pass

    # Check valid products
    if recommended:
        valid_products = [p for p in recommended if p in available]
        invalid_products = [p for p in recommended if p not in available]
        results["num_valid"] = len(valid_products)
        results["valid_ratio"] = len(valid_products) / len(recommended)
        results["invalid_products"] = invalid_products

    # Check count (should be <= 5)
    results["count_valid"] = 0 < len(recommended) <= 5

    return results


def compute_crm_metrics(samples: List[EvalSample]) -> Dict[str, Any]:
    """Tính toán metrics tổng hợp cho CRM dataset"""
    all_results = []

    for sample in samples:
        result = evaluate_product_recommendation(sample)
        all_results.append(result)

    # Aggregate
    n = len(all_results)
    if n == 0:
        return {"error": "No samples"}

    avg_valid_ratio = sum(r["valid_ratio"] for r in all_results) / n
    json_valid_rate = sum(1 for r in all_results if r["json_valid"]) / n
    count_valid_rate = sum(1 for r in all_results if r["count_valid"]) / n
    avg_recommendations = sum(r["num_recommended"] for r in all_results) / n

    return {
        "num_samples": n,
        "avg_valid_ratio": avg_valid_ratio,
        "json_valid_rate": json_valid_rate,
        "count_valid_rate": count_valid_rate,
        "avg_recommendations": avg_recommendations,
        "per_sample_results": all_results,
    }


# ============================================================================
# MAIN EVALUATION SCRIPT
# ============================================================================

def evaluate_crm_dataset(
    data_path: str,
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    generate_gt: bool = False,
):
    """
    Đánh giá dataset CRM.

    Args:
        data_path: Đường dẫn file JSON
        api_key: OpenAI API key (cho LLM-based metrics)
        max_samples: Giới hạn số samples (để test nhanh)
        generate_gt: Có generate ground truth không
    """
    print("=" * 70)
    print(" ĐÁNH GIÁ DATASET CRM - GỢI Ý SẢN PHẨM")
    print("=" * 70)

    # 1. Load data với custom parser
    print(f"\n📂 Loading data từ: {data_path}")
    parser = CRMLogParser()
    samples = parser.parse(data_path)

    if max_samples:
        samples = samples[:max_samples]

    print(f"   Loaded {len(samples)} samples")

    # 2. Hiển thị sample info
    print("\n📋 Sample info:")
    for i, sample in enumerate(samples[:3]):
        print(f"\n   [{i+1}] ID: {sample.id}")
        print(f"       Model: {sample.metadata.get('model', 'N/A')}")
        print(f"       Input: {sample.input[:100]}...")
        print(f"       Output: {sample.output[:100]}..." if sample.output else "       Output: (empty)")
        print(f"       Available products: {len(sample.metadata.get('available_products', []))}")
        print(f"       Recommended: {sample.metadata.get('recommended_products', [])}")

    # 3. Đánh giá với CRM-specific metrics
    print("\n" + "=" * 70)
    print(" 1. CRM-SPECIFIC METRICS (Product Recommendation)")
    print("=" * 70)

    crm_results = compute_crm_metrics(samples)

    print(f"\n📊 Kết quả:")
    print(f"   - Số samples: {crm_results['num_samples']}")
    print(f"   - Tỷ lệ sản phẩm hợp lệ: {crm_results['avg_valid_ratio']:.2%}")
    print(f"   - Tỷ lệ JSON đúng format: {crm_results['json_valid_rate']:.2%}")
    print(f"   - Tỷ lệ số lượng đúng (1-5): {crm_results['count_valid_rate']:.2%}")
    print(f"   - Số gợi ý trung bình: {crm_results['avg_recommendations']:.1f}")

    # 4. Đánh giá với framework metrics
    print("\n" + "=" * 70)
    print(" 2. FRAMEWORK METRICS")
    print("=" * 70)

    # Config
    config = EvalConfig(
        openai=OpenAIConfig(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            model="gpt-4o",
        ),
        metrics=MetricConfig(
            g_eval_criteria=["coherence", "relevance", "consistency"],
        )
    )

    evaluator = LLMEvaluator(config)

    # 4.1 IFEval - Kiểm tra tuân theo hướng dẫn
    print("\n📏 IFEval (Instruction Following):")
    print("   Kiểm tra: JSON format, số lượng sản phẩm, giữ nguyên mã hàng")

    results = evaluator.evaluate(samples, metrics=[MetricType.IFEVAL])
    print(f"   Score: {results['ifeval'].score:.2%}")

    # 4.2 Nếu có API key, chạy thêm các metrics khác
    if config.openai.api_key:
        print("\n🤖 LLM-based Metrics (cần API key):")

        # Faithfulness - kiểm tra có bịa sản phẩm không
        print("\n   Faithfulness (có bịa sản phẩm không):")
        try:
            results = evaluator.evaluate(samples[:5], metrics=[MetricType.FAITHFULNESS])
            print(f"   Score: {results['faithfulness'].score:.2%}")
        except Exception as e:
            print(f"   Error: {e}")

        # Answer Relevancy
        print("\n   Answer Relevancy (câu trả lời có liên quan):")
        try:
            results = evaluator.evaluate(samples[:5], metrics=[MetricType.ANSWER_RELEVANCY])
            print(f"   Score: {results['answer_relevancy'].score:.2%}")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("\n⚠️  Set OPENAI_API_KEY để chạy LLM-based metrics")

    # 5. Generate Ground Truth (nếu cần)
    if generate_gt and config.openai.api_key:
        print("\n" + "=" * 70)
        print(" 3. GENERATE GROUND TRUTH")
        print("=" * 70)

        gt_prompt = """Bạn là chuyên gia phân tích dữ liệu bán hàng.
Dựa vào lịch sử mua hàng được cung cấp, hãy gợi ý 5 sản phẩm phù hợp nhất.

Yêu cầu:
- Chỉ gợi ý sản phẩm có trong lịch sử
- Ưu tiên sản phẩm mua thường xuyên
- Trả về JSON array với ProductCode

Lịch sử mua hàng:
{input}

Trả về:
[{{"ProductCode": "..."}}]"""

        print("   Generating GT cho 5 samples đầu tiên...")
        samples_for_gt = samples[:5]
        samples_with_gt = evaluator.generate_ground_truth(
            samples_for_gt,
            task_type="qa",
            custom_prompt=gt_prompt,
        )

        for sample in samples_with_gt:
            if sample.reference:
                print(f"\n   Sample {sample.id}:")
                print(f"   GT: {sample.reference[:200]}...")

    # 6. Lưu kết quả
    print("\n" + "=" * 70)
    print(" 4. LƯU KẾT QUẢ")
    print("=" * 70)

    output_file = Path(data_path).stem + "_evaluation_results.json"
    output_data = {
        "dataset": data_path,
        "num_samples": len(samples),
        "crm_metrics": {
            k: v for k, v in crm_results.items()
            if k != "per_sample_results"
        },
        "samples_preview": [
            {
                "id": s.id,
                "recommended": s.metadata.get("recommended_products", []),
                "evaluation": evaluate_product_recommendation(s),
            }
            for s in samples[:10]
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"   Saved to: {output_file}")

    print("\n" + "=" * 70)
    print(" ✅ HOÀN THÀNH!")
    print("=" * 70)

    return crm_results


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá CRM Dataset")
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(Path(__file__).parent.parent / "data" / "examples" / "crm.json"),
        help="Đường dẫn file CRM JSON",
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--max-samples", type=int, help="Giới hạn số samples")
    parser.add_argument("--generate-gt", action="store_true", help="Generate ground truth")

    args = parser.parse_args()

    evaluate_crm_dataset(
        args.data_path,
        api_key=args.api_key,
        max_samples=args.max_samples,
        generate_gt=args.generate_gt,
    )


if __name__ == "__main__":
    main()
