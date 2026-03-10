"""
Ví Dụ Chi Tiết Sử Dụng Tất Cả Metrics
=====================================
File này chứa code mẫu cho từng metric trong framework.
Chạy: python all_metrics_examples.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType, MetricConfig, TaskType
from llm_eval_framework.data_parsers import EvalSample


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(results: dict, metric_name: str):
    if metric_name in results:
        result = results[metric_name]
        print(f"\n📊 {metric_name.upper()} Score: {result.score:.4f} ({result.score*100:.2f}%)")
        if result.details:
            print("   Details:")
            for key, value in result.details.items():
                if isinstance(value, (int, float)):
                    print(f"     - {key}: {value}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        print(f"     - {k}: {v:.4f}" if isinstance(v, float) else f"     - {k}: {v}")


# ============================================================================
# 1. TRANSLATION METRICS
# ============================================================================

def example_bleu():
    """BLEU - Đánh giá dịch thuật bằng n-gram overlap"""
    print_header("1.1 BLEU - Translation Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="The cat sat on the mat",
            output="Con mèo ngồi trên thảm",
            reference="Chú mèo ngồi trên tấm thảm",
        ),
        EvalSample(
            id="2",
            input="I love programming",
            output="Tôi yêu lập trình",
            reference="Tôi thích lập trình",
        ),
        EvalSample(
            id="3",
            input="Hello world",
            output="Xin chào thế giới",
            reference="Chào thế giới",
        ),
    ]

    # Config tùy chỉnh BLEU
    config = EvalConfig(
        metrics=MetricConfig(
            bleu_max_ngram=4,      # Tính đến 4-gram
            bleu_smoothing=True,   # Tránh score = 0
        )
    )

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.BLEU])

    print_result(results, "bleu")

    # Xem score từng sample
    print("\n   Per-sample scores:")
    for i, score in enumerate(results['bleu'].per_sample_scores):
        print(f"     Sample {i+1}: {score:.4f}")


def example_comet():
    """COMET - Đánh giá dịch thuật bằng AI (hiểu nghĩa)"""
    print_header("1.2 COMET - Neural Translation Evaluation")
    print("   ⚠️ Yêu cầu: pip install unbabel-comet và GPU")
    print("   ⚠️ Skipping (demo only)")

    # Code mẫu (không chạy nếu không có COMET installed)
    code = '''
from llm_eval_framework.metrics.translation import COMETMetric

samples = [
    EvalSample(
        id="1",
        input="The cat sat on the mat",
        output="Con mèo ngồi trên thảm",
        reference="Chú mèo ngồi trên tấm thảm",
    ),
]

metric = COMETMetric({"model": "Unbabel/wmt22-comet-da"})
result = metric.compute(samples)
print(f"COMET: {result.score:.4f}")
# COMET hiểu "con mèo" = "chú mèo" nên score cao
'''
    print(f"\n   Sample code:\n{code}")


# ============================================================================
# 2. SUMMARIZATION METRICS
# ============================================================================

def example_rouge():
    """ROUGE - Đánh giá tóm tắt"""
    print_header("2.1 ROUGE - Summarization Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="""Apple Inc. công bố kết quả kinh doanh quý 3 năm 2024
            với doanh thu đạt 89.5 tỷ USD, tăng 5% so với cùng kỳ năm trước.
            CEO Tim Cook cho biết iPhone và dịch vụ đều tăng trưởng mạnh.
            Công ty cũng công bố chương trình mua lại cổ phiếu trị giá 90 tỷ USD.""",
            output="Apple đạt doanh thu 89.5 tỷ USD trong Q3, tăng 5%. iPhone bán chạy.",
            reference="Doanh thu Apple Q3 đạt 89.5 tỷ USD, tăng 5%. iPhone tăng trưởng tốt. Mua lại cổ phiếu 90 tỷ.",
        ),
        EvalSample(
            id="2",
            input="Việt Nam vô địch AFF Cup 2024 sau khi đánh bại Thái Lan 3-2 trong trận chung kết.",
            output="Việt Nam thắng Thái Lan 3-2, vô địch AFF Cup.",
            reference="Việt Nam vô địch AFF Cup 2024, thắng Thái Lan 3-2 ở chung kết.",
        ),
    ]

    config = EvalConfig(
        metrics=MetricConfig(
            rouge_types=["rouge1", "rouge2", "rougeL"],
        )
    )

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.ROUGE])

    print_result(results, "rouge")


def example_g_eval():
    """G-Eval - LLM đánh giá tóm tắt (cần API key)"""
    print_header("2.2 G-Eval - LLM-based Summarization Evaluation")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("   ⚠️ Skipping - Set OPENAI_API_KEY để chạy")
        return

    config = EvalConfig(
        openai=OpenAIConfig(api_key=api_key, model="gpt-4o"),
        metrics=MetricConfig(
            g_eval_criteria=["coherence", "fluency", "relevance", "consistency"],
        )
    )

    samples = [
        EvalSample(
            id="1",
            input="Apple công bố doanh thu 89.5 tỷ USD...",
            output="Apple đạt doanh thu kỷ lục. CEO rất vui mừng.",  # "rất vui mừng" là bịa
            reference="Apple đạt 89.5 tỷ USD doanh thu.",
        ),
    ]

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.G_EVAL])

    print_result(results, "g_eval")


# ============================================================================
# 3. QA METRICS
# ============================================================================

def example_exact_match():
    """Exact Match - Đúng hoàn toàn"""
    print_header("3.1 Exact Match - QA Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="Thủ đô Việt Nam là gì?",
            output="Hà Nội",
            reference="Hà Nội",
        ),
        EvalSample(
            id="2",
            input="Thủ đô Việt Nam là gì?",
            output="Thành phố Hà Nội",  # Thêm "Thành phố" -> EM = 0
            reference="Hà Nội",
        ),
        EvalSample(
            id="3",
            input="1 + 1 = ?",
            output="2",
            reference="2",
        ),
        EvalSample(
            id="4",
            input="Ai là tổng thống đầu tiên của Mỹ?",
            output="George Washington",
            reference="George Washington",
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(samples, metrics=[MetricType.EXACT_MATCH])

    print_result(results, "exact_match")
    print("\n   Per-sample scores:")
    for i, score in enumerate(results['exact_match'].per_sample_scores):
        status = "✅ Đúng" if score == 1.0 else "❌ Sai"
        print(f"     Sample {i+1}: {status}")


def example_token_f1():
    """Token F1 - Đánh giá từng phần đúng"""
    print_header("3.2 Token F1 - Partial Match QA")

    samples = [
        EvalSample(
            id="1",
            input="Thủ đô Việt Nam là gì?",
            output="Thành phố Hà Nội",  # EM = 0, nhưng F1 > 0 vì "Hà Nội" có
            reference="Hà Nội",
        ),
        EvalSample(
            id="2",
            input="Các thành phần của nước?",
            output="Nước gồm hydrogen và oxygen",
            reference="Hydrogen và oxygen",
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(samples, metrics=[MetricType.TOKEN_F1])

    print_result(results, "token_f1")


# ============================================================================
# 4. RANKING METRICS
# ============================================================================

def example_ndcg():
    """NDCG - Đánh giá xếp hạng"""
    print_header("4.1 NDCG@K - Ranking Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="Tìm phim hay",
            output="",
            # relevance_scores: điểm liên quan của từng item (thứ tự trả về)
            relevance_scores=[3, 2, 1, 0, 0],  # Item đầu tiên tốt nhất -> NDCG cao
        ),
        EvalSample(
            id="2",
            input="Tìm phim hay",
            output="",
            relevance_scores=[0, 0, 3, 2, 1],  # Item tốt nhất ở cuối -> NDCG thấp
        ),
        EvalSample(
            id="3",
            input="Tìm sản phẩm",
            output="",
            relevance_scores=[3, 3, 2, 1, 0],  # Optimal ranking
        ),
    ]

    config = EvalConfig(
        metrics=MetricConfig(
            k_values=[1, 3, 5, 10],
        )
    )

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.NDCG])

    print_result(results, "ndcg")


def example_recall_at_k():
    """Recall@K - Không bỏ sót"""
    print_header("4.2 Recall@K - Retrieval Coverage")

    samples = [
        EvalSample(
            id="1",
            input="Giá iPhone 16?",
            output="",
            # Có 3 docs liên quan trong tổng số 10
            relevance_scores=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            # @3: tìm được 1/3 = 33%
            # @5: tìm được 2/3 = 67%
            # @10: tìm được 3/3 = 100%
        ),
    ]

    config = EvalConfig(
        metrics=MetricConfig(
            k_values=[1, 3, 5, 10],
            relevance_threshold=0.5,
        )
    )

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.RECALL_AT_K])

    print_result(results, "recall_at_k")


def example_mrr():
    """MRR - Mean Reciprocal Rank"""
    print_header("4.3 MRR - First Relevant Position")

    samples = [
        EvalSample(
            id="1",
            input="Query 1",
            output="",
            relevance_scores=[0.0, 0.0, 1.0, 0.0, 0.0],  # First relevant at position 3 -> RR = 1/3
        ),
        EvalSample(
            id="2",
            input="Query 2",
            output="",
            relevance_scores=[1.0, 0.0, 0.0, 0.0, 0.0],  # First relevant at position 1 -> RR = 1/1
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(samples, metrics=[MetricType.MRR])

    print_result(results, "mrr")
    print(f"\n   Giải thích: MRR = (1/3 + 1/1) / 2 = {(1/3 + 1)/2:.4f}")


# ============================================================================
# 5. TOOL CALLING METRICS
# ============================================================================

def example_ast_accuracy():
    """AST Accuracy - Đánh giá function calling"""
    print_header("5.1 AST Accuracy - Tool Calling Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="Thời tiết Hà Nội hôm nay?",
            output="",
            tool_calls=[{"name": "get_weather", "arguments": {"city": "Hà Nội"}}],
            expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "Hà Nội"}}],
        ),
        EvalSample(
            id="2",
            input="Thời tiết Hà Nội?",
            output="",
            tool_calls=[{"name": "get_forecast", "arguments": {"city": "HN"}}],  # Sai tên function!
            expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "Hà Nội"}}],
        ),
        EvalSample(
            id="3",
            input="Kể chuyện cười về thời tiết",  # Không nên gọi tool
            output="Có một ngày trời mưa...",
            tool_calls=None,
            expected_tool_calls=None,  # Không gọi tool là đúng!
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(samples, metrics=[MetricType.AST_ACCURACY])

    print_result(results, "ast_accuracy")


def example_task_success():
    """Task Success Rate - Đánh giá multi-step task"""
    print_header("5.2 Task Success Rate - Multi-step Agent Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="Đặt vé máy bay HN->HCM và book khách sạn",
            output="",
            tool_calls=[
                {"name": "book_flight", "arguments": {"from": "HN", "to": "HCM"}},
                {"name": "book_hotel", "arguments": {"city": "HCM"}},
            ],
            expected_tool_calls=[
                {"name": "book_flight"},
                {"name": "book_hotel"},
            ],
        ),
        EvalSample(
            id="2",
            input="Đặt vé + khách sạn + gửi email xác nhận",
            output="",
            tool_calls=[
                {"name": "book_flight"},
                {"name": "book_hotel"},
                # Thiếu send_email!
            ],
            expected_tool_calls=[
                {"name": "book_flight"},
                {"name": "book_hotel"},
                {"name": "send_email"},
            ],
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(samples, metrics=[MetricType.TASK_SUCCESS_RATE])

    print_result(results, "task_success_rate")


# ============================================================================
# 6. CODING METRICS
# ============================================================================

def example_pass_at_k():
    """Pass@K - Code execution evaluation"""
    print_header("6. Pass@K - Code Generation Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="Viết hàm kiểm tra số nguyên tố",
            output='''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
''',
            test_cases=[
                {"test": "assert is_prime(7) == True"},
                {"test": "assert is_prime(4) == False"},
                {"test": "assert is_prime(1) == False"},
                {"test": "assert is_prime(2) == True"},
            ],
        ),
        EvalSample(
            id="2",
            input="Viết hàm tính tổng",
            output='''
def sum_list(lst):
    return sum(lst)
''',
            test_cases=[
                {"test": "assert sum_list([1,2,3]) == 6"},
                {"test": "assert sum_list([]) == 0"},
            ],
        ),
        EvalSample(
            id="3",
            input="Viết hàm đảo chuỗi (có bug)",
            output='''
def reverse_string(s):
    return s  # Bug: không đảo!
''',
            test_cases=[
                {"test": "assert reverse_string('abc') == 'cba'"},
            ],
        ),
    ]

    config = EvalConfig(
        metrics=MetricConfig(
            pass_k_values=[1, 5, 10],
            code_timeout=10,
        )
    )

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.PASS_AT_K])

    print_result(results, "pass_at_k")


# ============================================================================
# 7. REASONING METRICS
# ============================================================================

def example_accuracy():
    """Accuracy - Multiple choice evaluation"""
    print_header("7. Accuracy - Multiple Choice Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="Hành tinh lớn nhất trong hệ Mặt Trời?",
            output="Đáp án là B) Jupiter vì nó có khối lượng lớn nhất.",
            choices=["Mars", "Jupiter", "Saturn", "Neptune"],
            correct_answer="B",
        ),
        EvalSample(
            id="2",
            input="H2O là công thức của chất nào?",
            output="A",
            choices=["Nước", "Muối", "Đường", "Axit"],
            correct_answer="A",
        ),
        EvalSample(
            id="3",
            input="1 + 1 = ?",
            output="Đáp án C: 2",
            choices=["1", "3", "2", "4"],
            correct_answer="C",
        ),
        EvalSample(
            id="4",
            input="Thủ đô của Nhật Bản?",
            output="B - Tokyo",  # Sai, đáp án đúng là A
            choices=["Tokyo", "Osaka", "Kyoto", "Hokkaido"],
            correct_answer="A",
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(samples, metrics=[MetricType.ACCURACY])

    print_result(results, "accuracy")

    print("\n   Per-sample:")
    for i, score in enumerate(results['accuracy'].per_sample_scores):
        status = "✅" if score == 1.0 else "❌"
        print(f"     Sample {i+1}: {status}")


# ============================================================================
# 8. RAG METRICS
# ============================================================================

def example_faithfulness():
    """Faithfulness - Phát hiện hallucination trong RAG"""
    print_header("8.1 Faithfulness - RAG Hallucination Detection")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("   ⚠️ Skipping - Set OPENAI_API_KEY để chạy")
        print("\n   Sample code:")
        print('''
samples = [
    EvalSample(
        id="1",
        input="Doanh thu Tesla Q3 2024?",
        context="Tesla đạt doanh thu 25.2 tỷ USD trong Q3 2024.",
        output="Tesla đạt 25.2 tỷ USD. CEO Elon Musk nói đây là kỷ lục.",
        # ❌ "CEO nói đây là kỷ lục" không có trong context -> BỊA!
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.FAITHFULNESS])
# Faithfulness ~ 0.5 (1/2 claims đúng)
''')
        return

    config = EvalConfig(
        openai=OpenAIConfig(api_key=api_key, model="gpt-4o")
    )

    samples = [
        EvalSample(
            id="1",
            input="Doanh thu Tesla Q3 2024?",
            context="Tesla đạt doanh thu 25.2 tỷ USD trong Q3 2024.",
            output="Tesla đạt 25.2 tỷ USD. CEO Elon Musk gọi đây là kỷ lục.",
        ),
        EvalSample(
            id="2",
            input="Doanh thu Tesla?",
            context="Tesla đạt 25.2 tỷ USD trong Q3 2024, tăng 8%.",
            output="Doanh thu Tesla Q3 là 25.2 tỷ USD, tăng 8%.",  # ✅ Đúng
        ),
    ]

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.FAITHFULNESS])

    print_result(results, "faithfulness")


def example_context_metrics():
    """Context Precision & Recall"""
    print_header("8.2 Context Precision & Recall")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("   ⚠️ Skipping - Set OPENAI_API_KEY để chạy")
        return

    config = EvalConfig(
        openai=OpenAIConfig(api_key=api_key, model="gpt-4o")
    )

    samples = [
        EvalSample(
            id="1",
            input="Giá iPhone 16?",
            context="""
            Doc 1: iPhone 16 có giá từ 799 USD.
            Doc 2: iPhone 16 có camera 48MP.
            Doc 3: So sánh iPhone vs Samsung.
            Doc 4: Phụ kiện iPhone chính hãng.
            """,
            output="iPhone 16 giá 799 USD",
            reference="799 USD",
        ),
    ]

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[
        MetricType.CONTEXT_PRECISION,
        MetricType.CONTEXT_RECALL,
    ])

    print_result(results, "context_precision")
    print_result(results, "context_recall")


# ============================================================================
# 9. SAFETY METRICS
# ============================================================================

def example_factscore():
    """FactScore - Kiểm tra sự thật từng claim"""
    print_header("9.1 FactScore - Atomic Fact Verification")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("   ⚠️ Skipping - Set OPENAI_API_KEY để chạy")
        return

    config = EvalConfig(
        openai=OpenAIConfig(api_key=api_key, model="gpt-4o")
    )

    samples = [
        EvalSample(
            id="1",
            input="Kể về Albert Einstein",
            output="""
            Einstein sinh năm 1879 tại Đức.
            Ông đoạt giải Nobel Vật lý năm 1921.
            Giải Nobel là vì Thuyết Tương Đối.
            """,
            # Fact 1: sinh 1879 ✅
            # Fact 2: tại Đức ✅
            # Fact 3: Nobel 1921 ✅
            # Fact 4: vì Thuyết Tương Đối ❌ (thực tế là Hiệu ứng Quang điện)
            reference="Einstein sinh 1879, Nobel 1921 cho Hiệu ứng Quang điện",
        ),
    ]

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.FACTSCORE])

    print_result(results, "factscore")


def example_ifeval():
    """IFEval - Kiểm tra tuân theo hướng dẫn"""
    print_header("9.2 IFEval - Instruction Following Evaluation")

    samples = [
        EvalSample(
            id="1",
            input="Viết về mùa xuân. Dùng đúng 3 câu. Không dùng từ 'hoa'. Kết thúc bằng dấu !",
            output="Mùa xuân ấm áp. Cây cối xanh tươi. Thật tuyệt vời!",
            # ✅ 3 câu ✅ không có 'hoa' ✅ kết thúc !
        ),
        EvalSample(
            id="2",
            input="Viết 3 câu về mùa xuân. Không dùng từ 'hoa'.",
            output="Mùa xuân với những bông hoa đẹp. Thật tuyệt!",
            # ❌ chỉ 2 câu ❌ có từ 'hoa'
        ),
        EvalSample(
            id="3",
            input="Viết 5 từ về biển.",
            output="Biển xanh sóng vỗ bờ cát.",  # Đúng 5 từ
        ),
    ]

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(samples, metrics=[MetricType.IFEVAL])

    print_result(results, "ifeval")


# ============================================================================
# 10. CHAT METRICS
# ============================================================================

def example_win_rate():
    """Win Rate - So sánh với baseline"""
    print_header("10.1 Win Rate - LLM-as-Judge Comparison")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("   ⚠️ Skipping - Set OPENAI_API_KEY để chạy")
        return

    config = EvalConfig(
        openai=OpenAIConfig(api_key=api_key, model="gpt-4o")
    )

    samples = [
        EvalSample(
            id="1",
            input="Giải thích AI là gì?",
            output="""AI (Artificial Intelligence) là trí tuệ nhân tạo,
            cho phép máy tính học hỏi và ra quyết định như con người.
            AI được ứng dụng trong nhiều lĩnh vực như y tế, giao thông, giáo dục.""",
            reference="AI là Artificial Intelligence.",  # Baseline đơn giản
        ),
    ]

    evaluator = LLMEvaluator(config)
    results = evaluator.evaluate(samples, metrics=[MetricType.WIN_RATE])

    print_result(results, "win_rate")


# ============================================================================
# MAIN - CHẠY TẤT CẢ VÍ DỤ
# ============================================================================

def main():
    print("\n" + "🚀" * 35)
    print(" LLM EVALUATION FRAMEWORK - VÍ DỤ METRICS")
    print("🚀" * 35)

    # Metrics không cần API key
    print("\n📌 METRICS KHÔNG CẦN API KEY:")
    example_bleu()
    example_comet()
    example_rouge()
    example_exact_match()
    example_token_f1()
    example_ndcg()
    example_recall_at_k()
    example_mrr()
    example_ast_accuracy()
    example_task_success()
    example_pass_at_k()
    example_accuracy()
    example_ifeval()

    # Metrics cần API key
    print("\n📌 METRICS CẦN OPENAI_API_KEY:")
    example_g_eval()
    example_faithfulness()
    example_context_metrics()
    example_factscore()
    example_win_rate()

    print("\n" + "=" * 70)
    print(" ✅ HOÀN THÀNH!")
    print("=" * 70)


if __name__ == "__main__":
    main()
