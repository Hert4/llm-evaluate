# Hướng Dẫn Sử Dụng Metrics

## Mục Lục
1. [Translation Metrics](#1-translation-metrics)
2. [Summarization Metrics](#2-summarization-metrics)
3. [QA Metrics](#3-qa-metrics)
4. [Ranking Metrics](#4-ranking-metrics)
5. [Tool Calling Metrics](#5-tool-calling-metrics)
6. [Coding Metrics](#6-coding-metrics)
7. [Reasoning Metrics](#7-reasoning-metrics)
8. [RAG Metrics](#8-rag-metrics)
9. [Safety Metrics](#9-safety-metrics)
10. [Chat Metrics](#10-chat-metrics)

---

## 1. Translation Metrics

### BLEU (Bilingual Evaluation Understudy)

**Mô tả:** Đếm số n-gram (cụm từ) trùng nhau giữa bản dịch và bản dịch mẫu.

**Khi nào dùng:**
- Đánh giá nhanh hệ thống dịch máy
- So sánh nhiều model dịch với nhau
- Chạy trong CI/CD pipeline (nhanh, không cần API)

**Điểm mạnh:**
- Nhanh, không cần GPU/API
- Dễ so sánh giữa các model

**Hạn chế:**
- Không hiểu nghĩa (đồng nghĩa bị phạt)
- Cần có bản dịch mẫu sẵn

**Code:**
```python
from llm_eval_framework import LLMEvaluator
from llm_eval_framework.config import MetricType
from llm_eval_framework.data_parsers import EvalSample

# Chuẩn bị data
samples = [
    EvalSample(
        id="1",
        input="The cat sat on the mat",  # Câu gốc (source)
        output="Con mèo ngồi trên thảm",  # Bản dịch của model
        reference="Chú mèo ngồi trên tấm thảm",  # Bản dịch chuẩn
    ),
    EvalSample(
        id="2",
        input="Hello world",
        output="Xin chào thế giới",
        reference="Chào thế giới",
    ),
]

# Đánh giá
evaluator = LLMEvaluator()
results = evaluator.evaluate(samples, metrics=[MetricType.BLEU])

print(f"BLEU Score: {results['bleu'].score:.4f}")
# Output: BLEU Score: 0.4532

# Xem chi tiết từng sample
for i, score in enumerate(results['bleu'].per_sample_scores):
    print(f"Sample {i+1}: {score:.4f}")
```

**Tùy chỉnh:**
```python
from llm_eval_framework.config import EvalConfig, MetricConfig

config = EvalConfig(
    metrics=MetricConfig(
        bleu_max_ngram=4,      # Tính đến 4-gram (mặc định)
        bleu_smoothing=True,   # Smoothing để tránh score = 0
    )
)
evaluator = LLMEvaluator(config)
```

---

### COMET (Crosslingual Optimized Metric for Evaluation)

**Mô tả:** Dùng model AI để đánh giá chất lượng dịch, hiểu được nghĩa.

**Khi nào dùng:**
- Đánh giá cuối cùng trước khi release
- Khi cần độ chính xác cao hơn BLEU
- Phát hiện lỗi nghĩa mà BLEU bỏ sót

**Yêu cầu:** `pip install unbabel-comet` và GPU

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="The cat sat on the mat",
        output="Con mèo ngồi trên thảm",
        reference="Chú mèo ngồi trên tấm thảm",
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.COMET])
print(f"COMET Score: {results['comet'].score:.4f}")
# COMET hiểu "con mèo" = "chú mèo" nên score cao hơn BLEU
```

---

## 2. Summarization Metrics

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Mô tả:** Đo độ trùng lặp giữa tóm tắt của model và tóm tắt chuẩn.

**Các loại:**
- `ROUGE-1`: Unigram (từ đơn)
- `ROUGE-2`: Bigram (cặp từ)
- `ROUGE-L`: Longest Common Subsequence

**Khi nào dùng:**
- So sánh nhanh các model tóm tắt
- Baseline đánh giá tóm tắt
- Không cần API, chạy offline

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="""Apple Inc. công bố kết quả kinh doanh quý 3 với doanh thu
        đạt 89.5 tỷ USD, tăng 5% so với cùng kỳ năm trước. CEO Tim Cook
        cho biết iPhone và dịch vụ tăng trưởng mạnh. Công ty cũng công bố
        chương trình mua lại cổ phiếu trị giá 90 tỷ USD.""",
        output="Apple đạt doanh thu 89.5 tỷ USD, iPhone bán chạy, mua lại 90 tỷ USD cổ phiếu.",
        reference="Doanh thu Apple Q3 đạt 89.5 tỷ USD, tăng 5%. iPhone và dịch vụ tăng trưởng tốt. Mua lại cổ phiếu 90 tỷ USD.",
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.ROUGE])

# Xem chi tiết các loại ROUGE
details = results['rouge'].details
print(f"ROUGE-1: {details['rouge_scores']['rouge1']:.4f}")
print(f"ROUGE-2: {details['rouge_scores']['rouge2']:.4f}")
print(f"ROUGE-L: {details['rouge_scores']['rougeL']:.4f}")
```

**Tùy chỉnh:**
```python
config = EvalConfig(
    metrics=MetricConfig(
        rouge_types=["rouge1", "rouge2", "rougeL"],  # Chọn loại ROUGE
    )
)
```

---

### G-Eval (LLM-as-Judge)

**Mô tả:** Dùng GPT-4 hoặc model mạnh làm giám khảo chấm điểm 1-5 theo nhiều tiêu chí.

**Tiêu chí đánh giá:**
- **Coherence**: Mạch lạc, dễ hiểu
- **Fluency**: Ngữ pháp, tự nhiên
- **Relevance**: Đúng trọng tâm
- **Consistency**: Không mâu thuẫn với nguồn

**Khi nào dùng:**
- Đánh giá chất lượng cao trước production
- Phát hiện hallucination
- Khi ROUGE không đủ (vì không hiểu nghĩa)

**Yêu cầu:** OpenAI API key

**Code:**
```python
from llm_eval_framework.config import OpenAIConfig

config = EvalConfig(
    openai=OpenAIConfig(
        api_key="sk-your-key",
        model="gpt-4o",
    )
)

evaluator = LLMEvaluator(config)

samples = [
    EvalSample(
        id="1",
        input="Bài báo về Apple...",
        output="Apple đạt doanh thu kỷ lục. CEO Tim Cook rất vui.",  # "rất vui" là bịa
        reference="Apple đạt 89.5 tỷ USD doanh thu.",
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.G_EVAL])

# Xem điểm từng tiêu chí
details = results['g_eval'].details
print(f"Coherence: {details['criteria_scores']['coherence']:.2f}")
print(f"Consistency: {details['criteria_scores']['consistency']:.2f}")  # Thấp vì bịa
```

---

## 3. QA Metrics

### Exact Match (EM)

**Mô tả:** Đúng hoàn toàn từng chữ = 1, sai = 0.

**Khi nào dùng:**
- QA với đáp án ngắn, rõ ràng
- Chatbot tra cứu thông tin
- Đánh giá nghiêm ngặt

**Code:**
```python
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
]

results = evaluator.evaluate(samples, metrics=[MetricType.EXACT_MATCH])
print(f"Exact Match: {results['exact_match'].score:.2%}")
# Output: Exact Match: 50.00% (1/2 đúng hoàn toàn)
```

**Tùy chỉnh:**
```python
from llm_eval_framework.metrics.qa import ExactMatchMetric

metric = ExactMatchMetric({
    "ignore_case": True,        # Không phân biệt hoa/thường
    "ignore_punctuation": True, # Bỏ qua dấu câu
    "ignore_articles": True,    # Bỏ qua a/an/the
})
```

---

### Token F1

**Mô tả:** Đo độ trùng lặp ở mức token, cho điểm từng phần đúng.

**Khi nào dùng:**
- Khi câu trả lời có thể dài hơn/ngắn hơn reference
- Cần đánh giá linh hoạt hơn EM
- Kết hợp với EM để có cái nhìn đầy đủ

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="Thủ đô Việt Nam là gì?",
        output="Thành phố Hà Nội",
        reference="Hà Nội",
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.TOKEN_F1])

details = results['token_f1'].details
print(f"F1: {details['f1']:.4f}")
print(f"Precision: {details['precision']:.4f}")
print(f"Recall: {details['recall']:.4f}")
# F1 cao vì "Hà Nội" có trong output
```

---

## 4. Ranking Metrics

### NDCG@K (Normalized Discounted Cumulative Gain)

**Mô tả:** Đánh giá chất lượng xếp hạng, item quan trọng ở đầu được thưởng nhiều hơn.

**Khi nào dùng:**
- Gợi ý sản phẩm, phim, bài hát
- Tìm kiếm có xếp hạng
- Retrieval trong RAG

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="Tìm phim hay",
        output="",
        # relevance_scores: điểm liên quan của mỗi item được trả về
        # Thứ tự: [item_1, item_2, item_3, ...]
        relevance_scores=[3, 2, 0, 1, 0],  # Item 1 liên quan nhất (3 điểm)
    ),
    EvalSample(
        id="2",
        input="Tìm phim hay",
        output="",
        relevance_scores=[0, 3, 2, 1, 0],  # Item tốt nhất ở vị trí 2 -> NDCG thấp hơn
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.NDCG])

details = results['ndcg'].details
print(f"NDCG@1: {details['ndcg_at_k']['ndcg@1']:.4f}")
print(f"NDCG@3: {details['ndcg_at_k']['ndcg@3']:.4f}")
print(f"NDCG@5: {details['ndcg_at_k']['ndcg@5']:.4f}")
```

---

### Recall@K

**Mô tả:** Trong top-K kết quả, tìm được bao nhiêu % item liên quan?

**Khi nào dùng:**
- Retrieval trong RAG (không được bỏ sót)
- Tìm kiếm y tế, pháp lý (bỏ sót là nguy hiểm)

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="Giá iPhone 16?",
        output="",
        # 1.0 = liên quan, 0.0 = không liên quan
        relevance_scores=[1.0, 0.0, 0.0, 1.0, 0.0],  # 2 docs liên quan ở vị trí 1 và 4
    ),
]

config = EvalConfig(
    metrics=MetricConfig(
        k_values=[1, 3, 5, 10],
        relevance_threshold=0.5,  # Score >= 0.5 được coi là liên quan
    )
)

evaluator = LLMEvaluator(config)
results = evaluator.evaluate(samples, metrics=[MetricType.RECALL_AT_K])

details = results['recall_at_k'].details
print(f"Recall@1: {details['recall_at_k']['recall@1']:.2%}")  # 50% (1/2)
print(f"Recall@5: {details['recall_at_k']['recall@5']:.2%}")  # 100% (2/2)
```

---

## 5. Tool Calling Metrics

### AST Accuracy

**Mô tả:** Kiểm tra model có gọi đúng function với đúng arguments không.

**Khi nào dùng:**
- Chatbot tích hợp API
- AI Agent tự động hóa
- Function calling evaluation

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="Thời tiết Hà Nội hôm nay thế nào?",
        output="",
        tool_calls=[
            {"name": "get_weather", "arguments": {"city": "Hà Nội"}}
        ],
        expected_tool_calls=[
            {"name": "get_weather", "arguments": {"city": "Hà Nội"}}
        ],
    ),
    EvalSample(
        id="2",
        input="Thời tiết Hà Nội?",
        output="",
        tool_calls=[
            {"name": "get_forecast", "arguments": {"location": "HN"}}  # Sai tên function
        ],
        expected_tool_calls=[
            {"name": "get_weather", "arguments": {"city": "Hà Nội"}}
        ],
    ),
    EvalSample(
        id="3",
        input="Kể chuyện cười về thời tiết",  # Không nên gọi tool
        output="Có một con mèo...",
        tool_calls=None,  # Model không gọi tool -> đúng!
        expected_tool_calls=None,
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.AST_ACCURACY])

details = results['ast_accuracy'].details
print(f"Full Match Accuracy: {details['full_match_accuracy']:.2%}")
print(f"Name Match Accuracy: {details['name_match_accuracy']:.2%}")
```

---

### Task Success Rate

**Mô tả:** Cả nhiệm vụ đa bước có hoàn thành không? Sai 1 bước = thất bại.

**Khi nào dùng:**
- AI Agent workflow phức tạp
- Multi-step tasks
- End-to-end evaluation

**Code:**
```python
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
            {"name": "book_flight", "arguments": {"from": "HN", "to": "HCM"}},
            {"name": "book_hotel", "arguments": {"city": "HCM"}},
        ],
    ),
    EvalSample(
        id="2",
        input="Đặt vé + khách sạn + gửi email xác nhận",
        output="",
        tool_calls=[
            {"name": "book_flight"},
            {"name": "book_hotel"},
            # Thiếu send_email -> Task Failed!
        ],
        expected_tool_calls=[
            {"name": "book_flight"},
            {"name": "book_hotel"},
            {"name": "send_email"},
        ],
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.TASK_SUCCESS_RATE])
print(f"Task Success Rate: {results['task_success_rate'].score:.2%}")
# 50% vì chỉ 1/2 task hoàn thành đầy đủ
```

---

## 6. Coding Metrics

### Pass@K

**Mô tả:** Code có chạy và pass tất cả test cases không?

**Khi nào dùng:**
- Evaluate AI coding assistant
- So sánh model cho task lập trình
- Code generation benchmark

**Code:**
```python
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
        input="Viết hàm tính giai thừa",
        output='''
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
''',
        test_cases=[
            {"test": "assert factorial(5) == 120"},
            {"test": "assert factorial(0) == 1"},
        ],
    ),
]

config = EvalConfig(
    metrics=MetricConfig(
        pass_k_values=[1, 5, 10],
        code_timeout=30,  # Timeout 30 giây
    )
)

evaluator = LLMEvaluator(config)
results = evaluator.evaluate(samples, metrics=[MetricType.PASS_AT_K])

details = results['pass_at_k'].details
print(f"Pass@1: {details['pass_at_k']['pass@1']:.2%}")
print(f"Correct: {details['correct_count']}/{details['total_samples']}")
```

---

## 7. Reasoning Metrics

### Accuracy (Multiple Choice)

**Mô tả:** Tỷ lệ trả lời đúng câu hỏi trắc nghiệm.

**Khi nào dùng:**
- Benchmark MMLU, MMLU-Pro
- Đánh giá kiến thức domain (y tế, luật, ...)
- So sánh tổng quát nhiều model

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="Hành tinh lớn nhất hệ Mặt Trời là gì?",
        output="Đáp án là B) Jupiter vì nó có khối lượng gấp 318 lần Trái Đất.",
        choices=["Mars", "Jupiter", "Saturn", "Neptune"],
        correct_answer="B",
    ),
    EvalSample(
        id="2",
        input="Nguyên tố có ký hiệu 'O' là gì?",
        output="A",  # Trả lời ngắn gọn
        choices=["Oxygen", "Gold", "Silver", "Iron"],
        correct_answer="A",
    ),
    EvalSample(
        id="3",
        input="1 + 1 = ?",
        output="Câu trả lời là C",
        choices=["1", "3", "2", "4"],
        correct_answer="C",
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.ACCURACY])

details = results['accuracy'].details
print(f"Accuracy: {details['accuracy_pct']:.1f}%")
print(f"Correct: {details['correct_count']}/{details['total_samples']}")
```

---

## 8. RAG Metrics

### Faithfulness (Độ trung thực)

**Mô tả:** Câu trả lời có dựa trên context không? Phát hiện hallucination.

**Quan trọng nhất cho RAG!**

**Khi nào dùng:**
- Bắt buộc với mọi hệ thống RAG
- Chatbot doanh nghiệp
- Tư vấn y tế, pháp lý, tài chính

**Code:**
```python
config = EvalConfig(
    openai=OpenAIConfig(api_key="sk-...", model="gpt-4o")
)

samples = [
    EvalSample(
        id="1",
        input="Doanh thu Tesla Q3 2024?",
        context="Tesla đạt doanh thu 25.2 tỷ USD trong Q3 2024.",
        output="Tesla đạt 25.2 tỷ USD. CEO Elon Musk gọi đây là kỷ lục lịch sử.",
        # ❌ "CEO gọi đây là kỷ lục" không có trong context -> BỊA!
    ),
    EvalSample(
        id="2",
        input="Doanh thu Tesla Q3 2024?",
        context="Tesla đạt doanh thu 25.2 tỷ USD trong Q3 2024, tăng 8% so với cùng kỳ.",
        output="Doanh thu Tesla Q3 2024 là 25.2 tỷ USD, tăng 8%.",
        # ✅ Tất cả thông tin đều có trong context
    ),
]

evaluator = LLMEvaluator(config)
results = evaluator.evaluate(samples, metrics=[MetricType.FAITHFULNESS])

for i, score in enumerate(results['faithfulness'].per_sample_scores):
    print(f"Sample {i+1} Faithfulness: {score:.2%}")
# Sample 1: ~50% (1 claim bịa)
# Sample 2: 100% (tất cả đúng)
```

---

### Context Precision & Recall

**Mô tả:**
- **Precision**: Trong context lấy về, bao nhiêu % thực sự có ích?
- **Recall**: Có bỏ sót thông tin quan trọng không?

**Khi nào dùng:**
- Debug hệ thống RAG
- Tối ưu số lượng chunks
- Xác định lỗi ở retrieval hay generation

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="Giá iPhone 16?",
        context="""
        Chunk 1: iPhone 16 có giá từ 799 USD.
        Chunk 2: iPhone 16 có camera 48MP.
        Chunk 3: So sánh iPhone vs Samsung.
        Chunk 4: Phụ kiện iPhone chính hãng.
        Chunk 5: iPhone 15 giá 699 USD.
        """,
        # Chỉ Chunk 1 có ích -> Precision thấp!
        output="iPhone 16 giá 799 USD",
        reference="799 USD",
    ),
]

results = evaluator.evaluate(samples, metrics=[
    MetricType.CONTEXT_PRECISION,
    MetricType.CONTEXT_RECALL,
])

print(f"Context Precision: {results['context_precision'].score:.2%}")
print(f"Context Recall: {results['context_recall'].score:.2%}")
```

---

### Answer Relevancy

**Mô tả:** Câu trả lời có liên quan đến câu hỏi không?

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="Thời tiết Hà Nội hôm nay?",
        output="Hà Nội là thủ đô của Việt Nam, có lịch sử ngàn năm...",
        # ❌ Không trả lời câu hỏi về thời tiết
    ),
    EvalSample(
        id="2",
        input="Thời tiết Hà Nội hôm nay?",
        output="Hà Nội hôm nay trời nắng, nhiệt độ 28-32°C.",
        # ✅ Trả lời đúng trọng tâm
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.ANSWER_RELEVANCY])
```

---

## 9. Safety Metrics

### FactScore

**Mô tả:** Chia câu trả lời thành từng sự kiện nhỏ, kiểm tra từng cái có đúng không.

**Khi nào dùng:**
- Chatbot y tế, pháp lý, báo chí
- Nơi sai sự thật gây hại nghiêm trọng
- Đánh giá chi tiết hallucination

**Code:**
```python
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

results = evaluator.evaluate(samples, metrics=[MetricType.FACTSCORE])

details = results['factscore'].details
print(f"FactScore: {results['factscore'].score:.2%}")
print(f"Correct facts: {details['total_correct']}")
print(f"Incorrect facts: {details['total_incorrect']}")
```

---

### IFEval (Instruction Following)

**Mô tả:** Kiểm tra model có làm đúng các yêu cầu cụ thể không (số câu, format, từ cấm...).

**Khi nào dùng:**
- Chatbot cần output JSON, markdown
- Hệ thống cần format chuẩn
- Đánh giá khả năng tuân theo quy tắc

**Code:**
```python
samples = [
    EvalSample(
        id="1",
        input="Viết về mùa xuân. Dùng đúng 3 câu. Không dùng từ 'hoa'. Kết thúc bằng dấu !",
        output="Mùa xuân ấm áp. Cây cối xanh tươi. Thật tuyệt vời!",
        # ✅ 3 câu ✅ không có 'hoa' ✅ kết thúc bằng !
    ),
    EvalSample(
        id="2",
        input="Viết về mùa xuân. Dùng đúng 3 câu. Không dùng từ 'hoa'.",
        output="Mùa xuân với những bông hoa đẹp. Tuyệt!",
        # ❌ chỉ 2 câu ❌ có từ 'hoa'
    ),
]

results = evaluator.evaluate(samples, metrics=[MetricType.IFEVAL])

details = results['ifeval'].details
print(f"IFEval Score: {results['ifeval'].score:.2%}")
print(f"Instructions followed: {details['total_followed']}/{details['total_instructions']}")
```

---

## 10. Chat Metrics

### Win Rate (LLM-as-Judge)

**Mô tả:** Dùng LLM so sánh output của model với baseline, đếm % thắng.

**Khi nào dùng:**
- So sánh model mới vs model cũ
- Quyết định model nào để deploy
- Tương tự AlpacaEval

**Code:**
```python
config = EvalConfig(
    openai=OpenAIConfig(api_key="sk-...", model="gpt-4o")
)

samples = [
    EvalSample(
        id="1",
        input="Giải thích AI là gì?",
        output="AI là trí tuệ nhân tạo, cho phép máy tính học và ra quyết định như con người...",
        reference="AI là Artificial Intelligence.",  # Baseline response
    ),
]

evaluator = LLMEvaluator(config)
results = evaluator.evaluate(samples, metrics=[MetricType.WIN_RATE])

details = results['win_rate'].details
print(f"Win Rate: {details['win_rate_with_ties']:.2%}")
print(f"Wins: {details['wins']}, Losses: {details['losses']}, Ties: {details['ties']}")
```

---

### Pairwise Comparison

**Mô tả:** So sánh trực tiếp 2 model trên cùng input.

**Code:**
```python
from llm_eval_framework.metrics.chat import PairwiseComparisonMetric

# Samples từ Model A
samples_a = [
    EvalSample(id="1", input="Explain AI", output="AI is artificial intelligence..."),
]

# Samples từ Model B (cùng input)
samples_b = [
    EvalSample(id="1", input="Explain AI", output="AI stands for Artificial Intelligence..."),
]

metric = PairwiseComparisonMetric(OpenAIConfig(api_key="sk-..."))
result = metric.compare_models(samples_a, samples_b)

print(f"Model A win rate: {result.details['a_win_rate']:.2%}")
print(f"Model B win rate: {result.details['b_win_rate']:.2%}")
```

---

## Tổng Hợp: Chọn Metric Theo Task

| Task | Metrics Khuyên Dùng |
|------|---------------------|
| **Dịch thuật** | BLEU (nhanh) → COMET (chính xác) |
| **Tóm tắt** | ROUGE (baseline) → G-Eval (production) |
| **QA ngắn** | Exact Match + Token F1 |
| **Tìm kiếm/Gợi ý** | NDCG@K, Recall@K, MRR |
| **Tool calling** | AST Accuracy |
| **AI Agent** | Task Success Rate |
| **Code generation** | Pass@K |
| **Trắc nghiệm** | Accuracy |
| **RAG** | Faithfulness (bắt buộc!) + Context Precision/Recall |
| **Chatbot** | Win Rate, G-Eval |
| **Safety** | FactScore, IFEval |

---

## Quick Reference Code

```python
from llm_eval_framework import LLMEvaluator, EvalConfig
from llm_eval_framework.config import OpenAIConfig, MetricType, TaskType

# 1. Config
config = EvalConfig(
    openai=OpenAIConfig(
        api_key="sk-...",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
    )
)

# 2. Load data
evaluator = LLMEvaluator(config)
samples = evaluator.load_data("data.json")

# 3. Generate GT if needed
samples = evaluator.generate_ground_truth(samples, task_type="qa")

# 4. Evaluate
results = evaluator.evaluate(
    samples,
    metrics=[MetricType.ROUGE, MetricType.FAITHFULNESS],
    task_type=TaskType.RAG,
)

# 5. Print & Save
print(evaluator.get_summary(results))
evaluator.save_results(results, "results.json")
```
