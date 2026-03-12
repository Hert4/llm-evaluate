# Generate Ground Truth — Hướng dẫn sử dụng

Script `scripts/generate_ground_truth.py` sinh **reference** (ground truth) cho các benchmark dataset bằng GPT-5.2 API.

## Tổng quan

```
Benchmark JSON (input + output, reference=null)
        │
        ▼
 ┌──────────────────────┐
 │  generate_ground_truth│──► GPT-5.2 API
 │  .py                  │◄── response
 └──────────────────────┘
        │
        ▼
Benchmark JSON (input + output + reference)
```

**Input**: Benchmark JSON đã có `input` + `output` (từ model production), chưa có `reference`.

**Output**: Ghi đè lại **chính file benchmark gốc**, thêm field `reference` vào mỗi sample.

## File liên quan

| File | Vai trò |
|------|---------|
| `scripts/generate_ground_truth.py` | Script chính |
| `scripts/tasks/*.yaml` | Config cho từng task (chứa `gt_prompt`, `gt_task_type`, `gt_metrics`) |
| `data/benchmarks/*.json` | Benchmark files — đọc vào, ghi đè ra |
| `data/benchmarks/*_backup.json` | Backup tự động trước khi ghi đè |

## Dữ liệu lưu ở đâu?

### Benchmark files (ghi đè tại chỗ)

```
data/benchmarks/
├── crm_recommendation_150.json        ← ghi đè, thêm reference
├── crm_recommendation_150_backup.json ← backup tự động
├── crm_intent_analysis_57.json
├── crm_intent_analysis_57_backup.json
├── crmmisa_dashboard_150.json
├── crmmisa_dashboard_150_backup.json
├── makt_forecast_150.json
├── makt_forecast_150_backup.json
├── mtrans_translation_150.json
└── mtrans_translation_150_backup.json
```

- **File gốc** bị ghi đè — mỗi sample có thêm `"reference": "..."`.
- **Backup** tự động tạo trước khi ghi đè: `{tên_file}_backup.json`.
- Nếu chạy lại, sample đã có reference sẽ **bị bỏ qua** (không gọi API lại).

### Metadata được thêm vào benchmark

Sau khi chạy, phần `metadata` trong benchmark JSON sẽ có thêm:

```json
{
  "metadata": {
    "gt_generated_date": "2026-03-12 14:30:00",
    "gt_model": "gpt-5.2",
    "gt_script": "generate_ground_truth.py",
    "gt_task_type": "qa",
    "gt_metrics": ["exact_match", "token_f1"],
    "samples_with_reference": 150,
    "samples_total": 150
  }
}
```

### Checkpoint (trong quá trình chạy)

Mỗi **20 samples** script tự lưu checkpoint vào file benchmark. Nếu crash giữa chừng:
- Samples đã sinh reference → **đã được lưu**.
- Chạy lại script → chỉ sinh cho samples **chưa có** reference.

## Cách sử dụng

### 1. Xem danh sách tasks đã cấu hình

```bash
python scripts/generate_ground_truth.py --list-tasks
```

Output:
```
Tasks with GT configuration:
────────────────────────────────────────────────────────
  crm_intent_analysis      type=qa          metrics=['exact_match', 'token_f1']
  crm_recommendation       type=qa          metrics=['exact_match', 'token_f1']
  crmmisa_dashboard        type=rag         metrics=['token_f1', 'faithfulness']
  makt_forecast            type=qa          metrics=['exact_match', 'token_f1']
  mtrans_translation       type=translation metrics=['bleu']
```

### 2. Preview prompts (dry-run)

Xem prompt sẽ gửi cho GPT-5.2 **mà không gọi API**:

```bash
# Preview 3 samples đầu tiên của 1 task
python scripts/generate_ground_truth.py --task crm_recommendation --dry-run --limit 3

# Preview tất cả tasks
python scripts/generate_ground_truth.py --dry-run --limit 2
```

### 3. Sinh GT cho 1 task cụ thể

```bash
python scripts/generate_ground_truth.py --task crm_recommendation
```

### 4. Sinh GT cho nhiều tasks

```bash
python scripts/generate_ground_truth.py --task crm_recommendation --task makt_forecast
```

### 5. Sinh GT cho tất cả tasks

```bash
python scripts/generate_ground_truth.py
```

### 6. Giới hạn số samples (test trước)

```bash
# Chỉ sinh 5 samples đầu tiên
python scripts/generate_ground_truth.py --task crm_recommendation --limit 5
```

### 7. Điều chỉnh concurrency

```bash
# Giảm xuống 3 concurrent calls (nếu API chậm/rate limit)
python scripts/generate_ground_truth.py --concurrency 3
```

### 8. Bỏ qua backup

```bash
python scripts/generate_ground_truth.py --skip-backup
```

## Tất cả CLI options

```
--task TASK          Task cần chạy (lặp lại được). Mặc định: tất cả.
--dry-run            Chỉ hiện prompt, không gọi API.
--limit N            Chỉ xử lý N samples đầu/task (để test).
--concurrency N      Số API calls đồng thời tối đa (mặc định: 5).
--benchmarks-dir DIR Thư mục benchmarks (mặc định: data/benchmarks/).
--tasks-dir DIR      Thư mục YAML tasks (mặc định: scripts/tasks/).
--list-tasks         Liệt kê tasks có GT config rồi thoát.
--skip-backup        Không tạo backup trước khi ghi đè.
```

## Cấu hình trong YAML

Mỗi task YAML cần thêm 3 fields:

```yaml
# scripts/tasks/crm_recommendation.yaml

gt_prompt: |
  Bạn là chuyên gia tư vấn bán hàng. Dựa vào lịch sử đơn hàng:
  {input}
  Trả lời CHÍNH XÁC theo format JSON:
  [{{"ProductCode": "..."}}, ...]
  Chỉ trả JSON, không giải thích.

gt_task_type: qa          # qa | translation | rag

gt_metrics:               # Metrics dùng để evaluate sau này
  - exact_match
  - token_f1
```

### Placeholders trong gt_prompt

| Placeholder | Giá trị | Nguồn |
|------------|---------|-------|
| `{input}` | Nội dung input của sample | `sample["input"]` |
| `{context}` | System prompt / context | `sample["context"]` (có thể null → "") |

### Task types và metrics khuyến nghị

| Module | gt_task_type | gt_metrics | Lý do |
|--------|-------------|------------|-------|
| crm_recommendation | qa | exact_match, token_f1 | So sánh set ProductCode |
| crm_intent_analysis | qa | exact_match, token_f1 | Classification output |
| crmmisa_dashboard | rag | token_f1, faithfulness | Văn bản phân tích + kiểm tra bịa số liệu |
| makt_forecast | qa | exact_match, token_f1 | So sánh product codes |
| mtrans_translation | translation | bleu | Chuẩn dịch thuật |

## API Configuration

Hardcoded trong script (dòng 60-62):

```python
API_URL = "https://numerous-catch-uploaded-compile.trycloudflare.com/v1/chat/completions"
API_KEY = "misa_misa_00t07fh7_ZFRMf6rOUaVHTv6CZH0uOzAx_LDP1IeWM"
MODEL = "gpt-5.2"
```

Các tham số khác:

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `DEFAULT_CONCURRENCY` | 5 | Max concurrent API calls |
| `DEFAULT_MAX_COMPLETION_TOKENS` | 4096 | Max tokens cho response |
| `DEFAULT_TEMPERATURE` | 0.0 | Deterministic output |
| `DEFAULT_TIMEOUT` | 120s | Timeout mỗi request |
| `MAX_RETRIES` | 5 | Số lần retry khi fail |
| `RETRY_DELAY_BASE` | 2 | Exponential backoff: 2s, 4s, 8s, 16s, 32s |
| `CHECKPOINT_INTERVAL` | 20 | Lưu checkpoint mỗi 20 samples |

## Verify kết quả

### Kiểm tra số lượng reference

```bash
python -c "
import json
for f in ['crm_recommendation_150', 'crm_intent_analysis_57', 'crmmisa_dashboard_150', 'makt_forecast_150', 'mtrans_translation_150']:
    data = json.load(open(f'data/benchmarks/{f}.json'))
    refs = sum(1 for s in data['data'] if s.get('reference'))
    print(f'{f}: {refs}/{len(data[\"data\"])} have reference')
"
```

### Chạy evaluation với framework

```python
from llm_eval_framework import LLMEvaluator
from llm_eval_framework.config import MetricType

evaluator = LLMEvaluator()
samples = evaluator.load_data("data/benchmarks/crm_recommendation_150.json")
results = evaluator.evaluate(samples, metrics=[MetricType.EXACT_MATCH, MetricType.TOKEN_F1])
print(evaluator.get_summary(results))
```

## Quy trình khuyến nghị

```bash
# Bước 1: Preview prompts trước
python scripts/generate_ground_truth.py --task crm_recommendation --dry-run --limit 3

# Bước 2: Test thật với ít samples
python scripts/generate_ground_truth.py --task crm_recommendation --limit 5

# Bước 3: Verify kết quả
python -c "
import json
data = json.load(open('data/benchmarks/crm_recommendation_150.json'))
for s in data['data'][:5]:
    if s.get('reference'):
        print(f'{s[\"id\"]}: {s[\"reference\"][:100]}...')
"

# Bước 4: Chạy full cho tất cả tasks
python scripts/generate_ground_truth.py

# Bước 5: Chạy evaluation
python -m llm_eval_framework evaluate data/benchmarks/crm_recommendation_150.json
```

## Xử lý lỗi

| Tình huống | Script xử lý |
|-----------|--------------|
| API timeout | Retry tối đa 5 lần, exponential backoff |
| Rate limit (429) | Retry với delay tăng dần |
| Server error (5xx) | Retry với delay tăng dần |
| Crash giữa chừng | Checkpoint đã lưu, chạy lại sẽ skip samples đã có reference |
| Muốn khôi phục file gốc | Copy `*_backup.json` đè lại file chính |
