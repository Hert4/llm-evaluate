# Leaderboard — LLM Evaluation Benchmark

Tổng hợp kết quả benchmark các model trên từng dự án. Ground Truth: `claude-sonnet-4-5` | Date: 03/2026

## Bảng tổng hợp nhanh

| App | Task | Metric | misa-ai-1.0-plus | mercury-2 | misa-ai-1.1 |
|-----|------|--------|:-----------------:|:---------:|:-----------:|
| `crmkh` | Product Rec. | ROUGE | **42.95%** | 49.06% ¹ | 35.86% |
| `crmkh` | Product Rec. | Token F1 | **80.61%** | 78.66% ¹ | 64.81% |
| `crmkh` | Product Rec. | JSON Valid | **100%** | 98% ¹ | **100%** |
| `crmkh` | Product Rec. | Latency | ~2.0s | ~2.1s ¹ | **~0.5s** |
| `crmmisa` | Biz Analysis | ROUGE | 30.70% | **31.72%** ¹ | **31.26%** |
| `crmmisa` | Biz Analysis | G-Eval | 50.00% | 50.00% ¹ | 50.00% |
| `crmmisa` | Biz Analysis | Relevancy | **94.40%** | 92.45% ¹ | **94.40%** |
| `crmmisa` | Biz Analysis | Latency | ~11.9s | ~2.9s ¹ | **~0.3s** |

> ¹ mercury-2 đánh giá ở Run 1 (GT khác), các model khác ở Run 2. Chỉ nên so sánh cùng superscript. Xem chi tiết từng dự án bên dưới.

## Chi tiết từng dự án

| # | File | Dự án | Models |
|---|------|-------|:------:|
| 1 | [crmkh.md](crmkh.md) | AMIS CRM KH — Gợi ý sản phẩm | 3 |
| 2 | [crmmisa.md](crmmisa.md) | CRM MISA — Phân tích kinh doanh | 3 |

## Models đã đánh giá

| Model | Loại | Đánh giá trên |
|-------|------|:-------------:|
| `misa-ai-1.0-plus` | Baseline (production) | crmkh, crmmisa |
| `mercury-2` | Challenger | crmkh, crmmisa |
| `misa-ai-1.1` | Challenger | crmkh, crmmisa |

---

## Hướng dẫn thêm dự án mới

### 1. Tạo file chi tiết

Tạo file `leaderboard/<app_code>.md` theo template:

```markdown
# `<app_code>` — <Tên dự án> — <Task>

> **Task:** <Mô tả task>
> **Ground Truth model:** `claude-sonnet-4-5`
> **Samples:** N per evaluation

---

## Kết quả Benchmark

| Metric | model_a | model_b | model_c |
|--------|:-------:|:-------:|:-------:|
| ROUGE | xx.xx% | xx.xx% | xx.xx% |
| ... | ... | ... | ... |

> ¹ Eval Run 1 — [JSON](link)
> ² Eval Run 2 — [JSON](link)

---

## Lưu ý
- ...
```

### 2. Cập nhật bảng tổng hợp

Thêm hàng vào bảng ở file này và bảng trong `README.md`.

### 3. Chạy evaluation

```bash
# Chạy evaluation cho dự án mới
python -m llm_eval_framework.cli evaluate <data_file> --metrics rouge token_f1

# Hoặc dùng Python API
python examples/evaluate_crm_by_project.py
```

---

## Data Sources

| Dự án | Run 1 (mercury-2) | Run 2 (misa-ai-1.1) |
|-------|:------------------:|:--------------------:|
| crmkh | [JSON](../examples/crm_evaluation_results/crmkh_mercury2_comparison.json) | [JSON](../examples/crm_evaluation_results_misaai/crmkh_model_comparison.json) |
| crmmisa | [JSON](../examples/crm_evaluation_results/crmmisa_mercury2_comparison.json) | [JSON](../examples/crm_evaluation_results_misaai/crmmisa_model_comparison.json) |
| Summary Run 1 | [JSON](../examples/crm_evaluation_results/mercury2_comparison_summary.json) | — |
| Summary Run 2 | — | [JSON](../examples/crm_evaluation_results_misaai/comparison_summary.json) |
