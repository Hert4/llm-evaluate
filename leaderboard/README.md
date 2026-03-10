# Leaderboard — LLM Evaluation Benchmark

Tổng hợp kết quả benchmark các model trên từng dự án. Ground Truth: `claude-sonnet-4-5` | Date: 03/2026

> 📊 **[Interactive Dashboard (Chart.js)](../examples/crm/leaderboard/dashboard.html)** — Visual bar-chart comparison across all models & runs. Open `dashboard.html` in a browser.

## Bảng tổng hợp nhanh

| App | Task | Metric | misa-ai-1.0-plus | mercury-2 | misa-ai-1.1 | misa-text2sql | gpt-oss-120b |
|-----|------|--------|:-----------------:|:---------:|:-----------:|:-------------:|:------------:|
| `crmkh` | Product Rec. | ROUGE | **44.40%** ⁴ / **48.83%** ⁵ | 49.06% ¹ | 37.66% ⁴ | 41.87% ³ | 48.25% ⁵ |
| `crmkh` | Product Rec. | Token F1 | **77.96%** ⁴ / **80.74%** ⁵ | 78.66% ¹ | 63.88% ⁴ | 57.71% ³ | 78.09% ⁵ |
| `crmkh` | Product Rec. | JSON Valid | **100%** | 98% ¹ | **100%** | **100%** | **100%** ⁵ |
| `crmkh` | Product Rec. | Latency | ~2.0s | ~2.1s ¹ | ~2.2s ⁴ | ~1.4s ³ | ~3.2s ⁵ |
| `crmmisa` | Biz Analysis | ROUGE | 30.75% ⁴ / 30.83% ⁵ | 31.72% ¹ | **31.80%** ⁴ | 24.78% ³ | **31.26%** ⁵ |
| `crmmisa` | Biz Analysis | G-Eval | 50.00% | 50.00% ¹ | 50.00% ⁴ | 50.00% ³ | 50.00% ⁵ |
| `crmmisa` | Biz Analysis | Relevancy | **94.40%** ⁴ / 94.40% ⁵ | 92.45% ¹ | **94.40%** ⁴ | 79.25% ³ | **94.55%** ⁵ |
| `crmmisa` | Biz Analysis | Latency | ~11.9s | ~2.9s ¹ | **~1.5s** ⁴ | ~0.7s ³ | ~1.5s ⁵ |

> ¹ mercury-2 ở Run 1, ³ misa-text2sql ở Run 3, ⁴ misa-ai-1.1 re-eval ở Run 4, ⁵ gpt-oss-120b ở Run 5. Mỗi Run có GT riêng — chỉ nên so sánh cùng superscript. Xem chi tiết từng dự án bên dưới.

## Chi tiết từng dự án

| # | File | Dự án | Models |
|---|------|-------|:------:|
| 1 | [crmkh.md](crmkh.md) | AMIS CRM KH — Gợi ý sản phẩm | 5 |
| 2 | [crmmisa.md](crmmisa.md) | CRM MISA — Phân tích kinh doanh | 5 |

## Models đã đánh giá

| Model | Loại | Đánh giá trên |
|-------|------|:-------------:|
| `misa-ai-1.0-plus` | Baseline (production) | `crmkh`, `crmmisa` |
| `mercury-2` | Challenger | `crmkh`, `crmmisa` |
| `misa-ai-1.1` | Challenger | `crmkh`, ``crmmisa`` |
| `misa-text2sql` | Challenger | `crmkh`, `crmmisa` |
| `gpt-oss-120b` | Challenger | `crmkh`, `crmmisa` |

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

| Dự án | Run 1 (mercury-2) | Run 2 (misa-ai-1.1) | Run 4 (misa-ai-1.1 re-eval) | Run 5 (gpt-oss-120b) |
|-------|:------------------:|:--------------------:|:----------------------------:|:--------------------:|
| crmkh | [JSON](../examples/crm/crm_evaluation_results/crmkh_mercury2_comparison.json) | [JSON](../examples/crm/crm_evaluation_results_misaai/crmkh_model_comparison.json) | [JSON](../examples/crm/crm_evaluation_results_misaai11/crmkh_misaai11_comparison.json) | [JSON](../examples/crm/crm_evaluation_results_gptoss120b/crmkh_gptoss120b_comparison.json) |
| crmmisa | [JSON](../examples/crm/crm_evaluation_results/crmmisa_mercury2_comparison.json) | [JSON](../examples/crm/crm_evaluation_results_misaai/crmmisa_model_comparison.json) | [JSON](../examples/crm/crm_evaluation_results_misaai11/crmmisa_misaai11_comparison.json) | [JSON](../examples/crm/crm_evaluation_results_gptoss120b/crmmisa_gptoss120b_comparison.json) |
| Summary Run 1 | [JSON](../examples/crm/crm_evaluation_results/mercury2_comparison_summary.json) | — | — | — |
| Summary Run 2 | — | [JSON](../examples/crm/crm_evaluation_results_misaai/comparison_summary.json) | — | — |
| Summary Run 4 | — | — | [JSON](../examples/crm/crm_evaluation_results_misaai11/misaai11_comparison_summary.json) | — |
| Summary Run 5 | — | — | — | [JSON](../examples/crm/crm_evaluation_results_gptoss120b/gptoss120b_comparison_summary.json) |
