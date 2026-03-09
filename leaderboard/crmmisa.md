# `crmmisa` — CRM MISA — Phân tích kinh doanh (Business Analysis)

> **Task:** Phân tích dữ liệu kinh doanh từ CRM
> **Ground Truth model:** `claude-sonnet-4-5`
> **Samples:** 50 (ROUGE) / 20 (G-Eval, Answer Relevancy) per evaluation

---

## Kết quả Benchmark

| Metric | misa-ai-1.0-plus | mercury-2 | misa-ai-1.1 |
|--------|:-----------------:|:---------:|:-----------:|
| ROUGE | 30.65% ¹ / 30.70% ² | **31.72%** ¹ | **31.26%** ² |
| G-Eval | 50.00% | 50.00% ¹ | 50.00% ² |
| Answer Relevancy | **94.55%** ¹ / **94.40%** ² | 92.45% ¹ | **94.40%** ² |
| Avg Latency | ~11,922 ms | ~2,947 ms ¹ | **~306 ms** ² |

> ¹ Eval Run 1 — GT đợt 1 ([JSON](../examples/crm_evaluation_results/crmmisa_mercury2_comparison.json))
> ² Eval Run 2 — GT đợt 2 ([JSON](../examples/crm_evaluation_results_misaai/crmmisa_model_comparison.json))

---

## So sánh Latency

| Model | Avg Latency | So với baseline |
|-------|:-----------:|:---------------:|
| misa-ai-1.0-plus | ~11,922 ms | baseline |
| mercury-2 | ~2,947 ms | **-75.3%** |
| misa-ai-1.1 | **~306 ms** | **-97.4%** |

---

## Lưu ý

- **`misa-ai-1.0-plus` có 2 bộ số liệu** (¹ và ²) vì Ground Truth được generate lại bằng `claude-sonnet-4-5` ở mỗi đợt đánh giá. Chỉ nên so sánh các model cùng đợt (cùng superscript).
- `mercury-2` chỉ được đánh giá ở Run 1, `misa-ai-1.1` chỉ ở Run 2.
- ROUGE đo trên 50 samples; G-Eval và Answer Relevancy đo trên 20 samples (do chi phí LLM-as-judge).
- Task phân tích kinh doanh có output dài, nên ROUGE score tự nhiên thấp hơn so với `crmkh`.
