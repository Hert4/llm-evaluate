# `crmmisa` — CRM MISA — Phân tích kinh doanh (Business Analysis)

> **Task:** Phân tích dữ liệu kinh doanh từ CRM
> **Ground Truth model:** `claude-sonnet-4-5`
> **Samples:** 50 (ROUGE) / 20 (G-Eval, Answer Relevancy) per evaluation

---

## Kết quả Benchmark

| Metric | misa-ai-1.0-plus | mercury-2 | misa-ai-1.1 | misa-text2sql | gpt-oss-120b |
|--------|:-----------------:|:---------:|:-----------:|:-------------:|:------------:|
| ROUGE | 30.65% ¹ / 30.70% ² / 30.75% ⁴ / 30.83% ⁵ | **31.72%** ¹ | **31.26%** ² / **31.80%** ⁴ | 24.78% ³ | **31.26%** ⁵ |
| G-Eval | 50.00% | 50.00% ¹ | 50.00% ² ⁴ | 50.00% ³ | 50.00% ⁵ |
| Answer Relevancy | **94.55%** ¹ / **94.40%** ² ⁴ / 94.40% ⁵ | 92.45% ¹ | **94.40%** ² ⁴ | 79.25% ³ | **94.55%** ⁵ |
| Avg Latency | ~11,922 ms | ~2,947 ms ¹ | ~306 ms ² / ~1,528 ms ⁴ | ~650 ms ³ | **~1,540 ms** ⁵ |

> ¹ Eval Run 1 — GT đợt 1 ([JSON](../examples/crm/crm_evaluation_results/crmmisa_mercury2_comparison.json))
> ² Eval Run 2 — GT đợt 2 ([JSON](../examples/crm/crm_evaluation_results_misaai/crmmisa_model_comparison.json))
> ³ Eval Run 3 — misa-text2sql ([JSON](../examples/crm/crm_evaluation_results/crmmisa_text2sql_comparison.json))
> ⁴ Eval Run 4 — GT đợt 4, misa-ai-1.1 re-eval ([JSON](../examples/crm/crm_evaluation_results_misaai11/crmmisa_misaai11_comparison.json))
> ⁵ Eval Run 5 — GT đợt 5, gpt-oss-120b ([JSON](../examples/crm/crm_evaluation_results_gptoss120b/crmmisa_gptoss120b_comparison.json))

---

## So sánh Latency

| Model | Avg Latency | So với baseline |
|-------|:-----------:|:---------------:|
| misa-ai-1.0-plus | ~11,922 ms | baseline |
| mercury-2 | ~2,947 ms | **-75.3%** |
| misa-ai-1.1 | **~306 ms** ² / ~1,528 ms ⁴ | **-97.4%** ² / **-87.2%** ⁴ |
| misa-text2sql | ~650 ms | **-94.5%** |
| gpt-oss-120b | ~1,540 ms ⁵ | **-87.1%** ⁵ |

---

## Lưu ý

- **`misa-ai-1.0-plus` có nhiều bộ số liệu** (¹, ², ⁴, ⁵) vì Ground Truth được generate lại bằng `claude-sonnet-4-5` ở mỗi đợt đánh giá. Chỉ nên so sánh các model cùng đợt (cùng superscript).
- `mercury-2` chỉ được đánh giá ở Run 1, `misa-text2sql` ở Run 3.
- `misa-ai-1.1` được đánh giá ở Run 2 và **Run 4** (re-eval với GT mới).
- `gpt-oss-120b` được đánh giá ở **Run 5**.
- ROUGE đo trên 50 samples; G-Eval và Answer Relevancy đo trên 20 samples (do chi phí LLM-as-judge).
- Task phân tích kinh doanh có output dài, nên ROUGE score tự nhiên thấp hơn so với `crmkh`.
- Run 5: `gpt-oss-120b` thắng baseline trên cả ROUGE (31.26% vs 30.83%) và Answer Relevancy (94.55% vs 94.40%), latency nhanh hơn ~87% (~1.5s vs ~11.9s).
