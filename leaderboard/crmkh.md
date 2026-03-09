# `crmkh` — AMIS CRM KH — Gợi ý sản phẩm (Product Recommendation)

> **Task:** Gợi ý sản phẩm cho khách hàng CRM
> **Ground Truth model:** `claude-sonnet-4-5`
> **Samples:** 50 per evaluation

---

## Kết quả Benchmark

| Metric | misa-ai-1.0-plus | mercury-2 | misa-ai-1.1 |
|--------|:-----------------:|:---------:|:-----------:|
| ROUGE | **49.22%** ¹ / **42.95%** ² | 49.06% ¹ | 35.86% ² |
| Token F1 | **81.96%** ¹ / **80.61%** ² | 78.66% ¹ | 64.81% ² |
| JSON Valid Rate | **100%** | 98% ¹ | **100%** ² |
| Avg Latency | ~2,028 ms | ~2,104 ms ¹ | **~498 ms** ² |

> ¹ Eval Run 1 — GT đợt 1 ([JSON](../examples/crm_evaluation_results/crmkh_mercury2_comparison.json))
> ² Eval Run 2 — GT đợt 2 ([JSON](../examples/crm_evaluation_results_misaai/crmkh_model_comparison.json))

---

## So sánh Latency

| Model | Avg Latency | So với baseline |
|-------|:-----------:|:---------------:|
| misa-ai-1.0-plus | ~2,028 ms | baseline |
| mercury-2 | ~2,104 ms | +3.7% |
| misa-ai-1.1 | **~498 ms** | **-75.4%** |

---

## Lưu ý

- **`misa-ai-1.0-plus` có 2 bộ số liệu** (¹ và ²) vì Ground Truth được generate lại bằng `claude-sonnet-4-5` ở mỗi đợt đánh giá. Chỉ nên so sánh các model cùng đợt (cùng superscript).
- `mercury-2` chỉ được đánh giá ở Run 1, `misa-ai-1.1` chỉ ở Run 2.
- Metric ROUGE và Token F1 đo trên toàn bộ 50 samples.
- JSON Valid Rate đo tỉ lệ output parse được thành JSON hợp lệ.
