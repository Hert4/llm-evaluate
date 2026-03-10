# `crmkh` — AMIS CRM KH — Gợi ý sản phẩm (Product Recommendation)

> **Task:** Gợi ý sản phẩm cho khách hàng CRM
> **Ground Truth model:** `claude-sonnet-4-5`
> **Samples:** 50 per evaluation

---

## Kết quả Benchmark

| Metric | misa-ai-1.0-plus | mercury-2 | misa-ai-1.1 | misa-text2sql | gpt-oss-120b |
|--------|:-----------------:|:---------:|:-----------:|:-------------:|:------------:|
| ROUGE | **49.22%** ¹ / **42.95%** ² / **44.40%** ⁴ / **48.83%** ⁵ | 49.06% ¹ | 35.86% ² / 37.66% ⁴ | 41.87% ³ | 48.25% ⁵ |
| Token F1 | **81.96%** ¹ / **80.61%** ² / **77.96%** ⁴ / **80.74%** ⁵ | 78.66% ¹ | 64.81% ² / 63.88% ⁴ | 57.71% ³ | 78.09% ⁵ |
| JSON Valid Rate | **100%** | 98% ¹ | **100%** ² ⁴ | **100%** ³ | **100%** ⁵ |
| Count Valid (1-5) | 100% ² ⁴ ⁵ | — | 100% ⁴ | **100%** ³ | 100% ⁵ |
| SP Hợp Lệ | **99.20%** ² ⁴ ⁵ | — | 98.40% ⁴ | **100%** ³ | 99.20% ⁵ |
| Avg Gợi Ý | 4.9 ² ⁴ | — | **5.0** ⁴ | **5.0** ³ | **5.0** ⁵ |
| Avg Latency | **~2,028 ms** | ~2,104 ms ¹ | **~498 ms** ² / ~2,161 ms ⁴ | ~1,430 ms ³ | ~3,220 ms ⁵ |

> ¹ Eval Run 1 — GT đợt 1 ([JSON](../examples/crm/crm_evaluation_results/crmkh_mercury2_comparison.json))
> ² Eval Run 2 — GT đợt 2 ([JSON](../examples/crm/crm_evaluation_results_misaai/crmkh_model_comparison.json))
> ³ Eval Run 3 — misa-text2sql ([JSON](../examples/crm/crm_evaluation_results/crmkh_text2sql_comparison.json))
> ⁴ Eval Run 4 — GT đợt 4, misa-ai-1.1 re-eval ([JSON](../examples/crm/crm_evaluation_results_misaai11/crmkh_misaai11_comparison.json))
> ⁵ Eval Run 5 — GT đợt 5, gpt-oss-120b ([JSON](../examples/crm/crm_evaluation_results_gptoss120b/crmkh_gptoss120b_comparison.json))

---

## So sánh Latency

| Model | Avg Latency | So với baseline |
|-------|:-----------:|:---------------:|
| misa-ai-1.0-plus | ~2,028 ms | baseline |
| mercury-2 | ~2,104 ms | +3.7% |
| misa-ai-1.1 | **~498 ms** ² / ~2,161 ms ⁴ | **-75.4%** ² / +6.6% ⁴ |
| misa-text2sql | ~1,430 ms | **-29.5%** |
| gpt-oss-120b | ~3,220 ms ⁵ | +58.8% ⁵ |

---

## Lưu ý

- **`misa-ai-1.0-plus` có nhiều bộ số liệu** (¹, ², ⁴, ⁵) vì Ground Truth được generate lại bằng `claude-sonnet-4-5` ở mỗi đợt đánh giá. Chỉ nên so sánh các model cùng đợt (cùng superscript).
- `mercury-2` chỉ được đánh giá ở Run 1, `misa-text2sql` ở Run 3.
- `misa-ai-1.1` được đánh giá ở Run 2 và **Run 4** (re-eval với GT mới).
- `gpt-oss-120b` được đánh giá ở **Run 5**.
- Metric ROUGE và Token F1 đo trên toàn bộ 50 samples.
- JSON Valid Rate đo tỉ lệ output parse được thành JSON hợp lệ.
- Run 4 bổ sung các metric đặc thù Product Recommendation cho `misa-ai-1.1`: Count Valid (1-5), SP Hợp Lệ, Avg Gợi Ý.
- Run 5: `gpt-oss-120b` đạt chất lượng gần baseline (ROUGE 48.25% vs 48.83%, Token F1 78.09% vs 80.74%) nhưng latency cao hơn (~3.2s vs ~2.0s).
