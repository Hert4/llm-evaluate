#!/usr/bin/env python3
"""
Generate a pure-CSS benchmark dashboard from evaluation summary JSONs.

Usage:
    python3 leaderboard/generate_dashboard.py

Reads 5 summary JSON files (Run 1–5), extracts per-model scores from the
latest run for each model, and generates leaderboard/dashboard.html.

Style: Meta-benchmark vertical bars — zero JS dependencies, pure CSS.
Dependencies: Python 3 stdlib only (json, pathlib, datetime)
"""

import json
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # leaderboard/
FRAMEWORK_ROOT = SCRIPT_DIR.parent                    # llm_eval_framework/

_DATA_ROOT_CANDIDATES = [
    FRAMEWORK_ROOT.parent / "llm-evaluate" / "examples" / "crm",
    FRAMEWORK_ROOT / "examples" / "crm",
]


def _find_data_root():
    for root in _DATA_ROOT_CANDIDATES:
        if (root / "crm_evaluation_results" / "mercury2_comparison_summary.json").exists():
            return root
    raise FileNotFoundError(
        "Cannot locate evaluation result JSONs. "
        "Searched: " + ", ".join(str(p) for p in _DATA_ROOT_CANDIDATES)
    )


DATA_ROOT = _find_data_root()
RESULTS_DIR = DATA_ROOT / "crm_evaluation_results"

RUN_FILES = {
    1: RESULTS_DIR / "mercury2_comparison_summary.json",
    2: DATA_ROOT / "crm_evaluation_results_misaai" / "comparison_summary.json",
    3: RESULTS_DIR / "text2sql_comparison_summary.json",
    4: DATA_ROOT / "crm_evaluation_results_misaai11" / "misaai11_comparison_summary.json",
    5: DATA_ROOT / "crm_evaluation_results_gptoss120b" / "gptoss120b_comparison_summary.json",
}

MODEL_RUN = {
    "mercury-2":        1,
    "misa-ai-1.1":      4,
    "misa-text2sql":    3,
    "gpt-oss-120b":     5,
    "misa-ai-1.0-plus": 5,
}

MODELS = ["misa-ai-1.0-plus", "mercury-2", "misa-ai-1.1", "misa-text2sql", "gpt-oss-120b"]

MODEL_LABELS = {
    "misa-ai-1.0-plus": "baseline",
    "mercury-2":        "mercury-2",
    "misa-ai-1.1":      "misa-ai-1.1",
    "misa-text2sql":    "text2sql",
    "gpt-oss-120b":     "gpt-oss-120b",
}

MODEL_COLORS = {
    "misa-ai-1.0-plus": "#3b82f6",   # blue
    "mercury-2":        "#14b8a6",   # teal
    "misa-ai-1.1":      "#22c55e",   # green
    "misa-text2sql":    "#f59e0b",   # amber
    "gpt-oss-120b":     "#ef4444",   # red
}

MODEL_BORDER_COLORS = {
    "misa-ai-1.0-plus": "#2563eb",
    "mercury-2":        "#0d9488",
    "misa-ai-1.1":      "#16a34a",
    "misa-text2sql":    "#d97706",
    "gpt-oss-120b":     "#dc2626",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def load_runs():
    runs = {}
    for num, path in RUN_FILES.items():
        with open(path, encoding="utf-8") as f:
            runs[num] = json.load(f)
    return runs


def find_project(run_data, app_code):
    for proj in run_data["projects"].values():
        if proj.get("application_code") == app_code:
            return proj
    return None


def get_framework_metric(proj, model, metric_name):
    key = f"framework_metrics_{model}"
    m = proj.get(key, {}).get(metric_name)
    return m["score"] if m else None


def get_product_metric(proj, model, metric_name):
    key = f"product_metrics_{model}"
    return proj.get(key, {}).get(metric_name)


def get_latency(proj, model):
    perf = proj.get("performance", {})
    ms_key = f"{model}_avg_ms"
    if ms_key in perf:
        return perf[ms_key] / 1000.0
    s_key = f"{model}_avg_s"
    if s_key in perf:
        return perf[s_key]
    return None


# ---------------------------------------------------------------------------
# Collect structured data for all charts
# ---------------------------------------------------------------------------

def collect_all(runs):
    """Return list of chart-group dicts, each with title + list of metrics."""
    groups = []

    # ── CRM KH ────────────────────────────────────────────────────────────
    kh_quality = []
    kh_product = []
    kh_latency = []

    for model in MODELS:
        rn = MODEL_RUN[model]
        proj = find_project(runs[rn], "crmkh")

        rouge = get_framework_metric(proj, model, "rouge")
        tf1   = get_framework_metric(proj, model, "token_f1")
        kh_quality.append({"model": model, "run": rn,
                           "ROUGE": round(rouge * 100, 2) if rouge else 0,
                           "Token F1": round(tf1 * 100, 2) if tf1 else 0})

        jv = get_product_metric(proj, model, "json_valid_rate")
        cv = get_product_metric(proj, model, "count_valid_rate")
        sp = get_product_metric(proj, model, "avg_valid_ratio")
        kh_product.append({"model": model, "run": rn,
                           "JSON Valid": round(jv * 100, 2) if jv else 0,
                           "Count Valid": round(cv * 100, 2) if cv else 0,
                           "Suggestion Precision": round(sp * 100, 2) if sp else 0})

        lat = get_latency(proj, model)
        kh_latency.append({"model": model, "run": rn,
                           "Latency": round(lat, 2) if lat else 0})

    groups.append({
        "section": "📦 CRM KH — Product Recommendation",
        "charts": [
            {"title": "Quality Metrics", "unit": "%", "max": 100, "metrics": ["ROUGE", "Token F1"], "data": kh_quality},
            {"title": "Product Metrics", "unit": "%", "max": 100, "metrics": ["JSON Valid", "Count Valid", "Suggestion Precision"], "data": kh_product},
            {"title": "Avg Latency", "unit": "s", "max": None, "metrics": ["Latency"], "data": kh_latency, "lower_better": True},
        ],
    })

    # ── CRM MISA ──────────────────────────────────────────────────────────
    misa_quality = []
    misa_latency = []

    for model in MODELS:
        rn = MODEL_RUN[model]
        proj = find_project(runs[rn], "crmmisa")

        rouge = get_framework_metric(proj, model, "rouge")
        geval = get_framework_metric(proj, model, "g_eval")
        arel  = get_framework_metric(proj, model, "answer_relevancy")
        misa_quality.append({"model": model, "run": rn,
                             "ROUGE": round(rouge * 100, 2) if rouge else 0,
                             "G-Eval": round(geval * 100, 2) if geval else 0,
                             "Ans. Relevancy": round(arel * 100, 2) if arel else 0})

        lat = get_latency(proj, model)
        misa_latency.append({"model": model, "run": rn,
                             "Latency": round(lat, 2) if lat else 0})

    groups.append({
        "section": "📊 CRM MISA — Business Analysis",
        "charts": [
            {"title": "Quality Metrics", "unit": "%", "max": 100, "metrics": ["ROUGE", "G-Eval", "Ans. Relevancy"], "data": misa_quality},
            {"title": "Avg Latency", "unit": "s", "max": None, "metrics": ["Latency"], "data": misa_latency, "lower_better": True},
        ],
    })

    return groups


# ---------------------------------------------------------------------------
# HTML generation — pure CSS vertical bars (Meta benchmark style)
# ---------------------------------------------------------------------------

def _bar_html(model, run, value, height_pct, unit, color, border_color, is_best):
    """Single vertical bar block."""
    label = MODEL_LABELS[model]
    display_val = f"{value}{unit}"
    best_cls = " best" if is_best else ""
    return (
        f'<div class="bar-item">'
        f'<div class="bar-value">{display_val}</div>'
        f'<div class="bar-track">'
        f'<div class="bar-fill{best_cls}" style="height:{height_pct:.1f}%;'
        f'background:{color};border-color:{border_color};" '
        f'data-value="{display_val}">'
        f'</div>'
        f'</div>'
        f'<div class="bar-label">{label}</div>'
        f'<div class="bar-run">R{run}</div>'
        f'</div>'
    )


def build_chart_html(chart):
    """Build one chart card with grouped metric sub-charts."""
    title = chart["title"]
    unit = chart["unit"]
    metrics = chart["metrics"]
    data = chart["data"]
    lower_better = chart.get("lower_better", False)
    forced_max = chart.get("max")

    sub_charts_html = []

    for metric in metrics:
        values = [d[metric] for d in data]
        max_val = forced_max if forced_max else (max(values) * 1.15 if values and max(values) > 0 else 100)

        # Determine best value
        if lower_better:
            positive_vals = [v for v in values if v > 0]
            best_val = min(positive_vals) if positive_vals else 0
        else:
            best_val = max(values) if values else 0

        bars_html = ""
        for d in data:
            v = d[metric]
            h = (v / max_val * 100) if max_val > 0 else 0
            h = min(h, 100)
            is_best = (v == best_val and v > 0)
            bars_html += _bar_html(
                d["model"], d["run"], v, h, unit,
                MODEL_COLORS[d["model"]], MODEL_BORDER_COLORS[d["model"]], is_best
            )

        metric_label = metric if len(metrics) > 1 else ""
        label_html = f'<div class="metric-label">{metric_label}</div>' if metric_label else ""

        sub_charts_html.append(
            f'<div class="metric-group">'
            f'{label_html}'
            f'<div class="bars-row">{bars_html}</div>'
            f'</div>'
        )

    lower_tag = ' <span class="lower-tag">▼ lower is better</span>' if lower_better else ""

    return (
        f'<div class="chart-card">'
        f'<h3>{title}{lower_tag}</h3>'
        f'{"".join(sub_charts_html)}'
        f'</div>'
    )


def build_html(groups, gen_date):
    # Legend
    legend_items = []
    for model in MODELS:
        rn = MODEL_RUN[model]
        c = MODEL_COLORS[model]
        lbl = MODEL_LABELS[model]
        legend_items.append(
            f'<span class="legend-item">'
            f'<span class="legend-dot" style="background:{c}"></span>'
            f'{lbl} (R{rn})</span>'
        )
    legend_html = "\n    ".join(legend_items)

    # Sections
    sections_html = ""
    for group in groups:
        charts = group["charts"]
        ncols = len(charts)
        grid_cls = f"chart-grid-{ncols}" if ncols <= 3 else "chart-grid-3"
        cards = "\n  ".join(build_chart_html(c) for c in charts)
        sections_html += (
            f'<h2 class="section-title">{group["section"]}</h2>\n'
            f'<div class="chart-grid {grid_cls}">\n  {cards}\n</div>\n\n'
        )

    return HTML_SHELL.replace("%%DATE%%", gen_date) \
                      .replace("%%LEGEND%%", legend_html) \
                      .replace("%%SECTIONS%%", sections_html)


HTML_SHELL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Benchmark Dashboard</title>
<style>
/* ── Dark theme tokens ──────────────────────────────────────────────── */
:root {
  --bg-900: #0f172a;
  --bg-800: #1e293b;
  --bg-700: #334155;
  --bg-600: #475569;
  --text-1: #f1f5f9;
  --text-2: #94a3b8;
  --text-3: #64748b;
}
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg-900);
  color: var(--text-1);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  padding: 32px 28px;
  min-height: 100vh;
}

/* ── Header ─────────────────────────────────────────────────────────── */
h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
.subtitle { color: var(--text-2); font-size: 0.9rem; margin: 4px 0 20px; }
.subtitle code {
  background: var(--bg-700); padding: 2px 6px; border-radius: 4px; font-size: 0.82rem;
}

/* ── Warning banner ─────────────────────────────────────────────────── */
.warning {
  background: rgba(245,158,11,0.1);
  border: 1px solid rgba(245,158,11,0.3);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 24px;
  font-size: 0.85rem;
  color: #fbbf24;
  display: flex; align-items: center; gap: 8px;
}

/* ── Legend ──────────────────────────────────────────────────────────── */
.legend {
  display: flex; flex-wrap: wrap; gap: 14px;
  margin-bottom: 28px;
  padding: 10px 14px;
  background: var(--bg-800);
  border-radius: 8px;
}
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.82rem; }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }

/* ── Section titles ─────────────────────────────────────────────────── */
.section-title {
  font-size: 1.2rem; font-weight: 600;
  margin: 8px 0 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--bg-700);
}

/* ── Chart grid ─────────────────────────────────────────────────────── */
.chart-grid { display: grid; gap: 18px; margin-bottom: 36px; }
.chart-grid-1 { grid-template-columns: 1fr; }
.chart-grid-2 { grid-template-columns: 1.6fr 1fr; }
.chart-grid-3 { grid-template-columns: 1fr 1fr 1fr; }

.chart-card {
  background: var(--bg-800);
  border: 1px solid var(--bg-700);
  border-radius: 10px;
  padding: 18px 16px 14px;
}
.chart-card h3 {
  font-size: 0.88rem; font-weight: 600;
  color: var(--text-2);
  margin-bottom: 14px;
}
.lower-tag {
  font-size: 0.72rem;
  color: var(--text-3);
  font-weight: 400;
  margin-left: 6px;
}

/* ── Metric sub-group ───────────────────────────────────────────────── */
.metric-group { margin-bottom: 18px; }
.metric-group:last-child { margin-bottom: 0; }
.metric-label {
  font-size: 0.75rem;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 8px;
  font-weight: 600;
}

/* ── Bar row (the Meta-style graph) ─────────────────────────────────── */
.bars-row {
  display: flex;
  align-items: flex-end;
  gap: 6px;
  height: 180px;
  padding-bottom: 0;
}

.bar-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  min-width: 0;
}

/* Score on top */
.bar-value {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--text-1);
  margin-bottom: 4px;
  white-space: nowrap;
  min-height: 16px;
}

/* Track (the grey background column) */
.bar-track {
  flex: 1;
  width: 100%;
  display: flex;
  align-items: flex-end;
  position: relative;
  border-radius: 4px 4px 0 0;
  background: var(--bg-700);
  overflow: hidden;
}

/* Fill (the coloured bar) */
.bar-fill {
  width: 100%;
  border-radius: 4px 4px 0 0;
  border-top: 3px solid;
  transition: height 1s cubic-bezier(0.22, 1, 0.36, 1);
  position: relative;
}
.bar-fill.best {
  box-shadow: 0 0 10px rgba(255,255,255,0.12);
}
.bar-fill.best::after {
  content: "★";
  position: absolute;
  top: -18px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 0.7rem;
  color: #fbbf24;
}

/* Model name below */
.bar-label {
  font-size: 0.68rem;
  color: var(--text-2);
  margin-top: 6px;
  text-align: center;
  line-height: 1.2;
  word-break: break-word;
  max-width: 100%;
}

/* Run tag */
.bar-run {
  font-size: 0.62rem;
  color: var(--text-3);
  margin-top: 1px;
}

/* ── Footer ─────────────────────────────────────────────────────────── */
footer {
  margin-top: 40px;
  padding-top: 14px;
  border-top: 1px solid var(--bg-700);
  color: var(--text-3);
  font-size: 0.75rem;
  text-align: center;
}

/* ── Responsive ─────────────────────────────────────────────────────── */
@media (max-width: 960px) {
  .chart-grid-3, .chart-grid-2 { grid-template-columns: 1fr; }
  .bars-row { height: 150px; }
}
</style>
</head>
<body>

<h1>LLM Benchmark Dashboard</h1>
<p class="subtitle">Ground Truth: <code>claude-sonnet-4-5</code> &middot; 50 samples / project &middot; Generated %%DATE%%</p>

<div class="warning">
  <span>⚠️</span>
  <span><b>Lưu ý:</b> Mỗi Evaluation Run có Ground Truth riêng — chỉ nên so sánh cùng Run (xem nhãn R1–R5).</span>
</div>

<div class="legend">
    %%LEGEND%%
</div>

%%SECTIONS%%

<footer>Auto-generated by <code>generate_dashboard.py</code> &middot; Pure CSS — no JavaScript dependencies</footer>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading evaluation runs …")
    runs = load_runs()

    print("Collecting data …")
    groups = collect_all(runs)

    print("Generating HTML …")
    html = build_html(groups, date.today().isoformat())

    out_path = SCRIPT_DIR / "dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard written to {out_path}")
    print(f"   Open in browser: file://{out_path}")


if __name__ == "__main__":
    main()