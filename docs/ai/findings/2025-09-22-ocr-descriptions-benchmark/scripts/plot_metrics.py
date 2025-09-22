#!/usr/bin/env python3
"""Render figures and update README for the OCR descriptions benchmark.

Reads:    apps/tests/outputs/ocr_descriptions/descriptions_summary.csv
Writes:   ../assets/*.png and ../README.md

Run from any directory:
  python docs/ai/findings/2025-09-22-ocr-descriptions-benchmark/scripts/plot_metrics.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, List
import json
import matplotlib.pyplot as plt


THIS = Path(__file__).resolve()
DOC_DIR = THIS.parent.parent
ASSETS = DOC_DIR / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

CSV_PATH = "apps/tests/outputs/ocr_descriptions/descriptions_summary.csv"

def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(CSV_PATH, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["success"] = str(row.get("success", "")).lower() in ("true", "1", "yes")
            for k in ("init_time", "ocr_time", "total_time"):
                try:
                    row[k] = float(row.get(k, 0) or 0)
                except ValueError:
                    row[k] = 0.0
            try:
                row["text_len"] = int(row.get("text_len", 0) or 0)
            except ValueError:
                row["text_len"] = 0
            rows.append(row)
    return rows


def agg(rows: List[Dict[str, Any]]):
    stats: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        eng = r["engine"]
        d = stats.setdefault(eng, {"total":0, "success":0, "sum_time":0.0, "sum_time_s":0.0, "sum_text":0, "eff": []})
        d["total"] += 1
        d["sum_time"] += r["total_time"]
        if r["success"] and r["total_time"] > 0:
            d["success"] += 1
            d["sum_time_s"] += r["total_time"]
            d["sum_text"] += r["text_len"]
            d["eff"].append(r["text_len"] / r["total_time"])  # chars/sec
    # finalize
    for eng, d in stats.items():
        d["success_rate"] = (d["success"]/d["total"]*100) if d["total"] else 0
        d["avg_time_success"] = (d["sum_time_s"]/d["success"]) if d["success"] else 0
        d["avg_text_len_success"] = (d["sum_text"]/d["success"]) if d["success"] else 0
        d["avg_efficiency"] = (sum(d["eff"])/len(d["eff"])) if d["eff"] else 0
    return stats


def bar(ax, labels, values, title, ylabel):
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][:len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}" if isinstance(v, float) else str(v), ha="center", va="bottom", fontsize=9)


def plot(stats: Dict[str, Dict[str, Any]]):
    engines = list(stats.keys())
    if not engines:
        return {}
    sr = [stats[e]["success_rate"] for e in engines]
    t = [stats[e]["avg_time_success"] for e in engines]
    txt = [stats[e]["avg_text_len_success"] for e in engines]
    eff = [stats[e]["avg_efficiency"] for e in engines]

    out = {}
    import matplotlib
    matplotlib.rcParams.update({"figure.dpi": 160})

    fig, ax = plt.subplots(figsize=(6,4)); bar(ax, engines, sr, "Success Rate by Engine", "Percent"); fig.tight_layout();
    p = ASSETS/"success_rate.png"; fig.savefig(p); plt.close(fig); out["success_rate"] = p.name

    fig, ax = plt.subplots(figsize=(6,4)); bar(ax, engines, t, "Avg Time (successful) by Engine", "Seconds"); fig.tight_layout();
    p = ASSETS/"avg_time_success.png"; fig.savefig(p); plt.close(fig); out["avg_time_success"] = p.name

    fig, ax = plt.subplots(figsize=(6,4)); bar(ax, engines, txt, "Avg Extracted Text Length (successful)", "Characters"); fig.tight_layout();
    p = ASSETS/"avg_text_len_success.png"; fig.savefig(p); plt.close(fig); out["avg_text_len_success"] = p.name

    fig, ax = plt.subplots(figsize=(6,4)); bar(ax, engines, eff, "Chars per Second (successful)", "chars/sec"); fig.tight_layout();
    p = ASSETS/"efficiency.png"; fig.savefig(p); plt.close(fig); out["efficiency"] = p.name

    return out


def write_readme(stats: Dict[str, Dict[str, Any]], assets: Dict[str, str]):
    readme = DOC_DIR / "README.md"
    best_success = max(stats.items(), key=lambda kv: kv[1]["success_rate"])[0] if stats else "N/A"
    best_eff = max(stats.items(), key=lambda kv: kv[1]["avg_efficiency"])[0] if stats else "N/A"
    frontmatter = {
        "title": "Balatro OCR (Descriptions) Benchmark Findings",
        "date": "2025-09-22",
        "coding_agents": {
            "authors": ["neko", "Agent"],
            "project": "proj-airi/game-playing-ai-balatro",
            "context": "Benchmark OCR engines on YOLO description crops",
            "technologies": ["YOLO", "EasyOCR", "PaddleOCR", "Tesseract", "OpenCV"],
        },
        "tags": ["ocr", "benchmark", "balatro"],
    }
    doc = [
        "---",
        json.dumps(frontmatter, indent=2),
        "---",
        "",
        "# Balatro OCR (Descriptions) Benchmark Findings",
        "",
        "## Executive Summary",
        f"- Best success rate: **{best_success}**",
        f"- Best efficiency (chars/sec): **{best_eff}**",
        "- 2x–3x scaling generally improves OCR accuracy for UI text.",
        "- Tesseract requires language packs (eng, chi_sim) to participate fully.",
        "",
        "## Figures",
        f"![](assets/{assets.get('success_rate','')})",
        f"![](assets/{assets.get('avg_time_success','')})",
        f"![](assets/{assets.get('avg_text_len_success','')})",
        f"![](assets/{assets.get('efficiency','')})",
        "",
        "## Method",
        "- Inputs: crops from description detections in out_00104, out_00166, out_00114.",
        "- Engines: EasyOCR, PaddleOCR, Tesseract (mapped langs: en→eng, ch_sim→chi_sim).",
        "- Variants: original/grayscale/binary/adaptive/enhanced × scale 1x/2x/3x.",
        "- Metrics (successful only): success rate, avg time, extracted text length, chars/sec.",
    ]
    readme.write_text("\n".join(doc))


def main() -> int:
    rows = load_rows()
    stats = agg(rows)
    imgs = plot(stats)
    write_readme(stats, imgs)
    print(f"Wrote figures to {ASSETS} and README.md to {DOC_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

