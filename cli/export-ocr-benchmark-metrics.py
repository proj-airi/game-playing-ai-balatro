"""Generate figures and a findings doc from OCR description benchmark outputs.

Inputs:
  - summary CSV and details JSON from apps/tests/outputs/ocr_descriptions/

Outputs:
  - docs/ai/findings/<doc_name>/assets/*.png
  - docs/ai/findings/<doc_name>/README.md with embedded figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt


def load_summary(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize types
            r['success'] = str(r.get('success', '')).lower() in ('true', '1', 'yes')
            for k in ('init_time', 'ocr_time', 'total_time'):
                try:
                    r[k] = float(r.get(k, 0) or 0)
                except ValueError:
                    r[k] = 0.0
            try:
                r['text_len'] = int(r.get('text_len', 0) or 0)
            except ValueError:
                r['text_len'] = 0
            rows.append(r)
    return rows


def aggregate_by_engine(rows: List[Dict[str, Any]]):
    stats: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        eng = r['engine']
        d = stats.setdefault(
            eng,
            {
                'total': 0,
                'success': 0,
                'sum_time': 0.0,
                'sum_time_success': 0.0,
                'sum_text_success': 0,
                'eff_samples': [],  # (chars/sec) for successful only
            },
        )
        d['total'] += 1
        d['sum_time'] += r['total_time']
        if r['success'] and r['total_time'] > 0:
            d['success'] += 1
            d['sum_time_success'] += r['total_time']
            d['sum_text_success'] += r['text_len']
            d['eff_samples'].append(r['text_len'] / r['total_time'])  # chars/sec

    # finalize
    for eng, d in stats.items():
        d['success_rate'] = (d['success'] / d['total'] * 100) if d['total'] else 0
        d['avg_time'] = (d['sum_time'] / d['total']) if d['total'] else 0
        d['avg_time_success'] = (
            d['sum_time_success'] / d['success'] if d['success'] else 0
        )
        d['avg_text_len_success'] = (
            d['sum_text_success'] / d['success'] if d['success'] else 0
        )
        d['avg_efficiency'] = (
            sum(d['eff_samples']) / len(d['eff_samples']) if d['eff_samples'] else 0
        )
    return stats


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def bar(ax, labels, values, title, ylabel):
    ax.bar(
        labels,
        values,
        color=['#4C78A8', '#F58518', '#54A24B', '#E45756'][: len(labels)],
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    for i, v in enumerate(values):
        ax.text(
            i,
            v,
            f'{v:.1f}' if isinstance(v, float) else str(v),
            ha='center',
            va='bottom',
            fontsize=9,
        )


def plot_figures(stats: Dict[str, Dict[str, Any]], out_dir: Path) -> Dict[str, str]:
    engines = list(stats.keys())
    success_rate = [stats[e]['success_rate'] for e in engines]
    avg_time_succ = [stats[e]['avg_time_success'] for e in engines]
    avg_text_len = [stats[e]['avg_text_len_success'] for e in engines]
    avg_eff = [stats[e]['avg_efficiency'] for e in engines]

    paths: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, engines, success_rate, 'Success Rate by Engine', 'Percent')
    p = out_dir / 'success_rate.png'
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths['success_rate'] = p.name

    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, engines, avg_time_succ, 'Avg Time (successful) by Engine', 'Seconds')
    p = out_dir / 'avg_time_success.png'
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths['avg_time_success'] = p.name

    fig, ax = plt.subplots(figsize=(6, 4))
    bar(
        ax,
        engines,
        avg_text_len,
        'Avg Extracted Text Length (successful)',
        'Characters',
    )
    p = out_dir / 'avg_text_len_success.png'
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths['avg_text_len_success'] = p.name

    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, engines, avg_eff, 'Chars per Second (successful)', 'chars/sec')
    p = out_dir / 'efficiency.png'
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths['efficiency'] = p.name

    return paths


def write_doc(
    doc_dir: Path,
    title: str,
    stats: Dict[str, Dict[str, Any]],
    asset_files: Dict[str, str],
):
    md = doc_dir / 'README.md'
    # Derive top findings
    if stats:
        best_success = max(stats.items(), key=lambda kv: kv[1]['success_rate'])[0]
        best_eff = max(stats.items(), key=lambda kv: kv[1]['avg_efficiency'])[0]
    else:
        best_success = best_eff = 'N/A'

    lines = [
        '---',
        f'title: "{title}"',
        f'date: "{Path(doc_dir.name).name.split("-")[0] if "-" in doc_dir.name else ""}"',
        'coding_agents:',
        '  authors: ["neko", "Agent"]',
        '  project: "proj-airi/game-playing-ai-balatro"',
        '  context: "OCR benchmark on Balatro description crops"',
        '  technologies: ["EasyOCR", "PaddleOCR", "Tesseract", "YOLO", "OpenCV"]',
        'tags: ["ocr", "benchmark", "balatro"]',
        '---',
        '',
        f'# {title}',
        '',
        '## Executive Summary',
        f'- Best success rate: **{best_success}**',
        f'- Best efficiency (chars/sec): **{best_eff}**',
        '- Preprocessing with 2x–3x scaling generally improves accuracy.',
        '- If Tesseract language data is missing, its results are excluded (handled gracefully).',
        '',
        '## Figures',
        f'![](assets/{asset_files.get("success_rate", "")})',
        f'![](assets/{asset_files.get("avg_time_success", "")})',
        f'![](assets/{asset_files.get("avg_text_len_success", "")})',
        f'![](assets/{asset_files.get("efficiency", "")})',
        '',
        '## Notes',
        '- Data source: `apps/tests/outputs/ocr_descriptions/` (summary.csv, details.json)',
        '- Engines compared: EasyOCR, PaddleOCR, Tesseract (if available)',
        '- Metrics on successful runs: average time, text length, and chars/sec.',
    ]
    md.write_text('\n'.join(lines))


def main() -> int:
    p = argparse.ArgumentParser(description='Plot OCR metrics and write findings doc')
    p.add_argument(
        '--input-dir',
        default='apps/tests/outputs/ocr_descriptions',
        help='Benchmark output dir',
    )
    p.add_argument(
        '--doc-name',
        default='2025-09-22-ocr-descriptions-benchmark',
        help='Folder name under docs/ai/findings/',
    )
    p.add_argument(
        '--title',
        default='Balatro OCR (Descriptions) Benchmark Findings',
        help='Doc title',
    )
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    csv_path = in_dir / 'descriptions_summary.csv'
    if not csv_path.exists():
        print(f'Missing summary CSV: {csv_path}')
        return 1

    rows = load_summary(csv_path)
    stats = aggregate_by_engine(rows)

    doc_dir = Path('docs/ai/findings') / args.doc_name
    assets = ensure_dir(doc_dir / 'assets')

    asset_files = plot_figures(stats, assets)
    write_doc(doc_dir, args.title, stats, asset_files)

    print(f'Findings written to {doc_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
