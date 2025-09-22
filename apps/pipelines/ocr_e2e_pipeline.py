#!/usr/bin/env python3
"""End-to-end pipeline: YOLO → tooltip/card crops → OCR benchmark.

- Loads YOLO model from configured paths (apps/config/config.yaml)
- Detects tooltips and cards; pairs them; writes crops
- Benchmarks OCR engines on the crops and saves CSV/JSON summaries

Usage examples:
  pixi run python apps/pipelines/ocr_e2e_pipeline.py --limit 10
  pixi run python apps/pipelines/ocr_e2e_pipeline.py --images-dir ../data/datasets/.../images --lang "en,ch_sim"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2

# Local imports (ensure repo root is importable)
import sys as _sys
_THIS_DIR = Path(__file__).resolve().parent   # apps/pipelines
_APPS = _THIS_DIR.parent                      # apps/
_REPO = _APPS.parent                          # repo root
_sys.path.insert(0, str(_APPS))
_sys.path.insert(0, str(_REPO))

from core.yolo_detector import YOLODetector
from services.card_tooltip_service import CardTooltipService
from utils.logger import get_logger
from config.settings import settings

from ocr.engines import available_engines
from ocr_benchmark_cli import build_variants  # reuse
from tests.test_utils import TestOutputManager


logger = get_logger(__name__)


def default_images_dir() -> Path:
    # Derive from classes.txt path: .../yolo/classes.txt -> .../yolo/images
    classes_file = Path(settings.model_classes_file).resolve()
    yolo_dir = classes_file.parent
    images_dir = yolo_dir / "images"
    return images_dir


def iter_images(base: Path) -> List[Path]:
    return sorted(list(base.glob("*.png")) + list(base.glob("*.jpg")) + list(base.glob("*.jpeg")))


def run_pipeline(images_dir: Path, out_name: str, limit: int | None, lang: str) -> int:
    logger.info(f"Images dir: {images_dir}")
    imgs = iter_images(images_dir)
    if not imgs:
        print(f"No images in {images_dir}")
        return 1
    if limit:
        imgs = imgs[:limit]
    print(f"Processing {len(imgs)} images…")

    # Initialize components
    detector = YOLODetector()
    tooltip_service = CardTooltipService()
    engines = available_engines(lang)

    with TestOutputManager(out_name, keep_outputs=True) as mgr:
        out_dir = mgr.get_output_dir()
        crops_dir = out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []
        detailed_rows: list[dict[str, Any]] = []

        for img_idx, img_path in enumerate(imgs):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            detections = detector.detect(image)
            matches = tooltip_service.match_tooltips_to_cards(detections)
            crops = tooltip_service.create_tooltip_card_crops(image, matches)

            img_out = crops_dir / f"{img_idx:03d}_{img_path.stem}"
            img_out.mkdir(exist_ok=True)

            # Save crops and benchmark each
            def _save_and_bench(arr, tag):
                crop_file = img_out / f"{tag}.png"
                cv2.imwrite(str(crop_file), arr)
                variants = build_variants(arr)
                for vname, vimg in variants.items():
                    cv2.imwrite(str(img_out / f"{tag}_variant_{vname}.png"), vimg)
                    for eng in engines:
                        res = eng.run(vimg).to_dict()
                        row = {
                            "scene_image": str(img_path),
                            "crop_file": str(crop_file),
                            "crop_tag": tag,
                            "variant": vname,
                            "engine": res["name"],
                            "init_time": res["init_time"],
                            "ocr_time": res["ocr_time"],
                            "total_time": res["total_time"],
                            "success": res["success"],
                            "text_len": len(res["text"]) if res["text"] else 0,
                        }
                        summary_rows.append(row)
                        detailed_rows.append({**row, "text": res["text"], "raw": res.get("raw_results")})

            for m_idx, m in enumerate(crops):
                _save_and_bench(m["tooltip_crop"], f"tooltip_{m_idx:02d}")
                _save_and_bench(m["context_crop"], f"context_{m_idx:02d}")

        # Persist summaries
        csv_file = out_dir / "e2e_summary.csv"
        if summary_rows:
            with open(csv_file, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)
        (out_dir / "e2e_details.json").write_text(json.dumps(detailed_rows, indent=2, ensure_ascii=False))

        print(f"End-to-end results -> {out_dir}")
        print(f"CSV: {csv_file}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="YOLO→OCR end-to-end pipeline")
    p.add_argument("--images-dir", help="Directory of source images")
    p.add_argument("--limit", type=int, help="Limit number of images")
    p.add_argument("--lang", default="en", help="OCR languages, e.g. 'en' or 'en,ch_sim'")
    p.add_argument("--out", default="e2e_ocr_benchmark", help="Output folder name")
    args = p.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else default_images_dir()
    return run_pipeline(images_dir, args.out, args.limit, args.lang)


if __name__ == "__main__":
    raise SystemExit(main())
