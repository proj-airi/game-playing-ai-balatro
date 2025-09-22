#!/usr/bin/env python3
"""Pipeline: detect card_description only → crop → OCR → optional LLM judge.

Why: Some scenes contain two descriptions with no close card pairing (e.g., out_00166). We avoid
pairing assumptions and focus solely on `*description*` detections.

Usage examples:
  pixi run python apps/pipelines/ocr_descriptions_pipeline.py --targets out_00104.jpg out_00166.jpg out_00114.jpg
  pixi run python apps/pipelines/ocr_descriptions_pipeline.py --lang "en,ch_sim" --judge-model openrouter/auto
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2

import sys as _sys
_THIS = Path(__file__).resolve().parent  # apps/pipelines
_APPS = _THIS.parent                     # apps/
_REPO = _APPS.parent                     # repo root
# Make both apps/ and repo root importable
_sys.path.insert(0, str(_APPS))
_sys.path.insert(0, str(_REPO))

from core.yolo_detector import YOLODetector
from config.settings import settings
from utils.logger import get_logger

from ocr_benchmark_cli import build_variants
from ocr.engines import available_engines
from ocr.llm_judge import LlmJudge
from tests.test_utils import TestOutputManager


logger = get_logger(__name__)


def default_images_dir() -> Path:
    classes_file = Path(settings.model_classes_file).resolve()
    return classes_file.parent / "images"


def resolve_targets(images_dir: Path, targets: List[str] | None) -> List[Path]:
    if not targets:
        # Sensible defaults confirmed to contain descriptions
        targets = ["out_00104.jpg", "out_00166.jpg", "out_00114.jpg"]
    return [images_dir / t for t in targets]


def run(images_dir: Path, targets: List[str] | None, lang: str, judge_model: str | None, out_name: str) -> int:
    paths = [p for p in resolve_targets(images_dir, targets) if p.exists()]
    if not paths:
        print("No target images found. Check --images-dir or filenames.")
        return 1

    detector = YOLODetector()
    engines = available_engines(lang)
    judge = LlmJudge(judge_model) if judge_model else None

    with TestOutputManager(out_name, keep_outputs=True) as mgr:
        out_dir = mgr.get_output_dir()
        summary_rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []

        for img_path in paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            detections = detector.detect(image)
            descs = [d for d in detections if "description" in d.class_name.lower()]
            if not descs:
                print(f"No descriptions in {img_path.name}")
                continue

            img_out = out_dir / img_path.stem
            img_out.mkdir(parents=True, exist_ok=True)

            for idx, d in enumerate(descs):
                crop = image[d.bbox[1]:d.bbox[3], d.bbox[0]:d.bbox[2]]
                crop_file = img_out / f"desc_{idx:02d}.png"
                cv2.imwrite(str(crop_file), crop)

                variants = build_variants(crop)
                candidate_texts: list[str] = []

                for vname, vimg in variants.items():
                    cv2.imwrite(str(img_out / f"desc_{idx:02d}_{vname}.png"), vimg)
                    for eng in engines:
                        res = eng.run(vimg).to_dict()
                        candidate_texts.append(res.get("text") or "")
                        row = {
                            "image": str(img_path),
                            "crop": str(crop_file),
                            "variant": vname,
                            "engine": res["name"],
                            "text_len": len(res["text"]) if res["text"] else 0,
                            "success": res["success"],
                            "total_time": res["total_time"],
                        }
                        summary_rows.append(row)
                        details.append({**row, "text": res.get("text", "")})

                # Optional LLM judging across candidates
                if judge is not None:
                    judge_res = judge.score(candidate_texts, context=f"Balatro card description from {img_path.name}")
                    (img_out / f"desc_{idx:02d}_judge.json").write_text(json.dumps(judge_res, indent=2, ensure_ascii=False))

        # Save summaries
        if summary_rows:
            csv_file = out_dir / "descriptions_summary.csv"
            with open(csv_file, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)
        (out_dir / "descriptions_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False))

        print(f"Description-only OCR results -> {out_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Detect description-only and OCR them")
    p.add_argument("--images-dir", help="Directory containing target images")
    p.add_argument("--targets", nargs="*", help="Specific filenames (e.g., out_00104.jpg)")
    p.add_argument("--lang", default="en", help="OCR languages, e.g., 'en' or 'en,ch_sim'")
    p.add_argument("--judge-model", help="OpenRouter model id for LLM judge (optional)")
    p.add_argument("--out", default="ocr_descriptions", help="Output folder name")
    args = p.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else default_images_dir()
    return run(images_dir, args.targets, args.lang, args.judge_model, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
