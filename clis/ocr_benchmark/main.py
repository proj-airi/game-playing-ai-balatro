"""CLI to benchmark OCR engines (PaddleOCR, Tesseract, RapidOCR) on images.

Examples:
  pixi run python -m clis.ocr_benchmark.main --images-dir data/datasets/games-balatro-2024-entities-detection/data/train/yolo/images/ --targets out_00104.jpg out_00166.jpg out_00114.jpg --lang "en,ch_sim"
  pixi run python -m clis.ocr_benchmark.main --images-dir data/datasets/games-balatro-2024-entities-detection/data/train/yolo/images/ --targets out_00104.jpg out_00166.jpg out_00114.jpg --lang "en,ch_sim" --judge-model "google/gemini-2.5-flash"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, List

import cv2

# Local imports (ensure repo root is importable)
import sys as _sys
_THIS_DIR = Path(__file__).resolve().parent   # clis/ocr_benchmark/
_CLIS = _THIS_DIR.parent                      # clis/
_REPO = _CLIS.parent                          # repo root
_APPS = _REPO / "apps"
_sys.path.insert(0, str(_APPS))
_sys.path.insert(0, str(_CLIS))
_sys.path.insert(0, str(_REPO))

from apps.core.yolo_detector import YOLODetector
from apps.config.settings import settings
from apps.utils.logger import get_logger
from apps.ocr.engines import available_engines
from apps.ocr.llm_judge import LlmJudge
from apps.tests.test_utils import TestOutputManager

logger = get_logger(__name__)

def build_variants(image):
    """Generate OCR-friendly variants (safe set).

    Rationale:
    - Binary/adaptive thresholding often harms small UI fonts; drop by default.
    - Prefer scaling, grayscale, CLAHE, and light sharpening/denoise.
    """
    variants: dict[str, Any] = {"original": image}

    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    variants["grayscale"] = gray

    # CLAHE contrast boost on grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants["clahe"] = enhanced

    # Unsharp mask (light sharpening) on CLAHE result
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpen = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    variants["sharpen"] = sharpen

    # Gentle denoise (bilateral) on grayscale 2x later
    # We'll build scaled versions and then apply bilateral on 2x grayscale

    # Scale factors (apply after base ops)
    variants_scaled = {}
    for name, img in variants.items():
        for sf in (1, 2, 3):
            if sf == 1:
                variants_scaled[name] = img
            else:
                scaled = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_CUBIC)
                variants_scaled[f"{name}_{sf}x"] = scaled

    # Add denoised grayscale_2x variant
    if "grayscale_2x" in variants_scaled:
        denoised = cv2.bilateralFilter(variants_scaled["grayscale_2x"], d=5, sigmaColor=60, sigmaSpace=60)
        variants_scaled["grayscale_2x_denoise"] = denoised

    return variants_scaled

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
                        logger.info(f"OCR {img_path.name} desc#{idx} variant={vname} with {eng.name}")
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
