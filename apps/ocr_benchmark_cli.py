#!/usr/bin/env python3
"""CLI to benchmark OCR engines (EasyOCR, PaddleOCR, Tesseract) on images.

Examples:
  pixi run python apps/ocr_benchmark_cli.py --glob "apps/tests/outputs/**/description_*_crop.png"
  pixi run python apps/ocr_benchmark_cli.py --dir apps/tests/outputs --lang en
  pixi run python apps/ocr_benchmark_cli.py --images path/a.png path/b.png --lang "en,ch_sim"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2

# Ensure repo root and apps/ are importable when executed directly
import sys as _sys
_THIS_DIR = Path(__file__).resolve().parent
_sys.path.insert(0, str(_THIS_DIR))
_sys.path.insert(0, str(_THIS_DIR.parent))

from apps.ocr.engines import available_engines
from apps.ocr.metrics import cer, wer
from apps.tests.test_utils import TestOutputManager


def find_images(args) -> List[Path]:
    images: list[Path] = []
    if args.images:
        images = [Path(p) for p in args.images]
    elif args.glob:
        images = [Path(p) for p in Path.cwd().glob(args.glob)]
    elif args.dir:
        base = Path(args.dir)
        images = [p for p in base.rglob("*.png")] + [p for p in base.rglob("*.jpg")]
    else:
        # Default: try known output locations from existing tests
        base = Path(__file__).parent / "tests" / "outputs"
        if base.exists():
            images = [p for p in base.rglob("*.png")] + [p for p in base.rglob("*.jpg")]
    return [p for p in images if p.exists()]


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


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR Benchmark CLI")
    parser.add_argument("--images", nargs="*", help="Specific image paths")
    parser.add_argument("--dir", help="Search directory for images")
    parser.add_argument("--glob", help="Glob relative to CWD to find images")
    parser.add_argument("--lang", default="en", help="Languages, e.g. 'en' or 'en,ch_sim'")
    parser.add_argument("--gt-json", help="Optional ground truth JSON {image: text}")
    parser.add_argument("--out", help="Output directory name", default="ocr_benchmark")
    args = parser.parse_args()

    images = find_images(args)
    if not images:
        print("No images found. Provide --images/--dir/--glob or run tests to generate crops.")
        return 1

    gt_map: Dict[str, str] = {}
    if args.gt_json and Path(args.gt_json).exists():
        gt_map = json.loads(Path(args.gt_json).read_text())

    engines = available_engines(args.lang)

    with TestOutputManager(args.out, keep_outputs=True) as mgr:
        out_dir = mgr.get_output_dir()
        summary_rows: list[dict[str, Any]] = []
        detailed: list[dict[str, Any]] = []

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            variants = build_variants(image)
            img_out = out_dir / f"{img_path.stem}"
            img_out.mkdir(exist_ok=True, parents=True)

            for vname, vimg in variants.items():
                cv2.imwrite(str(img_out / f"variant_{vname}.png"), vimg)

                for eng in engines:
                    res = eng.run(vimg).to_dict()
                    row = {
                        "image": str(img_path),
                        "variant": vname,
                        "engine": res["name"],
                        "init_time": res["init_time"],
                        "ocr_time": res["ocr_time"],
                        "total_time": res["total_time"],
                        "success": res["success"],
                        "text_len": len(res["text"]) if res["text"] else 0,
                    }

                    # Metrics if ground truth available
                    gt = gt_map.get(img_path.name)
                    if gt:
                        c, _, _ = cer(gt, res["text"] or "")
                        w, _, _ = wer(gt, res["text"] or "")
                        row.update({"cer": c, "wer": w})

                    summary_rows.append(row)
                    detailed.append({
                        **row,
                        "text": res["text"],
                        "raw": res.get("raw_results"),
                    })

        # Save CSV summary
        csv_file = out_dir / "summary.csv"
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        if fieldnames:
            with open(csv_file, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(summary_rows)

        # Save JSON detailed
        (out_dir / "details.json").write_text(
            json.dumps(detailed, indent=2, ensure_ascii=False)
        )

        print(f"Processed {len(images)} images. Results -> {out_dir}")
        if fieldnames:
            print(f"CSV: {csv_file}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
