#!/usr/bin/env python3
"""Export Balatro card-corner crops from a YOLO dataset."""

import argparse
import sys
from pathlib import Path

_cli_dir = Path(__file__).parent
_src_dir = _cli_dir.parent / 'src'
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ai_balatro.datasets.card_corners import (
    CardCornerExportOptions,
    export_card_corner_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Export card-corner crops from YOLO card-front annotations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--images-dir', required=True, type=Path)
    parser.add_argument('--labels-dir', required=True, type=Path)
    parser.add_argument('--classes-file', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--class-name', default='poker_card_front')
    parser.add_argument('--corner-width-ratio', default=0.36, type=float)
    parser.add_argument('--corner-height-ratio', default=0.46, type=float)
    parser.add_argument('--padding', default=0, type=int)
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument(
        '--min-light-ratio',
        default=None,
        type=float,
        help='Drop crops whose white/light pixel ratio is below this threshold',
    )
    parser.add_argument(
        '--min-ink-ratio',
        default=0.04,
        type=float,
        help='Drop crops with too little rank/suit-like dark or saturated ink',
    )
    parser.add_argument('--write-csv', action='store_true')
    args = parser.parse_args()

    options = CardCornerExportOptions(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        classes_file=args.classes_file,
        out_dir=args.out_dir,
        class_name=args.class_name,
        corner_width_ratio=args.corner_width_ratio,
        corner_height_ratio=args.corner_height_ratio,
        padding=args.padding,
        limit=args.limit,
        min_light_ratio=args.min_light_ratio,
        min_ink_ratio=args.min_ink_ratio,
        write_csv=args.write_csv,
    )
    result = export_card_corner_dataset(options)

    print(f'Exported {result.sample_count} card-corner crops')
    print(f'  crops: {result.crops_dir}')
    print(f'  manifest: {result.manifest_path}')
    if result.csv_path is not None:
        print(f'  csv: {result.csv_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
