#!/usr/bin/env python3
"""Export the Balatro card-corner CNN classifier checkpoint to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_cli_dir = Path(__file__).parent
_repo_root = _cli_dir.parent
_src_dir = _repo_root / 'src'
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ai_balatro.datasets.card_corner_classifier import CardCornerClassifier  # noqa: E402

DEFAULT_OUTPUT = (
    _repo_root
    / 'models'
    / 'games-balatro-2024-card-corner-classifier'
    / 'onnx'
    / 'model.onnx'
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Export card-corner rank/suit CNN checkpoint to ONNX',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument('--image-size', default=None, type=int)
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=args.image_size,
    )
    print(f'wrote ONNX model: {args.output}')
    return 0


def export_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
    image_size: int | None = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = CardCornerClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    resolved_image_size = int(image_size or checkpoint.get('image_size', 64))
    dummy = torch.zeros(1, 3, resolved_image_size, resolved_image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['images'],
        output_names=['rank_logits', 'suit_logits'],
        dynamic_axes={
            'images': {0: 'batch'},
            'rank_logits': {0: 'batch'},
            'suit_logits': {0: 'batch'},
        },
        opset_version=17,
    )


if __name__ == '__main__':
    raise SystemExit(main())
