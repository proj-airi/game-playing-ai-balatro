"""Tests for the card-corner rank/suit classifier."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import onnx
import torch
from PIL import Image

from ai_balatro.datasets.card_corner_classifier import (
    CardCornerClassifier,
    CardCornerDataset,
    RANK_TO_INDEX,
    SUIT_TO_INDEX,
    class_weights,
    load_labeled_rows,
    trim_to_ink,
)


def test_load_labeled_rows_keeps_readable_rows_with_rank_and_suit(tmp_path):
    manifest = tmp_path / 'labeled.jsonl'
    manifest.write_text(
        '\n'.join(
            [
                json.dumps(
                    {
                        'image': 'crops/readable.png',
                        'rank': 'A',
                        'suit': 'spades',
                        'quality': 'readable',
                    }
                ),
                json.dumps(
                    {
                        'image': 'crops/unreadable.png',
                        'rank': 'K',
                        'suit': 'clubs',
                        'quality': 'unreadable',
                    }
                ),
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    rows = load_labeled_rows(manifest)

    assert len(rows) == 1
    assert rows[0].rank == 'A'
    assert rows[0].suit == 'spades'


def test_card_corner_dataset_loads_image_and_encodes_labels(tmp_path):
    crops_dir = tmp_path / 'crops'
    crops_dir.mkdir()
    Image.new('RGB', (12, 8), 'white').save(crops_dir / 'ace.png')
    manifest = tmp_path / 'labeled.jsonl'
    manifest.write_text(
        json.dumps(
            {
                'image': 'crops/ace.png',
                'rank': 'A',
                'suit': 'hearts',
                'quality': 'readable',
            }
        )
        + '\n',
        encoding='utf-8',
    )

    dataset = CardCornerDataset(
        rows=load_labeled_rows(manifest),
        image_root=tmp_path,
        image_size=32,
    )

    image, rank, suit = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert image.min() >= 0
    assert image.max() <= 1
    assert rank.item() == RANK_TO_INDEX['A']
    assert suit.item() == SUIT_TO_INDEX['hearts']


def test_card_corner_classifier_predicts_rank_and_suit_logits():
    model = CardCornerClassifier()
    images = torch.rand(2, 3, 64, 64)

    rank_logits, suit_logits = model(images)

    assert rank_logits.shape == (2, 13)
    assert suit_logits.shape == (2, 4)


def test_train_script_default_output_dir_uses_card_corner_cnn_runs():
    module = load_cli_module('train-card-corner-classifier.py')
    args = Namespace(
        out_dir=None,
        run_name=None,
        device='mps',
        learning_rate=3e-4,
        seed=11,
        augment=True,
        trim_ink=False,
        weighted_loss=True,
    )

    out_dir = module._resolve_output_dir(args)

    assert out_dir.parts[-3:] == (
        'classify',
        'card-corner-cnn',
        'mps-lr0p0003-seed11-aug-weighted',
    )


def test_export_script_writes_card_corner_cnn_onnx_signature(tmp_path):
    module = load_cli_module('export-card-corner-classifier-onnx.py')
    checkpoint = tmp_path / 'card-corner-classifier-best.pt'
    output = tmp_path / 'model.onnx'
    torch.save(
        {
            'model_state_dict': CardCornerClassifier().state_dict(),
            'image_size': 64,
        },
        checkpoint,
    )

    module.export_onnx(checkpoint_path=checkpoint, output_path=output)

    graph = onnx.load(output).graph
    assert [input.name for input in graph.input] == ['images']
    assert [output.name for output in graph.output] == [
        'rank_logits',
        'suit_logits',
    ]


def test_class_weights_are_inverse_frequency_weights():
    rows = [
        load_labeled_rows_item('A', 'spades'),
        load_labeled_rows_item('A', 'hearts'),
        load_labeled_rows_item('K', 'clubs'),
    ]

    rank_weights = class_weights(rows, label='rank')
    suit_weights = class_weights(rows, label='suit')

    assert rank_weights[RANK_TO_INDEX['K']] > rank_weights[RANK_TO_INDEX['A']]
    assert suit_weights[SUIT_TO_INDEX['diamonds']] == 0


def test_trim_to_ink_crops_around_dark_or_saturated_pixels():
    image = Image.new('RGB', (20, 20), 'white')
    for x in range(8, 12):
        for y in range(7, 13):
            image.putpixel((x, y), (20, 20, 20))

    trimmed = trim_to_ink(image, padding=1)

    assert trimmed.size == (6, 8)


def load_labeled_rows_item(rank: str, suit: str):
    from ai_balatro.datasets.card_corner_classifier import LabeledCardCorner

    return LabeledCardCorner(image='unused.png', rank=rank, suit=suit)


def load_cli_module(name: str):
    script = Path(__file__).parents[1] / 'cli' / name
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
