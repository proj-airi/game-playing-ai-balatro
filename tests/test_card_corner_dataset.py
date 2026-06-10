"""Tests for exporting Balatro card-corner crops from YOLO labels."""

import csv
import json

from PIL import Image

from ai_balatro.datasets.card_corners import (
    CardCornerExportOptions,
    corner_bbox_from_card_bbox,
    export_card_corner_dataset,
    yolo_bbox_to_xyxy,
)


def test_yolo_bbox_to_xyxy_converts_normalized_box_to_pixels():
    bbox = yolo_bbox_to_xyxy(
        image_width=200,
        image_height=100,
        center_x=0.5,
        center_y=0.5,
        width=0.4,
        height=0.2,
    )

    assert bbox == (60, 40, 140, 60)


def test_corner_bbox_from_card_bbox_uses_ratios_and_clamps_to_image():
    corner = corner_bbox_from_card_bbox(
        card_bbox=(5, 3, 105, 203),
        image_width=120,
        image_height=220,
        width_ratio=0.36,
        height_ratio=0.46,
        padding=10,
    )

    assert corner == (0, 0, 51, 105)


def test_export_card_corner_dataset_writes_crops_manifest_and_csv(tmp_path):
    images_dir = tmp_path / 'images'
    labels_dir = tmp_path / 'labels'
    out_dir = tmp_path / 'out'
    images_dir.mkdir()
    labels_dir.mkdir()

    image = Image.new('RGB', (200, 100), 'black')
    for x in range(60, 140):
        for y in range(40, 60):
            image.putpixel((x, y), (255, 0, 0))
    image.save(images_dir / 'screen.png')

    (labels_dir / 'screen.txt').write_text(
        # poker_card_front, centered at 100,50, width 80, height 20
        '6 0.5 0.5 0.4 0.2\n'
        # joker_card should be ignored by the default class filter
        '2 0.2 0.2 0.1 0.1\n',
        encoding='utf-8',
    )
    classes_file = tmp_path / 'classes.txt'
    classes_file.write_text(
        '\n'.join(
            [
                'card_description',
                'card_pack',
                'joker_card',
                'planet_card',
                'poker_card_back',
                'poker_card_description',
                'poker_card_front',
            ]
        ),
        encoding='utf-8',
    )

    result = export_card_corner_dataset(
        CardCornerExportOptions(
            images_dir=images_dir,
            labels_dir=labels_dir,
            classes_file=classes_file,
            out_dir=out_dir,
            write_csv=True,
        )
    )

    assert result.sample_count == 1
    assert (out_dir / 'crops' / 'screen-card-000.png').exists()

    manifest_lines = (
        (out_dir / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
    )
    assert len(manifest_lines) == 1
    manifest = json.loads(manifest_lines[0])
    assert manifest['image'] == 'crops/screen-card-000.png'
    assert manifest['source_image'] == str(images_dir / 'screen.png')
    assert manifest['class_name'] == 'poker_card_front'
    assert manifest['slot_index'] == 0
    assert manifest['card_bbox_xyxy'] == [60, 40, 140, 60]
    assert manifest['corner_bbox_xyxy'] == [60, 40, 89, 50]
    assert manifest['label'] == {'rank': None, 'suit': None}

    with (out_dir / 'manifest.csv').open(newline='', encoding='utf-8') as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]['image'] == 'crops/screen-card-000.png'
    assert rows[0]['rank'] == ''
    assert rows[0]['suit'] == ''


def test_export_card_corner_dataset_filters_crops_without_rank_or_suit_ink(tmp_path):
    images_dir = tmp_path / 'images'
    labels_dir = tmp_path / 'labels'
    out_dir = tmp_path / 'out'
    images_dir.mkdir()
    labels_dir.mkdir()

    image = Image.new('RGB', (200, 100), 'black')
    for x in range(60, 140):
        for y in range(40, 60):
            image.putpixel((x, y), (245, 245, 245))
    # Simulates a crop that mostly sees the white card edge with only a tiny
    # speck of dark pixels, not enough rank/suit ink for a useful label.
    image.putpixel((88, 40), (20, 20, 20))
    image.putpixel((88, 41), (20, 20, 20))
    image.save(images_dir / 'edge_only.png')

    (labels_dir / 'edge_only.txt').write_text(
        '6 0.5 0.5 0.4 0.2\n',
        encoding='utf-8',
    )
    classes_file = tmp_path / 'classes.txt'
    classes_file.write_text(
        '\n'.join(
            [
                'card_description',
                'card_pack',
                'joker_card',
                'planet_card',
                'poker_card_back',
                'poker_card_description',
                'poker_card_front',
            ]
        ),
        encoding='utf-8',
    )

    result = export_card_corner_dataset(
        CardCornerExportOptions(
            images_dir=images_dir,
            labels_dir=labels_dir,
            classes_file=classes_file,
            out_dir=out_dir,
            min_light_ratio=0.20,
            min_ink_ratio=0.05,
        )
    )

    assert result.sample_count == 0
    assert not list((out_dir / 'crops').glob('*.png'))
