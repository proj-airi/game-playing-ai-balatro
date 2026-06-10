"""Export card-corner crops from a YOLO-labelled Balatro dataset."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class CardCornerExportOptions:
    """Options for deriving card-corner crops from YOLO card detections."""

    images_dir: Path
    labels_dir: Path
    classes_file: Path
    out_dir: Path
    class_name: str = 'poker_card_front'
    corner_width_ratio: float = 0.36
    corner_height_ratio: float = 0.46
    padding: int = 0
    limit: int | None = None
    min_light_ratio: float | None = None
    min_ink_ratio: float | None = 0.04
    write_csv: bool = False


@dataclass(frozen=True)
class CardCornerSample:
    """One exported corner crop and its source geometry."""

    image: str
    source_image: str
    source_label: str
    class_id: int
    class_name: str
    slot_index: int
    card_bbox_xyxy: BBox
    corner_bbox_xyxy: BBox
    quality: dict[str, float]
    label: dict[str, str | None]


@dataclass(frozen=True)
class CardCornerExportResult:
    """Summary of an export run."""

    sample_count: int
    manifest_path: Path
    csv_path: Path | None
    crops_dir: Path


def yolo_bbox_to_xyxy(
    *,
    image_width: int,
    image_height: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> BBox:
    """Convert a normalized YOLO bbox into pixel-space ``(x1, y1, x2, y2)``."""

    box_width = width * image_width
    box_height = height * image_height
    x1 = round(center_x * image_width - box_width / 2)
    y1 = round(center_y * image_height - box_height / 2)
    x2 = round(center_x * image_width + box_width / 2)
    y2 = round(center_y * image_height + box_height / 2)

    return (
        _clamp_int(x1, 0, image_width),
        _clamp_int(y1, 0, image_height),
        _clamp_int(x2, 0, image_width),
        _clamp_int(y2, 0, image_height),
    )


def corner_bbox_from_card_bbox(
    *,
    card_bbox: BBox,
    image_width: int,
    image_height: int,
    width_ratio: float,
    height_ratio: float,
    padding: int = 0,
) -> BBox:
    """Return the top-left corner bbox used for rank/suit sample extraction."""

    x1, y1, x2, y2 = card_bbox
    card_width = max(0, x2 - x1)
    card_height = max(0, y2 - y1)
    corner_width = math.ceil(card_width * width_ratio)
    corner_height = math.ceil(card_height * height_ratio)

    return (
        _clamp_int(x1 - padding, 0, image_width),
        _clamp_int(y1 - padding, 0, image_height),
        _clamp_int(x1 + corner_width + padding, 0, image_width),
        _clamp_int(y1 + corner_height + padding, 0, image_height),
    )


def export_card_corner_dataset(
    options: CardCornerExportOptions,
) -> CardCornerExportResult:
    """Export card-corner crop images and label-ready manifests."""

    class_names = _read_classes(options.classes_file)
    if options.class_name not in class_names:
        raise ValueError(
            f'class name {options.class_name!r} not found in {options.classes_file}'
        )

    class_id = class_names.index(options.class_name)
    crops_dir = options.out_dir / 'crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    samples: list[CardCornerSample] = []
    image_paths = _iter_image_paths(options.images_dir)
    if options.limit is not None:
        image_paths = image_paths[: options.limit]

    for image_path in image_paths:
        label_path = options.labels_dir / f'{image_path.stem}.txt'
        if not label_path.exists():
            continue

        with Image.open(image_path) as image:
            image = image.convert('RGB')
            card_boxes = _read_card_boxes(
                label_path=label_path,
                target_class_id=class_id,
                image_width=image.width,
                image_height=image.height,
            )

            for slot_index, card_bbox in enumerate(_sort_card_boxes(card_boxes)):
                corner_bbox = corner_bbox_from_card_bbox(
                    card_bbox=card_bbox,
                    image_width=image.width,
                    image_height=image.height,
                    width_ratio=options.corner_width_ratio,
                    height_ratio=options.corner_height_ratio,
                    padding=options.padding,
                )
                if _empty_bbox(corner_bbox):
                    continue

                crop_name = f'{image_path.stem}-card-{slot_index:03d}.png'
                crop_path = crops_dir / crop_name
                crop = image.crop(corner_bbox)
                light_ratio = _light_pixel_ratio(crop)
                ink_ratio = _ink_pixel_ratio(crop)
                if (
                    options.min_light_ratio is not None
                    and light_ratio < options.min_light_ratio
                ):
                    continue
                if (
                    options.min_ink_ratio is not None
                    and ink_ratio < options.min_ink_ratio
                ):
                    continue

                crop.save(crop_path)

                samples.append(
                    CardCornerSample(
                        image=str(Path('crops') / crop_name),
                        source_image=str(image_path),
                        source_label=str(label_path),
                        class_id=class_id,
                        class_name=options.class_name,
                        slot_index=slot_index,
                        card_bbox_xyxy=card_bbox,
                        corner_bbox_xyxy=corner_bbox,
                        quality={'light_ratio': light_ratio, 'ink_ratio': ink_ratio},
                        label={'rank': None, 'suit': None},
                    )
                )

    manifest_path = options.out_dir / 'manifest.jsonl'
    _write_manifest(manifest_path, samples)

    csv_path = None
    if options.write_csv:
        csv_path = options.out_dir / 'manifest.csv'
        _write_csv(csv_path, samples)

    return CardCornerExportResult(
        sample_count=len(samples),
        manifest_path=manifest_path,
        csv_path=csv_path,
        crops_dir=crops_dir,
    )


def _read_classes(classes_file: Path) -> list[str]:
    return [
        line.strip()
        for line in classes_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _iter_image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_card_boxes(
    *,
    label_path: Path,
    target_class_id: int,
    image_width: int,
    image_height: int,
) -> list[BBox]:
    boxes = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        if class_id != target_class_id:
            continue

        center_x, center_y, width, height = map(float, parts[1:5])
        boxes.append(
            yolo_bbox_to_xyxy(
                image_width=image_width,
                image_height=image_height,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
            )
        )

    return boxes


def _sort_card_boxes(card_boxes: Iterable[BBox]) -> list[BBox]:
    return sorted(card_boxes, key=lambda bbox: (_center_y(bbox), _center_x(bbox)))


def _center_x(bbox: BBox) -> float:
    return (bbox[0] + bbox[2]) / 2


def _center_y(bbox: BBox) -> float:
    return (bbox[1] + bbox[3]) / 2


def _write_manifest(manifest_path: Path, samples: list[CardCornerSample]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', encoding='utf-8') as manifest_file:
        for sample in samples:
            manifest_file.write(json.dumps(_sample_to_dict(sample)) + '\n')


def _write_csv(csv_path: Path, samples: list[CardCornerSample]) -> None:
    fieldnames = [
        'image',
        'rank',
        'suit',
        'source_image',
        'source_label',
        'slot_index',
        'card_bbox_x1',
        'card_bbox_y1',
        'card_bbox_x2',
        'card_bbox_y2',
        'corner_bbox_x1',
        'corner_bbox_y1',
        'corner_bbox_x2',
        'corner_bbox_y2',
        'light_ratio',
        'ink_ratio',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    'image': sample.image,
                    'rank': '',
                    'suit': '',
                    'source_image': sample.source_image,
                    'source_label': sample.source_label,
                    'slot_index': sample.slot_index,
                    'card_bbox_x1': sample.card_bbox_xyxy[0],
                    'card_bbox_y1': sample.card_bbox_xyxy[1],
                    'card_bbox_x2': sample.card_bbox_xyxy[2],
                    'card_bbox_y2': sample.card_bbox_xyxy[3],
                    'corner_bbox_x1': sample.corner_bbox_xyxy[0],
                    'corner_bbox_y1': sample.corner_bbox_xyxy[1],
                    'corner_bbox_x2': sample.corner_bbox_xyxy[2],
                    'corner_bbox_y2': sample.corner_bbox_xyxy[3],
                    'light_ratio': sample.quality['light_ratio'],
                    'ink_ratio': sample.quality['ink_ratio'],
                }
            )


def _sample_to_dict(sample: CardCornerSample) -> dict:
    return {
        'image': sample.image,
        'source_image': sample.source_image,
        'source_label': sample.source_label,
        'class_id': sample.class_id,
        'class_name': sample.class_name,
        'slot_index': sample.slot_index,
        'card_bbox_xyxy': list(sample.card_bbox_xyxy),
        'corner_bbox_xyxy': list(sample.corner_bbox_xyxy),
        'quality': sample.quality,
        'label': sample.label,
    }


def _light_pixel_ratio(image: Image.Image) -> float:
    data = image.convert('RGB').tobytes()
    total_pixels = len(data) // 3
    if total_pixels == 0:
        return 0.0

    light_pixels = sum(
        1
        for index in range(0, len(data), 3)
        if data[index] >= 200 and data[index + 1] >= 200 and data[index + 2] >= 200
    )
    return light_pixels / total_pixels


def _ink_pixel_ratio(image: Image.Image) -> float:
    data = image.convert('RGB').tobytes()
    total_pixels = len(data) // 3
    if total_pixels == 0:
        return 0.0

    ink_pixels = 0
    for index in range(0, len(data), 3):
        red = data[index]
        green = data[index + 1]
        blue = data[index + 2]
        dark = red <= 90 and green <= 90 and blue <= 90
        saturated = max(red, green, blue) - min(red, green, blue) >= 70
        bright_enough = max(red, green, blue) >= 120
        if dark or (saturated and bright_enough):
            ink_pixels += 1

    return ink_pixels / total_pixels


def _empty_bbox(bbox: BBox) -> bool:
    return bbox[2] <= bbox[0] or bbox[3] <= bbox[1]


def _clamp_int(value: float | int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
