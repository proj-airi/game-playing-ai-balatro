#!/usr/bin/env python3
"""
Convert YOLO dataset to HuggingFace format.
Preserves original YOLO structure in yolo/ subdirectory and adds metadata.jsonl for HuggingFace compatibility.
"""

import os
import json
import shutil
import argparse
from pathlib import Path


def read_yolo_label(label_file: str) -> str:
    """Read YOLO label file and return raw content."""
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            return f.read().strip()
    return ''


def create_huggingface_dataset(
    source_dir: str, output_dir: str, split_name: str = 'train'
):
    """Convert YOLO dataset to HuggingFace format."""

    # Paths
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    split_path = output_path / 'data' / split_name
    yolo_path = split_path / 'yolo'

    # Create output directories
    split_path.mkdir(parents=True, exist_ok=True)

    # Copy entire YOLO structure to yolo subdirectory
    print('Copying YOLO dataset structure...')

    # Copy images, labels, and other files
    for item in ['images', 'labels', 'classes.txt', 'project.json', 'notes.json']:
        src_item = source_path / item
        if src_item.exists():
            dst_item = yolo_path / item
            if src_item.is_dir():
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                dst_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_item, dst_item)
            print(f'  Copied {item}')

    # Process images and create metadata
    images_dir = yolo_path / 'images'
    labels_dir = yolo_path / 'labels'

    metadata_entries = []

    # Get all image files
    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    image_files.sort()

    print(f'\nProcessing {len(image_files)} images for metadata...')

    for image_file in image_files:
        # Label file path
        label_file = labels_dir / (Path(image_file).stem + '.txt')

        # Read raw YOLO label content
        label_content = read_yolo_label(str(label_file))

        # Create metadata entry with relative path to image in yolo structure
        metadata_entry = {
            'file_name': f'yolo/images/{image_file}',
            'label': label_content,
        }
        metadata_entries.append(metadata_entry)

    # Write metadata.jsonl
    metadata_file = split_path / 'metadata.jsonl'
    with open(metadata_file, 'w') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')

    print('\nConversion complete!')
    print(f'- Preserved original YOLO structure in {yolo_path}')
    print(f'- Generated metadata.jsonl with {len(metadata_entries)} entries')
    print("- Image paths in metadata reference 'yolo/images/' directory")
    print('\nDataset structure:')
    print(f'{output_path}/')
    print('└── data/')
    print(f'    └── {split_name}/')
    print('        ├── yolo/')
    print('        │   ├── images/')
    print('        │   ├── labels/')
    print('        │   └── classes.txt')
    print('        └── metadata.jsonl')


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO dataset to HuggingFace format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        'source_dir',
        help='Source directory containing YOLO dataset (images/, labels/, classes.txt)',
    )

    parser.add_argument('output_dir', help='Output directory for HuggingFace dataset')

    parser.add_argument(
        '--split', default='train', help='Split name (e.g., train, test, val)'
    )

    args = parser.parse_args()

    # Validate source directory
    source_path = Path(args.source_dir)
    if not source_path.exists():
        print(f"Error: Source directory '{args.source_dir}' does not exist")
        return 1

    required_items = ['images', 'labels']
    missing_items = [
        item for item in required_items if not (source_path / item).exists()
    ]
    if missing_items:
        print(f'Error: Missing required directories in source: {missing_items}')
        return 1

    print('Converting YOLO dataset:')
    print(f'  Source: {args.source_dir}')
    print(f'  Output: {args.output_dir}')
    print(f'  Split: {args.split}')
    print()

    # Create the dataset
    try:
        create_huggingface_dataset(args.source_dir, args.output_dir, args.split)
        return 0
    except Exception as e:
        print(f'Error during conversion: {e}')
        return 1


if __name__ == '__main__':
    exit(main())
