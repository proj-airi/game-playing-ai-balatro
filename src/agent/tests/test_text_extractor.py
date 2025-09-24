"""Test the integrated YOLO + RapidOCR text extraction pipeline."""

import cv2
from pathlib import Path

from ..core.text_extractor import TextExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


def test_text_extractor_on_sample_image():
    """Test the text extractor on a sample image with known card descriptions."""

    # Use one of the benchmark test images
    test_image_path = Path(
        'data/datasets/games-balatro-2024-entities-detection/data/train/yolo/images/out_00104.jpg'
    )

    if not test_image_path.exists():
        logger.warning(f'Test image not found: {test_image_path}')
        return

    # Load the test image
    image = cv2.imread(str(test_image_path))
    if image is None:
        logger.error(f'Failed to load image: {test_image_path}')
        return

    logger.info(f'Testing text extraction on image: {test_image_path}')
    logger.info(f'Image shape: {image.shape}')

    # Initialize text extractor
    text_extractor = TextExtractor(target_classes=['card_description'], ocr_lang='en')

    # Extract card descriptions
    card_descriptions = text_extractor.extract_card_descriptions(
        image,
        confidence_threshold=0.5,
        preprocessing_scale=2.0,  # Use 2x scaling as per benchmark results
    )

    logger.info(f'Found {len(card_descriptions)} card descriptions:')

    for i, desc in enumerate(card_descriptions):
        logger.info(f'Description {i + 1}:')
        logger.info(f'  Bbox: {desc["bbox"]}')
        logger.info(f'  Detection confidence: {desc["detection_confidence"]:.3f}')
        logger.info(f'  OCR success: {desc["ocr_success"]}')
        logger.info(f'  OCR confidence: {desc["ocr_confidence"]}')
        logger.info(f'  OCR time: {desc["ocr_time"]:.3f}s')
        logger.info(f"  Text: '{desc['text']}'")
        logger.info('')

    # Test full text detection pipeline
    text_detections = text_extractor.extract_text_from_image(
        image, confidence_threshold=0.5, apply_preprocessing=True, scale_factor=2.0
    )

    logger.info(f'Full pipeline found {len(text_detections)} text detections')

    # Create visualization
    vis_image = text_extractor.visualize_text_detections(
        image, text_detections, show_text=True, show_confidence=True
    )

    # Save visualization
    output_path = Path('apps/tests/outputs/text_extraction_demo.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)
    logger.info(f'Visualization saved to: {output_path}')

    return card_descriptions, text_detections


def test_text_extractor_performance():
    """Test the performance of the text extractor on multiple images."""

    test_images = ['out_00104.jpg', 'out_00166.jpg', 'out_00114.jpg']

    base_path = Path(
        'data/datasets/games-balatro-2024-entities-detection/data/train/yolo/images/'
    )

    text_extractor = TextExtractor(target_classes=['card_description'])

    all_results = []

    for img_name in test_images:
        img_path = base_path / img_name
        if not img_path.exists():
            logger.warning(f'Test image not found: {img_path}')
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        logger.info(f'Processing {img_name}...')

        descriptions = text_extractor.extract_card_descriptions(
            image, confidence_threshold=0.5, preprocessing_scale=2.0
        )

        result = {
            'image': img_name,
            'num_descriptions': len(descriptions),
            'descriptions': descriptions,
        }
        all_results.append(result)

        logger.info(f'  Found {len(descriptions)} descriptions')
        for desc in descriptions:
            if desc['ocr_success']:
                logger.info(
                    f"    Text: '{desc['text'][:50]}...' (conf: {desc['detection_confidence']:.2f})"
                )

    return all_results


if __name__ == '__main__':
    print('Testing integrated YOLO + RapidOCR text extraction pipeline...')

    # Test single image
    print('\n=== Single Image Test ===')
    test_text_extractor_on_sample_image()

    # Test performance across multiple images
    print('\n=== Performance Test ===')
    results = test_text_extractor_performance()

    print(f'\nProcessed {len(results)} images successfully!')
