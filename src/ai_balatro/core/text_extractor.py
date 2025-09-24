"""Text extraction pipeline integrating YOLO detection with OCR engines."""

from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from dataclasses import dataclass

from ..core.yolo_detector import YOLODetector
from ..core.detection import Detection
from ..ocr.engines import RapidOCREngine, OcrResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextDetection:
    """Represents a detected text region with OCR results."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    detection_confidence: float
    text: str
    ocr_success: bool
    ocr_confidence: Optional[float] = None
    raw_ocr_result: Optional[OcrResult] = None


class TextExtractor:
    """Combines YOLO object detection with OCR for text extraction from game UI."""

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        use_onnx: bool = False,
        target_classes: Optional[List[str]] = None,
        ocr_lang: str = 'en',
    ):
        """
        Initialize the text extractor.

        Args:
            yolo_model_path: Path to YOLO model file
            use_onnx: Whether to use ONNX model format
            target_classes: List of class names to extract text from. If None, extracts from all detected objects.
            ocr_lang: Language for OCR engine
        """
        self.yolo_detector = YOLODetector(model_path=yolo_model_path, use_onnx=use_onnx)
        self.ocr_engine = RapidOCREngine(lang=ocr_lang)
        self.ocr_engine.init()

        # Set target classes for text extraction
        self.target_classes = target_classes or ['card_description']
        self.target_class_ids = self._get_target_class_ids()

        logger.info(
            f'TextExtractor initialized with target classes: {self.target_classes}'
        )
        logger.info(f'OCR engine available: {self.ocr_engine.available}')

    def _get_target_class_ids(self) -> List[int]:
        """Get class IDs for target classes."""
        class_ids = []
        for target_class in self.target_classes:
            try:
                class_id = self.yolo_detector.class_names.index(target_class)
                class_ids.append(class_id)
                logger.info(f"Target class '{target_class}' mapped to ID {class_id}")
            except ValueError:
                logger.warning(
                    f"Target class '{target_class}' not found in model classes"
                )
        return class_ids

    def extract_text_from_image(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        apply_preprocessing: bool = True,
        scale_factor: float = 2.0,
    ) -> List[TextDetection]:
        """
        Extract text from detected regions in an image.

        Args:
            image: Input image in BGR format
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            apply_preprocessing: Whether to apply preprocessing to improve OCR
            scale_factor: Scale factor for OCR preprocessing (2.0-3.0 recommended based on benchmark)

        Returns:
            List of text detections with OCR results
        """
        # Run YOLO detection
        detections = self.yolo_detector.detect(
            image, confidence_threshold, iou_threshold
        )

        # Filter detections to target classes
        target_detections = [
            det for det in detections if det.class_id in self.target_class_ids
        ]

        if not target_detections:
            logger.info('No target class detections found')
            return []

        logger.info(f'Found {len(target_detections)} target detections')

        # Extract text from each detection
        text_detections = []
        for detection in target_detections:
            text_detection = self._extract_text_from_detection(
                image, detection, apply_preprocessing, scale_factor
            )
            text_detections.append(text_detection)

        return text_detections

    def _extract_text_from_detection(
        self,
        image: np.ndarray,
        detection: Detection,
        apply_preprocessing: bool,
        scale_factor: float,
    ) -> TextDetection:
        """Extract text from a single detection region."""
        x1, y1, x2, y2 = detection.bbox

        # Crop the detection region
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            logger.warning(f'Empty crop for detection {detection.class_name}')
            return TextDetection(
                bbox=detection.bbox,
                class_id=detection.class_id,
                class_name=detection.class_name,
                detection_confidence=detection.confidence,
                text='',
                ocr_success=False,
            )

        # Apply preprocessing if requested
        if apply_preprocessing:
            crop = self._preprocess_for_ocr(crop, scale_factor)

        # Run OCR
        ocr_result = self.ocr_engine.run(crop)

        # Extract confidence if available in raw results
        ocr_confidence = None
        if ocr_result.raw_results and hasattr(ocr_result.raw_results, 'to_json'):
            try:
                json_results = ocr_result.raw_results.to_json()
                if json_results and len(json_results) > 0:
                    # Average confidence across all detected text regions
                    confidences = []
                    for line in json_results:
                        if 'conf' in line:
                            confidences.append(line['conf'])
                    if confidences:
                        ocr_confidence = sum(confidences) / len(confidences)
            except Exception as e:
                logger.debug(f'Could not extract OCR confidence: {e}')

        return TextDetection(
            bbox=detection.bbox,
            class_id=detection.class_id,
            class_name=detection.class_name,
            detection_confidence=detection.confidence,
            text=ocr_result.text,
            ocr_success=ocr_result.success,
            ocr_confidence=ocr_confidence,
            raw_ocr_result=ocr_result,
        )

    def _preprocess_for_ocr(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        Based on benchmark results, scaling 2x-3x generally improves accuracy.
        """
        # Scale the image
        if scale_factor != 1.0:
            height, width = image.shape[:2]
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        return image

    def extract_card_descriptions(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        preprocessing_scale: float = 2.0,
    ) -> List[Dict]:
        """
        Convenience method specifically for extracting card descriptions.

        Args:
            image: Input image in BGR format
            confidence_threshold: Detection confidence threshold
            preprocessing_scale: Scale factor for OCR preprocessing

        Returns:
            List of dictionaries with card description data
        """
        # Temporarily set target classes to just card_description
        original_target_classes = self.target_classes
        original_target_class_ids = self.target_class_ids

        self.target_classes = ['card_description']
        self.target_class_ids = self._get_target_class_ids()

        try:
            text_detections = self.extract_text_from_image(
                image,
                confidence_threshold=confidence_threshold,
                apply_preprocessing=True,
                scale_factor=preprocessing_scale,
            )

            # Convert to simple dict format
            results = []
            for detection in text_detections:
                results.append(
                    {
                        'bbox': detection.bbox,
                        'text': detection.text,
                        'detection_confidence': detection.detection_confidence,
                        'ocr_success': detection.ocr_success,
                        'ocr_confidence': detection.ocr_confidence,
                        'ocr_time': detection.raw_ocr_result.ocr_time
                        if detection.raw_ocr_result
                        else None,
                    }
                )

            return results

        finally:
            # Restore original target classes
            self.target_classes = original_target_classes
            self.target_class_ids = original_target_class_ids

    def visualize_text_detections(
        self,
        image: np.ndarray,
        text_detections: List[TextDetection],
        show_text: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Visualize text detection results on image.

        Args:
            image: Input image
            text_detections: List of text detections
            show_text: Whether to show extracted text
            show_confidence: Whether to show confidence scores

        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()

        for detection in text_detections:
            x1, y1, x2, y2 = detection.bbox

            # Choose color based on OCR success
            color = (0, 255, 0) if detection.ocr_success else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label_parts = [detection.class_name]
            if show_confidence:
                label_parts.append(f'det:{detection.detection_confidence:.2f}')
                if detection.ocr_confidence is not None:
                    label_parts.append(f'ocr:{detection.ocr_confidence:.2f}')

            if show_text and detection.text:
                # Truncate long text for display
                text_preview = (
                    detection.text[:30] + '...'
                    if len(detection.text) > 30
                    else detection.text
                )
                label_parts.append(f'"{text_preview}"')

            label = ' '.join(label_parts)

            # Draw text background and label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )

            cv2.putText(
                vis_image,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        return vis_image
