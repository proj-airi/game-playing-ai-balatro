"""YOLO detection module for Balatro game elements."""

import os
from typing import List, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO
import onnxruntime as ort

from core.detection import Detection
from config.settings import settings
from utils.logger import get_logger
from utils.path_utils import find_model_file, resolve_path

logger = get_logger(__name__)


class YOLODetector:
    """YOLO detector supporting both PyTorch and ONNX models."""

    def __init__(self, model_path: Optional[str] = None, use_onnx: bool = False):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to model file. If None, searches default paths.
            use_onnx: Whether to use ONNX model format
        """
        self.use_onnx = use_onnx
        self.model = None
        self.class_names = []

        # Find model path if not provided
        if model_path is None:
            model_path = self._find_model_path()

        if not model_path:
            raise FileNotFoundError('No model file found')

        self.model_path = resolve_path(model_path)

        # Load class names and model
        self._load_class_names()
        self._load_model()

        logger.info(f'YOLO detector initialized with {len(self.class_names)} classes')

    def _find_model_path(self) -> Optional[str]:
        """Find model file from configured search paths."""
        search_paths = settings.model_search_paths
        if self.use_onnx and settings.model_onnx_path:
            search_paths = [settings.model_onnx_path] + search_paths

        return find_model_file(search_paths)

    def _load_class_names(self) -> None:
        """Load class names from configuration."""
        classes_file = resolve_path(settings.model_classes_file)

        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    self.class_names = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                logger.info(
                    f'Loaded {len(self.class_names)} classes from {classes_file}'
                )
            except Exception as e:
                logger.warning(f'Failed to load classes file: {e}')
                self.class_names = settings.model_default_classes
        else:
            logger.info('Classes file not found, using default classes')
            self.class_names = settings.model_default_classes

    def _load_model(self) -> None:
        """Load YOLO model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model file not found: {self.model_path}')

        try:
            if self.use_onnx:
                self.model = ort.InferenceSession(self.model_path)
                logger.info(f'Loaded ONNX model: {self.model_path}')
            else:
                self.model = YOLO(self.model_path)
                logger.info(f'Loaded PyTorch model: {self.model_path}')
        except Exception as e:
            raise RuntimeError(f'Failed to load model: {e}')

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Detect objects in image.

        Args:
            image: Input image in BGR format
            confidence_threshold: Confidence threshold (uses config default if None)
            iou_threshold: IoU threshold for NMS (uses config default if None)

        Returns:
            List of detection results
        """
        if confidence_threshold is None:
            confidence_threshold = settings.detection_confidence_threshold
        if iou_threshold is None:
            iou_threshold = settings.detection_iou_threshold

        try:
            if self.use_onnx:
                return self._detect_onnx(image, confidence_threshold, iou_threshold)
            else:
                return self._detect_pytorch(image, confidence_threshold, iou_threshold)
        except Exception as e:
            logger.error(f'Detection failed: {e}')
            return []

    def _detect_pytorch(
        self, image: np.ndarray, confidence_threshold: float, iou_threshold: float
    ) -> List[Detection]:
        """Detect using PyTorch model."""
        results = self.model(
            image, conf=confidence_threshold, iou=iou_threshold, verbose=False
        )

        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    class_id = class_ids[i]
                    class_name = (
                        self.class_names[class_id]
                        if class_id < len(self.class_names)
                        else f'class_{class_id}'
                    )
                    confidence = float(confidences[i])
                    bbox = tuple(map(int, boxes[i]))  # (x1, y1, x2, y2)

                    detection = Detection(class_id, class_name, confidence, bbox)
                    detections.append(detection)

        return detections

    def _detect_onnx(
        self, image: np.ndarray, confidence_threshold: float, iou_threshold: float
    ) -> List[Detection]:
        """Detect using ONNX model."""
        # Preprocess image
        input_tensor = self._preprocess_onnx(image)

        # Run inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})

        # Postprocess
        detections = self._postprocess_onnx(
            outputs[0], image.shape, confidence_threshold, iou_threshold
        )

        return detections

    def _preprocess_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model."""
        input_size = settings.detection_input_size
        h, w = image.shape[:2]

        # Maintain aspect ratio resize
        scale = min(input_size / w, input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Create square image with padding
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        top = (input_size - new_h) // 2
        left = (input_size - new_w) // 2
        padded[top : top + new_h, left : left + new_w] = resized

        # Convert to model input format
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

        return input_tensor

    def _postprocess_onnx(
        self,
        outputs: np.ndarray,
        original_shape: Tuple[int, int, int],
        confidence_threshold: float,
        iou_threshold: float,
    ) -> List[Detection]:
        """Postprocess ONNX model outputs."""
        detections = []
        predictions = outputs[0]  # Remove batch dimension

        h, w = original_shape[:2]
        input_size = settings.detection_input_size
        scale = min(input_size / w, input_size / h)

        for prediction in predictions:
            # Parse prediction
            x_center, y_center, width, height = prediction[:4]
            objectness = prediction[4]
            class_scores = prediction[5:]

            # Filter low confidence detections
            if objectness < confidence_threshold:
                continue

            # Find highest scoring class
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            final_confidence = objectness * class_score

            if final_confidence < confidence_threshold:
                continue

            # Convert coordinates to original image size
            x_center = (x_center - (input_size - w * scale) / 2) / scale
            y_center = (y_center - (input_size - h * scale) / 2) / scale
            width = width / scale
            height = height / scale

            # Convert to x1, y1, x2, y2 format
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Clamp coordinates to image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f'class_{class_id}'
            )
            detection = Detection(
                class_id, class_name, final_confidence, (x1, y1, x2, y2)
            )
            detections.append(detection)

        # Apply NMS
        detections = self._apply_nms(detections, iou_threshold)

        return detections

    def _apply_nms(
        self, detections: List[Detection], iou_threshold: float
    ) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            # Keep highest confidence detection
            current = detections.pop(0)
            keep.append(current)

            # Remove detections with high IoU
            detections = [
                det
                for det in detections
                if self._calculate_iou(current.bbox, det.bbox) < iou_threshold
            ]

        return keep

    def _calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Visualize detection results on image.

        Args:
            image: Input image
            detections: List of detections
            show_confidence: Whether to show confidence scores

        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        colors = settings.ui_colors

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors[detection.class_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = detection.class_name
            if show_confidence:
                label += f' {detection.confidence:.2f}'

            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw text background
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )

            # Draw text
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
