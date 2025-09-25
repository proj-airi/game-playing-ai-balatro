"""Multi-model YOLO detector for different detection tasks."""

import os
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

from .yolo_detector import YOLODetector
from .detection import Detection
from ..utils.logger import get_logger
from ..utils.path_utils import resolve_path

logger = get_logger(__name__)


class MultiYOLODetector:
    """Multi-model YOLO detector supporting different specialized models."""

    def __init__(self, use_onnx: bool = False):
        """
        Initialize multi-model YOLO detector.

        Args:
            use_onnx: Whether to use ONNX models
        """
        self.use_onnx = use_onnx
        self.detectors: Dict[str, YOLODetector] = {}

        # Model configurations
        self.model_configs = {
            'entities': {
                'path': self._get_entities_model_path(),
                'classes_file': self._get_entities_classes_path(),
                'description': 'Cards and game entities detection',
            },
            'ui': {
                'path': self._get_ui_model_path(),
                'classes_file': self._get_ui_classes_path(),
                'description': 'UI elements and buttons detection',
            },
        }

        # Initialize detectors
        self._initialize_detectors()

        logger.info(f'MultiYOLODetector initialized with {len(self.detectors)} models')

    def _get_entities_model_path(self) -> str:
        """Get entities model path."""
        if self.use_onnx:
            return resolve_path(
                'models/games-balatro-2024-yolo-entities-detection/onnx/model.onnx'
            )
        else:
            return resolve_path(
                'models/games-balatro-2024-yolo-entities-detection/model.pt'
            )

    def _get_ui_model_path(self) -> str:
        """Get UI model path."""
        if self.use_onnx:
            return resolve_path(
                'models/games-balatro-2024-yolo-ui-detection/onnx/model.onnx'
            )
        else:
            return resolve_path('models/games-balatro-2024-yolo-ui-detection/model.pt')

    def _get_entities_classes_path(self) -> str:
        """Get entities classes file path."""
        return resolve_path(
            'data/datasets/games-balatro-2024-entities-detection/data/train/yolo/classes.txt'
        )

    def _get_ui_classes_path(self) -> str:
        """Get UI classes file path."""
        return resolve_path(
            'data/datasets/games-balatro-2024-ui-detection/data/train/yolo/classes.txt'
        )

    def _initialize_detectors(self) -> None:
        """Initialize individual YOLO detectors."""
        for model_name, config in self.model_configs.items():
            try:
                model_path = config['path']
                classes_file = config['classes_file']

                if not os.path.exists(model_path):
                    logger.warning(f'{model_name} model not found: {model_path}')
                    continue

                # Load class names for this model
                class_names = self._load_class_names(classes_file)

                # Create custom detector with specific classes
                detector = YOLODetector(model_path, use_onnx=self.use_onnx)

                # Override class names
                detector.class_names = class_names

                self.detectors[model_name] = detector

                logger.info(f'Loaded {model_name} model: {config["description"]}')
                logger.info(
                    f'  Classes: {len(class_names)} ({", ".join(class_names[:3])}{"..." if len(class_names) > 3 else ""})'
                )

            except Exception as e:
                logger.error(f'Failed to load {model_name} model: {e}')

    def _load_class_names(self, classes_file: str) -> List[str]:
        """Load class names from file."""
        class_names = []

        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    class_names = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                logger.debug(f'Loaded {len(class_names)} classes from {classes_file}')
            except Exception as e:
                logger.warning(f'Failed to load classes from {classes_file}: {e}')
        else:
            logger.warning(f'Classes file not found: {classes_file}')

        return class_names

    def detect_entities(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Detect game entities (cards, jokers, etc.) in image.

        Args:
            image: Input image
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of entity detections
        """
        if 'entities' not in self.detectors:
            logger.error('Entities detector not available')
            return []

        try:
            detections = self.detectors['entities'].detect(
                image, confidence_threshold, iou_threshold
            )
            logger.debug(f'Entities detector found {len(detections)} objects')
            return detections
        except Exception as e:
            logger.error(f'Entities detection failed: {e}')
            return []

    def detect_ui(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Detect UI elements (buttons, score displays, etc.) in image.

        Args:
            image: Input image
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of UI detections
        """
        if 'ui' not in self.detectors:
            logger.error('UI detector not available')
            return []

        try:
            detections = self.detectors['ui'].detect(
                image, confidence_threshold, iou_threshold
            )
            logger.debug(f'UI detector found {len(detections)} objects')
            return detections
        except Exception as e:
            logger.error(f'UI detection failed: {e}')
            return []

    def detect_combined(
        self,
        image: np.ndarray,
        include_entities: bool = True,
        include_ui: bool = True,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Detect both entities and UI elements in image.

        Args:
            image: Input image
            include_entities: Whether to run entities detection
            include_ui: Whether to run UI detection
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            Tuple of (entity_detections, ui_detections)
        """
        entity_detections = []
        ui_detections = []

        if include_entities:
            entity_detections = self.detect_entities(
                image, confidence_threshold, iou_threshold
            )

        if include_ui:
            ui_detections = self.detect_ui(image, confidence_threshold, iou_threshold)

        logger.info(
            f'Combined detection: {len(entity_detections)} entities, {len(ui_detections)} UI elements'
        )

        return entity_detections, ui_detections

    def get_model_info(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """Get information about loaded models."""
        info = {}

        for model_name, detector in self.detectors.items():
            info[model_name] = {
                'description': self.model_configs[model_name]['description'],
                'model_path': self.model_configs[model_name]['path'],
                'classes_count': len(detector.class_names),
                'class_names': detector.class_names.copy(),
                'available': True,
            }

        # Add info for unavailable models
        for model_name, config in self.model_configs.items():
            if model_name not in self.detectors:
                info[model_name] = {
                    'description': config['description'],
                    'model_path': config['path'],
                    'classes_count': 0,
                    'class_names': [],
                    'available': False,
                }

        return info

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        return model_name in self.detectors

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.detectors.keys())

    def get_button_classes(self) -> List[str]:
        """Get list of button classes from UI model."""
        if 'ui' not in self.detectors:
            return []

        ui_classes = self.detectors['ui'].class_names
        button_classes = [cls for cls in ui_classes if 'button' in cls.lower()]

        return button_classes

    def get_card_classes(self) -> List[str]:
        """Get list of card classes from entities model."""
        if 'entities' not in self.detectors:
            return []

        entity_classes = self.detectors['entities'].class_names
        card_classes = [cls for cls in entity_classes if 'card' in cls.lower()]

        return card_classes
