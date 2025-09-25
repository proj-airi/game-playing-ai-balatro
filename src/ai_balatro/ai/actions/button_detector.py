"""Button detection and recognition functionality."""

from typing import List, Optional
import numpy as np
from ...core.detection import Detection
from ...core.multi_yolo_detector import MultiYOLODetector
from ...utils.logger import get_logger
from .detection_models import ButtonDetection
from .schemas import BUTTON_CONFIG

logger = get_logger(__name__)


class ButtonDetector:
    """Button detection and recognition using specialized UI detection model."""

    def __init__(self, multi_detector: Optional[MultiYOLODetector] = None):
        """
        Initialize button detector.

        Args:
            multi_detector: Multi-model detector instance, create new one if None
        """
        self.multi_detector = multi_detector

        # Button type mapping table, from UI model detection class names to standard button types
        self.button_class_map = {
            # Standard button class names from UI model
            'button_play': 'play',
            'button_discard': 'discard',
            'button_back': 'back',
            'button_card_pack_skip': 'skip',
            'button_cash_out': 'cash_out',
            'button_level_select': 'level_select',
            'button_level_skip': 'skip',
            'button_main_menu': 'main_menu',
            'button_main_menu_play': 'play',
            'button_new_run': 'new_run',
            'button_new_run_play': 'play',
            'button_options': 'options',
            'button_purchase': 'purchase',
            'button_run_info': 'info',
            'button_sell': 'sell',
            'button_sort_hand_rank': 'sort_rank',
            'button_sort_hand_suits': 'sort_suits',
            'button_store_next_round': 'next',
            'button_store_reroll': 'reroll',
            'button_use': 'use',
            # Backup mappings (compatibility)
            'play_button': 'play',
            'discard_button': 'discard',
            'skip_button': 'skip',
            'shop_button': 'shop',
            'next_button': 'next',
        }

    def find_buttons(
        self,
        image: np.ndarray,
        target_type: Optional[str] = None,
        show_visualization: bool = False,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[ButtonDetection]:
        """
        Detect and find buttons from image.

        Args:
            image: Input image
            target_type: Target button type, return all buttons if None
            show_visualization: Whether to show detection result visualization window
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            ButtonDetection list sorted by confidence
        """
        if self.multi_detector is None:
            logger.error('MultiYOLODetector not available for button detection')
            return []

        if not self.multi_detector.is_model_available('ui'):
            logger.error('UI detection model not available')
            return []

        # Use UI model to detect buttons
        try:
            ui_detections = self.multi_detector.detect_ui(
                image, confidence_threshold, iou_threshold
            )

            buttons = []

            for detection in ui_detections:
                if self._is_button(detection):
                    button_type = self._get_button_type(detection)
                    if target_type is None or button_type == target_type:
                        # Create ButtonDetection object
                        button_detection = ButtonDetection(
                            class_id=detection.class_id,
                            class_name=detection.class_name,
                            confidence=detection.confidence,
                            bbox=detection.bbox,
                            button_type=button_type,
                        )
                        buttons.append(button_detection)

            # Sort by confidence (higher confidence first)
            buttons.sort(key=lambda b: b.confidence, reverse=True)

            if buttons:
                logger.info(f'UI model detected {len(buttons)} buttons:')
                for i, button in enumerate(buttons):
                    logger.info(
                        f'  Button {i}: {button.button_type} ({button.class_name}) at {button.center} (confidence: {button.confidence:.3f})'
                    )
            else:
                logger.warning(
                    f'UI model did not detect {"any" if target_type is None else target_type} buttons'
                )
                logger.info(f'UI model detected {len(ui_detections)} objects total:')
                for i, det in enumerate(ui_detections[:5]):  # Show first 5
                    logger.info(
                        f'  {i + 1}. {det.class_name} (confidence: {det.confidence:.3f})'
                    )
                if len(ui_detections) > 5:
                    logger.info(f'  ... and {len(ui_detections) - 5} more objects')

            return buttons

        except Exception as e:
            logger.error(f'UI detection failed: {e}')
            return []

    def find_best_button(
        self,
        image: np.ndarray,
        target_type: str,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Optional[ButtonDetection]:
        """
        Find the best target button.

        Args:
            image: Input image
            target_type: Target button type
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            Best ButtonDetection, None if not found
        """
        buttons = self.find_buttons(
            image, target_type, False, confidence_threshold, iou_threshold
        )
        return buttons[0] if buttons else None

    def _is_button(self, detection: Detection) -> bool:
        """Check if the detection result is a button."""
        class_name = detection.class_name.lower()

        # Check if it contains button keyword or is in mapping table
        return (
            'button' in class_name
            or class_name in self.button_class_map
            or self._matches_button_keywords(class_name)
        )

    def _get_button_type(self, detection: Detection) -> str:
        """Get button type from detection result."""
        class_name = detection.class_name.lower()

        # Direct mapping
        if class_name in self.button_class_map:
            return self.button_class_map[class_name]

        # Keyword matching
        for button_type, config in BUTTON_CONFIG.items():
            keywords = [kw.lower() for kw in config['keywords']]
            if any(keyword in class_name for keyword in keywords):
                return button_type

        # Default return generic button type
        logger.warning(f'Cannot recognize button type: {class_name}, returning default type')
        return 'unknown'

    def _matches_button_keywords(self, class_name: str) -> bool:
        """Check if class name matches button keywords."""
        for config in BUTTON_CONFIG.values():
            keywords = [kw.lower() for kw in config['keywords']]
            if any(keyword in class_name for keyword in keywords):
                return True
        return False
