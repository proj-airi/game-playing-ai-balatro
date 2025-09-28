"""Balatro-specific service for matching card tooltips to cards with OCR and hover functionality."""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from ..core.detection import Detection
from ..core.multi_yolo_detector import MultiYOLODetector
from ..utils.image_cropper import ImageCropper, RegionMatcher
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CardTooltipService:
    """Service for matching Balatro card description tooltips to their cards with OCR and hover capabilities."""

    def __init__(
        self,
        image_cropper: Optional[ImageCropper] = None,
        screen_capture=None,
        mouse_controller=None,
        multi_detector: Optional[MultiYOLODetector] = None,
    ):
        """
        Initialize service.

        Args:
            image_cropper: Image cropper utility (creates default if None)
            screen_capture: Screen capture instance for hover functionality
            mouse_controller: Mouse controller for hover actions
        """
        self.image_cropper = image_cropper or ImageCropper(padding_pixels=15)
        self.region_matcher = RegionMatcher()

        # External dependencies for hover functionality
        self.screen_capture = screen_capture
        self.mouse_controller = mouse_controller
        self.multi_detector = multi_detector

        if self.multi_detector is None:
            try:
                self.multi_detector = MultiYOLODetector()
                logger.info(
                    'CardTooltipService created default MultiYOLODetector instance'
                )
            except Exception as exc:  # pragma: no cover - runtime dependency
                logger.warning(f'Failed to initialize default MultiYOLODetector: {exc}')

        # Hover settings
        self.hover_duration = 1.5  # Time to wait for tooltip to appear
        self.card_hover_offset = (0, -20)  # Hover slightly above card center
        self.tooltip_stabilization_time = 0.8  # Time to wait for tooltip to stabilize

        # OCR settings - use RapidOCR (fastest and best)
        self.ocr_enabled = self._check_ocr_availability()
        if self.ocr_enabled:
            from ..ocr.engines import RapidOCREngine

            self.ocr_engine = RapidOCREngine()
            self.ocr_engine.init()
        else:
            self.ocr_engine = None
            logger.warning('RapidOCR not available, text extraction will be limited')

        # Balatro-specific class mappings
        self.tooltip_classes = {'card_description', 'poker_card_description'}
        self.card_classes = {
            'poker_card_front',
            'joker_card',
            'planet_card',
            'tarot_card',
            'spectral_card',
            'poker_card_stack',
        }

        # Card information cache
        self.card_info_cache: Dict[str, Dict] = {}

        logger.info(
            'CardTooltipService initialized '
            f'(OCR: {self.ocr_enabled}, Hover: {screen_capture is not None}, '
            f'MultiDetector: {self.multi_detector is not None})'
        )

    def _check_ocr_availability(self) -> bool:
        """Check if RapidOCR is available."""
        try:
            from rapidocr import RapidOCR  # noqa: F401

            return True
        except ImportError:
            return False

    def _prepare_card_info(
        self, card: Detection, position_index: int
    ) -> Dict[str, Any]:
        """Create a baseline card info dictionary."""
        card_center_x, card_center_y = card.center
        hover_x = card_center_x + self.card_hover_offset[0]
        hover_y = card_center_y + self.card_hover_offset[1]

        return {
            'position_index': position_index,
            'card_detection': card,
            'card_class': card.class_name,
            'card_confidence': card.confidence,
            'card_bbox': card.bbox,
            'card_center': card.center,
            'hover_position': (hover_x, hover_y),
            'hover_screen_position': None,
            'description_text': '',
            'description_detected': False,
            'tooltip_match': None,
            'ocr_confidence': 0.0,
            'debug_image_path': None,
        }

    def _frame_to_screen_coordinates(
        self, x: float, y: float, frame_shape: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[int, int]:
        """Convert frame-relative coordinates to absolute screen coordinates."""
        if not self.screen_capture:
            return int(x), int(y)

        capture_region = self.screen_capture.get_capture_region()
        if not capture_region:
            return int(x), int(y)

        frame_width = None
        frame_height = None

        if frame_shape is not None and len(frame_shape) >= 2:
            frame_height, frame_width = frame_shape[:2]
        else:
            reference_frame = self.screen_capture.capture_once()
            if reference_frame is not None:
                frame_height, frame_width = reference_frame.shape[:2]

        if not frame_width or not frame_height:
            return int(x), int(y)

        scale_x = capture_region['width'] / frame_width
        scale_y = capture_region['height'] / frame_height

        screen_x = capture_region['left'] + int(x * scale_x)
        screen_y = capture_region['top'] + int(y * scale_y)

        return int(screen_x), int(screen_y)

    def hover_cards_sequence(
        self,
        hand_cards: List[Detection],
        capture_descriptions: bool = True,
        save_debug_images: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Hover over each card in sequence to capture descriptions.

        Args:
            hand_cards: List of card detections sorted by position
            capture_descriptions: Whether to capture and OCR descriptions
            save_debug_images: Whether to save debug images

        Returns:
            List of card information dictionaries
        """
        if not hand_cards:
            logger.warning('No hand cards provided for hovering')
            return []

        if not self.screen_capture or not self.mouse_controller:
            logger.error(
                'Screen capture or mouse controller not available for hovering'
            )
            return []

        logger.info(f'Starting hover sequence for {len(hand_cards)} cards')

        # Store original mouse position to restore later
        original_pos = self.mouse_controller.mouse.position

        card_infos = []

        try:
            reference_frame = self.screen_capture.capture_once()
            frame_shape = reference_frame.shape if reference_frame is not None else None

            for i, card in enumerate(hand_cards):
                logger.info(
                    f'Hovering over card {i + 1}/{len(hand_cards)}: {card.class_name}'
                )

                # Hover over card and capture description
                card_info = self.hover_card_and_capture(
                    card,
                    position_index=i,
                    capture_description=capture_descriptions,
                    save_debug_image=save_debug_images,
                    reference_frame_shape=frame_shape,
                )

                card_infos.append(card_info)

                # Brief pause between cards
                if i < len(hand_cards) - 1:
                    time.sleep(0.3)

            # Restore original mouse position
            self.mouse_controller.smooth_move_to(original_pos[0], original_pos[1])

        except Exception as e:
            logger.error(f'Error in hover sequence: {e}')
            # Try to restore mouse position even on error
            try:
                self.mouse_controller.mouse.position = original_pos
            except: # noqa: E722
                pass

        logger.info(
            f'Completed hover sequence, captured {len(card_infos)} card descriptions'
        )
        return card_infos

    def hover_card_and_capture(
        self,
        card: Detection,
        position_index: int = 0,
        capture_description: bool = True,
        save_debug_image: bool = False,
        reference_frame_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Hover over a single card and capture its description.

        Args:
            card: Card detection to hover over
            position_index: Index of card in hand (0-based)
            capture_description: Whether to capture description via OCR
            save_debug_image: Whether to save debug image

        Returns:
            Card information dictionary
        """
        # Calculate hover position (slightly above card center for better tooltip trigger)
        card_info = self._prepare_card_info(card, position_index)

        hover_x, hover_y = card_info['hover_position']
        screen_x, screen_y = self._frame_to_screen_coordinates(
            hover_x, hover_y, reference_frame_shape
        )
        card_info['hover_screen_position'] = (screen_x, screen_y)

        if not self.mouse_controller:
            logger.warning('Mouse controller not available for hovering')
            return card_info

        try:
            # Move to hover position
            if not self.mouse_controller.smooth_move_to(screen_x, screen_y):
                logger.warning(
                    f'Failed to move to card {position_index} hover position'
                )
                return card_info

            # Wait for tooltip to appear
            time.sleep(self.hover_duration)

            if capture_description and self.screen_capture:
                # Capture screen after hover
                hover_frame = self.screen_capture.capture_once()
                if hover_frame is not None:
                    # Process the frame to find and OCR description
                    description_info = self._process_hovered_card_frame(
                        hover_frame, card, position_index, save_debug_image
                    )
                    card_info.update(description_info)
                else:
                    logger.warning(
                        f'Failed to capture screen for card {position_index}'
                    )

        except Exception as e:
            logger.error(f'Error hovering over card {position_index}: {e}')

        return card_info

    def collect_card_infos(
        self,
        frame: np.ndarray,
        hand_cards: List[Detection],
        detections: Optional[List[Detection]] = None,
        auto_hover_missing: bool = True,
        save_debug_images: bool = False,
    ) -> List[Dict[str, Any]]:
        """Collect card information with optional hover fallback for missing descriptions."""

        if not hand_cards:
            return []

        if detections is None and self.multi_detector is not None:
            entity_detections = self.multi_detector.detect_entities(frame)
            ui_detections = self.multi_detector.detect_ui(frame)
            detections = entity_detections + ui_detections

        matches_by_card_bbox: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
        if detections:
            tooltip_matches = self.match_tooltips_to_cards(detections)
            enhanced_matches = self.create_tooltip_card_crops(frame, tooltip_matches)
            for match in enhanced_matches:
                matches_by_card_bbox[tuple(match['card'].bbox)] = match

        card_infos: List[Dict[str, Any]] = []
        hovered_any = False
        original_pos = None
        if self.mouse_controller:
            try:
                original_pos = self.mouse_controller.mouse.position
            except Exception:  # pragma: no cover - hardware specific
                original_pos = None

        for idx, card in enumerate(hand_cards):
            card_info = self._prepare_card_info(card, idx)
            hover_x, hover_y = card_info['hover_position']
            screen_x, screen_y = self._frame_to_screen_coordinates(
                hover_x, hover_y, frame.shape
            )
            card_info['hover_screen_position'] = (screen_x, screen_y)

            match = matches_by_card_bbox.get(tuple(card.bbox))

            if match:
                card_info['tooltip_match'] = match
                card_info['description_detected'] = True

                if self.ocr_enabled and match.get('tooltip_crop') is not None:
                    ocr_result = self._ocr_description_crop(match['tooltip_crop'])
                    card_info.update(ocr_result)

                if save_debug_images:
                    debug_path = self._save_debug_image(frame, match, idx)
                    card_info['debug_image_path'] = debug_path

                card_infos.append(card_info)
                continue

            if auto_hover_missing:
                hovered_any = True
                hover_result = self.hover_card_and_capture(
                    card,
                    position_index=idx,
                    capture_description=True,
                    save_debug_image=save_debug_images,
                    reference_frame_shape=frame.shape,
                )
                card_infos.append(hover_result)
            else:
                card_infos.append(card_info)

        if hovered_any and original_pos and self.mouse_controller:
            self.mouse_controller.smooth_move_to(original_pos[0], original_pos[1])

        return card_infos

    def format_card_info_for_llm(self, card_infos: List[Dict[str, Any]]) -> str:
        """
        Format card information for LLM prompt.

        Args:
            card_infos: List of card information dictionaries

        Returns:
            Formatted string for LLM prompt
        """
        if not card_infos:
            return 'No cards detected in hand.'

        formatted_lines = [f'CURRENT HAND ({len(card_infos)} cards):']

        for card_info in card_infos:
            card_line = f'Card {card_info["position_index"]}: {card_info["card_class"]} (confidence: {card_info["card_confidence"]:.2f})'

            if card_info['description_detected'] and card_info['description_text']:
                # Include OCR'd description text
                desc_text = card_info['description_text'][:200]  # Limit length
                card_line += f' - Description: {desc_text}'

                if card_info['ocr_confidence'] > 0:
                    card_line += f' (OCR confidence: {card_info["ocr_confidence"]:.2f})'

            formatted_lines.append(card_line)

        return '\n'.join(formatted_lines)

    def separate_tooltips_and_cards(
        self, detections: List[Detection]
    ) -> Dict[str, List[Detection]]:
        """
        Separate detections into tooltips and cards.

        Args:
            detections: All detections from YOLO

        Returns:
            Dictionary with 'tooltips' and 'cards' lists
        """
        tooltips = []
        cards = []

        for detection in detections:
            if detection.class_name in self.tooltip_classes:
                tooltips.append(detection)
            elif detection.class_name in self.card_classes:
                cards.append(detection)

        return {'tooltips': tooltips, 'cards': cards}

    def match_tooltips_to_cards(
        self, detections: List[Detection], max_distance: float = 300.0
    ) -> List[Dict[str, Any]]:
        """
        Match card tooltips to their corresponding cards.

        Args:
            detections: All detections from YOLO
            max_distance: Maximum distance for tooltip-card matching

        Returns:
            List of match dictionaries
        """
        separated = self.separate_tooltips_and_cards(detections)
        tooltips = separated['tooltips']
        cards = separated['cards']

        logger.info(f'Matching {len(tooltips)} tooltips to {len(cards)} cards')

        matches = []

        for tooltip in tooltips:
            # Find nearby cards for this tooltip
            nearby_cards = self._find_nearby_cards(tooltip, cards, max_distance)

            if nearby_cards:
                closest_card, distance = nearby_cards[0]  # Already sorted by distance

                match = {
                    'tooltip': tooltip,
                    'card': closest_card,
                    'distance': distance,
                    'tooltip_bbox': tooltip.bbox,
                    'card_bbox': closest_card.bbox,
                }

                matches.append(match)
                logger.debug(
                    f'Matched {tooltip.class_name} to {closest_card.class_name} '
                    f'(distance: {distance:.1f}px)'
                )

            else:
                logger.warning(f'No nearby cards found for {tooltip.class_name}')

        return matches

    def create_tooltip_card_crops(
        self, image: np.ndarray, matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create image crops for tooltip-card pairs.

        Args:
            image: Source image
            matches: List of tooltip-card matches

        Returns:
            List of match dictionaries with added crop data
        """
        enhanced_matches = []

        for match in matches:
            try:
                # Crop individual regions
                tooltip_crop = self.image_cropper.crop_detection(
                    image, match['tooltip']
                )
                card_crop = self.image_cropper.crop_detection(image, match['card'])

                # Create combined context crop
                combined_crop = self.image_cropper.create_combined_crop(
                    image, [match['tooltip_bbox'], match['card_bbox']], padding=20
                )

                enhanced_match = match.copy()
                enhanced_match.update(
                    {
                        'tooltip_crop': tooltip_crop,
                        'card_crop': card_crop,
                        'context_crop': combined_crop,
                        'tooltip_crop_shape': tooltip_crop.shape,
                        'card_crop_shape': card_crop.shape,
                        'context_crop_shape': combined_crop.shape,
                    }
                )

                enhanced_matches.append(enhanced_match)

            except Exception as e:
                logger.error(f'Failed to create crops for match: {e}')
                continue

        logger.info(f'Created crops for {len(enhanced_matches)} matches')
        return enhanced_matches

    def _process_hovered_card_frame(
        self,
        frame: np.ndarray,
        card: Detection,
        position_index: int,
        save_debug_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Process captured frame to extract card description.

        Args:
            frame: Captured frame during hover
            card: Original card detection
            position_index: Card position index
            save_debug_image: Whether to save debug image

        Returns:
            Dictionary with description information
        """
        description_info = {
            'description_text': '',
            'description_detected': False,
            'tooltip_match': None,
            'ocr_confidence': 0.0,
            'debug_image_path': None,
        }

        try:
            # Use the configured multi-detector to find description elements
            if self.multi_detector is None:
                logger.warning(
                    'Multi detector not available for hovered frame processing'
                )
                return description_info

            entity_detections = self.multi_detector.detect_entities(frame)
            ui_detections = self.multi_detector.detect_ui(frame)
            detections = entity_detections + ui_detections

            if not detections:
                logger.warning('No detections found in hovered frame')
                return description_info

            tooltip_matches = self.match_tooltips_to_cards(detections)
            enhanced_matches = self.create_tooltip_card_crops(frame, tooltip_matches)

            # Find the match for our hovered card
            hovered_card_match = self._find_matching_tooltip(enhanced_matches, card)

            if hovered_card_match:
                description_info['tooltip_match'] = hovered_card_match
                description_info['description_detected'] = True

                # Extract text from description area using OCR
                if (
                    self.ocr_enabled
                    and hovered_card_match.get('tooltip_crop') is not None
                ):
                    ocr_result = self._ocr_description_crop(
                        hovered_card_match['tooltip_crop']
                    )
                    description_info.update(ocr_result)

                # Save debug image if requested
                if save_debug_image:
                    debug_path = self._save_debug_image(
                        frame, hovered_card_match, position_index
                    )
                    description_info['debug_image_path'] = debug_path

            else:
                logger.warning(
                    f'No tooltip match found for hovered card {position_index}'
                )

        except Exception as e:
            logger.error(
                f'Error processing hovered frame for card {position_index}: {e}'
            )

        return description_info

    def _find_matching_tooltip(
        self, tooltip_matches: List[Dict[str, Any]], target_card: Detection
    ) -> Optional[Dict[str, Any]]:
        """
        Find the tooltip match that corresponds to the target card.

        Args:
            tooltip_matches: List of tooltip-card matches
            target_card: The card we're looking for a tooltip for

        Returns:
            Matching tooltip dictionary or None
        """
        if not tooltip_matches:
            return None

        # Find the closest match by comparing card positions
        best_match = None
        min_distance = float('inf')

        target_center = target_card.center

        for match in tooltip_matches:
            match_card = match['card']
            match_center = match_card.center

            # Calculate distance between card centers
            dx = target_center[0] - match_center[0]
            dy = target_center[1] - match_center[1]
            distance = (dx**2 + dy**2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                best_match = match

        # Only return match if it's reasonably close (within 100 pixels)
        if best_match and min_distance < 100:
            return best_match

        return None

    def _ocr_description_crop(self, description_crop: np.ndarray) -> Dict[str, Any]:
        """
        Perform OCR on a description crop using RapidOCR.

        Args:
            description_crop: Cropped image containing description text

        Returns:
            Dictionary with OCR results
        """
        ocr_result = {'description_text': '', 'ocr_confidence': 0.0}

        if not self.ocr_enabled or description_crop is None or self.ocr_engine is None:
            return ocr_result

        try:
            # Perform OCR using RapidOCR
            result = self.ocr_engine.run(description_crop)

            if result.success and result.text:
                ocr_result['description_text'] = result.text.strip()
                # RapidOCR doesn't provide individual confidence scores like EasyOCR,
                # but we can use success as a basic confidence indicator
                ocr_result['ocr_confidence'] = 0.8 if result.success else 0.0

                logger.info(
                    f"RapidOCR extracted text: '{ocr_result['description_text'][:50]}...' (success: {result.success})"
                )

        except Exception as e:
            logger.error(f'OCR processing failed: {e}')

        return ocr_result

    def _save_debug_image(
        self, frame: np.ndarray, tooltip_match: Dict[str, Any], position_index: int
    ) -> Optional[str]:
        """
        Save debug image showing card and tooltip match.

        Args:
            frame: Original frame
            tooltip_match: Tooltip match information
            position_index: Card position index

        Returns:
            Path to saved debug image or None
        """
        try:
            import cv2

            debug_frame = frame.copy()

            # Draw card bbox in green
            card_bbox = tooltip_match['card_bbox']
            cv2.rectangle(
                debug_frame,
                (int(card_bbox[0]), int(card_bbox[1])),
                (int(card_bbox[2]), int(card_bbox[3])),
                (0, 255, 0),
                2,
            )

            # Draw tooltip bbox in red
            tooltip_bbox = tooltip_match['tooltip_bbox']
            cv2.rectangle(
                debug_frame,
                (int(tooltip_bbox[0]), int(tooltip_bbox[1])),
                (int(tooltip_bbox[2]), int(tooltip_bbox[3])),
                (0, 0, 255),
                2,
            )

            # Save debug image
            debug_dir = Path('debug/card_hover')
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            debug_path = debug_dir / f'card_{position_index}_{timestamp}.jpg'

            cv2.imwrite(str(debug_path), debug_frame)
            logger.info(f'Saved debug image: {debug_path}')

            return str(debug_path)

        except Exception as e:
            logger.error(f'Failed to save debug image: {e}')
            return None

    def clear_cache(self):
        """Clear the card information cache."""
        self.card_info_cache.clear()
        logger.info('Card information cache cleared')

    def _find_nearby_cards(
        self, tooltip: Detection, cards: List[Detection], max_distance: float
    ) -> List[tuple]:
        """
        Find cards near a tooltip detection.

        Args:
            tooltip: Tooltip detection
            cards: List of card detections
            max_distance: Maximum distance to consider

        Returns:
            List of (card_detection, distance) tuples, sorted by distance
        """
        card_bboxes = [card.bbox for card in cards]

        nearby_with_distance = self.region_matcher.find_nearby_regions(
            tooltip.bbox, card_bboxes, max_distance, include_distance=True
        )

        # Convert back to Detection objects with distance
        nearby_cards = []
        for bbox, distance in nearby_with_distance:
            # Find the original Detection object
            for card in cards:
                if card.bbox == bbox:
                    nearby_cards.append((card, distance))
                    break

        return nearby_cards

    def get_processing_stats(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the tooltip-card matching process.

        Args:
            matches: List of matches from match_tooltips_to_cards

        Returns:
            Dictionary with statistics
        """
        if not matches:
            return {
                'total_matches': 0,
                'average_distance': 0,
                'min_distance': 0,
                'max_distance': 0,
                'tooltip_types': {},
                'card_types': {},
            }

        distances = [match['distance'] for match in matches]
        tooltip_types = {}
        card_types = {}

        for match in matches:
            tooltip_class = match['tooltip'].class_name
            card_class = match['card'].class_name

            tooltip_types[tooltip_class] = tooltip_types.get(tooltip_class, 0) + 1
            card_types[card_class] = card_types.get(card_class, 0) + 1

        return {
            'total_matches': len(matches),
            'average_distance': np.mean(distances),
            'min_distance': min(distances),
            'max_distance': max(distances),
            'tooltip_types': tooltip_types,
            'card_types': card_types,
        }
