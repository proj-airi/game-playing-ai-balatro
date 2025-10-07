"""Fast game state extraction pipeline with batched capture and OCR."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.detection import Detection
from ..core.multi_yolo_detector import MultiYOLODetector
from ..core.screen_capture import ScreenCapture
from ..utils.logger import get_logger
from .ui_text_service import UITextExtractionService
from .card_tooltip_service import CardTooltipService
from ..ai.actions.mouse_controller import MouseController

logger = get_logger(__name__)


@dataclass
class CardHoverFrame:
    """Stores data collected during a hover sweep."""

    card_index: int
    frame: np.ndarray


class GameStateExtractionService:
    """Service responsible for capturing and enriching Balatro game state snapshots."""

    def __init__(
        self,
        screen_capture: ScreenCapture,
        multi_detector: MultiYOLODetector,
        mouse_controller: Optional[MouseController] = None,
        card_tooltip_service: Optional[CardTooltipService] = None,
        ui_text_service: Optional[UITextExtractionService] = None,
        hover_dwell_time: float = 0.25,
        sweep_move_duration: float = 0.2,
        max_workers: int = 4,
    ) -> None:
        self.screen_capture = screen_capture
        self.multi_detector = multi_detector
        self.mouse_controller = mouse_controller
        self.card_tooltip_service = card_tooltip_service
        self.ui_text_service = ui_text_service or UITextExtractionService()
        self.hover_dwell_time = hover_dwell_time
        self.sweep_move_duration = sweep_move_duration
        self._detector_lock = threading.Lock()

    def capture_state(
        self,
        *,
        frame: Optional[np.ndarray] = None,
        entities_detection: Optional[Sequence[Detection]] = None,
        ui_detection: Optional[Sequence[Detection]] = None,
        capture_card_descriptions: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Capture a full game state snapshot with optional card description OCR."""
        frame = frame if frame is not None else self.screen_capture.capture_once()
        if frame is None:
            logger.error('Game state capture failed: no frame available')
            return None

        entities = (
            list(entities_detection)
            if entities_detection is not None
            else self.multi_detector.detect_entities(frame)
        )
        ui_elements = (
            list(ui_detection)
            if ui_detection is not None
            else self.multi_detector.detect_ui(frame)
        )

        game_state, hand_cards = self._build_base_state(frame, entities, ui_elements)

        self._enrich_ui_text(frame, ui_elements, game_state)

        if capture_card_descriptions and hand_cards:
            self._enrich_card_descriptions(frame, entities, ui_elements, hand_cards, game_state)

        return game_state

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _build_base_state(
        self,
        frame: np.ndarray,
        entities_detection: Sequence[Detection],
        ui_detection: Sequence[Detection],
    ) -> Tuple[Dict[str, Any], List[Detection]]:
        """Create baseline game state payload from detections."""
        timestamp = time.time()
        cards: List[Dict[str, Any]] = []
        jokers: List[Dict[str, Any]] = []
        buttons: List[Dict[str, Any]] = []
        hand_cards: List[Detection] = []

        for detection in entities_detection:
            name = detection.class_name.lower()
            x1, y1, x2, y2 = detection.bbox

            if 'card' in name and 'tooltip' not in name:
                hand_cards.append(detection)
                cards.append(
                    {
                        'index': len(cards),
                        'class_name': detection.class_name,
                        'confidence': detection.confidence,
                        'position': [x1, y1, x2, y2],
                        'center': detection.center,
                        'width': detection.width,
                        'height': detection.height,
                        'description_text': '',
                        'description_detected': False,
                        'parsed_description': None,
                        'ocr_confidence': 0.0,
                    }
                )
            elif 'joker' in name:
                jokers.append(
                    {
                        'index': len(jokers),
                        'class_name': detection.class_name,
                        'confidence': detection.confidence,
                        'position': [x1, y1, x2, y2],
                        'center': detection.center,
                        'width': detection.width,
                        'height': detection.height,
                    }
                )

        for detection in ui_detection:
            if 'button' in detection.class_name.lower():
                x1, y1, x2, y2 = detection.bbox
                buttons.append(
                    {
                        'class_name': detection.class_name,
                        'confidence': detection.confidence,
                        'position': [x1, y1, x2, y2],
                        'center': detection.center,
                        'width': detection.width,
                        'height': detection.height,
                    }
                )

        cards.sort(key=lambda c: c['position'][0])
        hand_cards.sort(key=lambda d: d.bbox[0])

        for idx, card in enumerate(cards):
            card['index'] = idx

        game_phase = self._infer_game_phase(buttons)

        game_state = {
            'timestamp': timestamp,
            'entities_raw': list(entities_detection),
            'ui_elements_raw': list(ui_detection),
            'cards': cards,
            'jokers': jokers,
            'ui_buttons': buttons,
            'game_phase': game_phase,
            'card_descriptions': [],
            'ui_text_elements': [],
            'ui_text_values': {},
        }

        return game_state, hand_cards

    def _infer_game_phase(self, buttons: Sequence[Dict[str, Any]]) -> str:
        names = [btn['class_name'].lower() for btn in buttons]
        playing_keywords = ('play', 'discard', 'hand', 'hold')

        if any(any(keyword in btn for keyword in playing_keywords) for btn in names):
            return 'playing'
        if any('shop' in btn for btn in names):
            return 'shop'
        if any('next' in btn for btn in names):
            return 'transition'
        return 'unknown'

    def _enrich_ui_text(
        self,
        frame: np.ndarray,
        ui_detection: Sequence[Detection],
        game_state: Dict[str, Any],
    ) -> None:
        if not ui_detection:
            return

        try:
            ui_text_results = self.ui_text_service.extract(frame, ui_detection)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'UI text extraction failed: {exc}')
            return

        if not ui_text_results:
            return

        text_entries: List[Dict[str, Any]] = []
        value_map: Dict[str, Dict[str, Any]] = {}

        for result in ui_text_results:
            entry = {
                'class_name': result.class_name,
                'text': result.text,
                'position': list(result.bbox),
                'detection_confidence': result.detection_confidence,
                'ocr_success': result.ocr_success,
                'ocr_confidence': result.ocr_confidence,
            }
            text_entries.append(entry)

            existing = value_map.get(result.class_name)
            if (
                existing is None
                or entry['ocr_confidence'] > existing.get('ocr_confidence', 0.0)
                or (
                    entry['ocr_confidence'] == existing.get('ocr_confidence', 0.0)
                    and entry['detection_confidence']
                    > existing.get('detection_confidence', 0.0)
                )
            ):
                value_map[result.class_name] = {
                    'text': entry['text'],
                    'ocr_confidence': entry['ocr_confidence'],
                    'detection_confidence': entry['detection_confidence'],
                    'position': entry['position'],
                }

        game_state['ui_text_elements'] = text_entries
        game_state['ui_text_values'] = value_map

    def _enrich_card_descriptions(
        self,
        base_frame: np.ndarray,
        entities_detection: Sequence[Detection],
        ui_detection: Sequence[Detection],
        hand_cards: Sequence[Detection],
        game_state: Dict[str, Any],
    ) -> None:
        if not self.card_tooltip_service or not self.mouse_controller:
            logger.debug('Card tooltip service or mouse controller missing; skipping fast hover')
            return

        ordered = list(hand_cards)
        if not ordered:
            return

        sweep_plan = self._plan_card_sweep(base_frame, ordered)
        if not sweep_plan:
            logger.debug('Unable to plan card sweep; skipping fast hover')
            return

        hover_frames: Dict[int, np.ndarray] = {}
        index_lookup = [card_index for card_index, _ in sweep_plan]
        positions = [coords for _, coords in sweep_plan]

        def capture_callback(step_index: int, _x: int, _y: int) -> None:
            card_index = index_lookup[min(step_index, len(index_lookup) - 1)]
            frame = self.screen_capture.capture_once()
            if frame is not None:
                hover_frames[card_index] = frame

        sweep_success = self.mouse_controller.sweep_path(
            positions,
            dwell_time=self.hover_dwell_time,
            move_duration=self.sweep_move_duration,
            capture_callback=capture_callback,
        )

        if not sweep_success or not hover_frames:
            logger.debug('Hover sweep produced no frames; skipping description OCR')
            return

        combined_detections = list(entities_detection) + list(ui_detection)

        card_descriptions: List[Dict[str, Any]] = [
            {
                'description_text': '',
                'description_detected': False,
                'ocr_confidence': 0.0,
                'parsed_description': None,
            }
            for _ in ordered
        ]

        for card_index, frame in hover_frames.items():
            if card_index >= len(ordered):
                continue

            card = ordered[card_index]
            description_info = self._process_hover_frame(card, frame, card_index)
            card_descriptions[card_index] = description_info

        game_state['card_descriptions'] = card_descriptions

        for idx, desc in enumerate(card_descriptions):
            if idx >= len(game_state['cards']):
                continue

            card_entry = game_state['cards'][idx]
            card_entry['description_text'] = desc.get('description_text', '')
            card_entry['description_detected'] = desc.get('description_detected', False)
            card_entry['parsed_description'] = desc.get('parsed_description')
            card_entry['ocr_confidence'] = desc.get('ocr_confidence', 0.0)

    def _plan_card_sweep(
        self, frame: np.ndarray, hand_cards: Sequence[Detection]
    ) -> List[Tuple[int, Tuple[int, int]]]:
        if not self.card_tooltip_service:
            return []

        plan: List[Tuple[int, Tuple[int, int]]] = []
        for idx, detection in enumerate(hand_cards):
            try:
                info = self.card_tooltip_service._prepare_card_info(detection, idx)
                hover_x, hover_y = info['hover_position']
                screen_x, screen_y = self.card_tooltip_service._frame_to_screen_coordinates(
                    hover_x, hover_y, frame.shape
                )
                plan.append((idx, (screen_x, screen_y)))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f'Failed to prepare sweep position for card {idx}: {exc}')

        plan.sort(key=lambda item: item[1][0])
        return plan

    def _process_hover_frame(
        self, card: Detection, frame: np.ndarray, position_index: int
    ) -> Dict[str, Any]:
        if not self.card_tooltip_service:
            return {
                'description_text': '',
                'description_detected': False,
                'ocr_confidence': 0.0,
                'parsed_description': None,
            }

        with self._detector_lock:
            try:
                return self.card_tooltip_service._process_hovered_card_frame(
                    frame, card, position_index, save_debug_image=False
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f'Failed to process hovered frame for card {position_index}: {exc}')
                return {
                    'description_text': '',
                    'description_detected': False,
                    'ocr_confidence': 0.0,
                    'parsed_description': None,
                }

    def _fallback_missing_cards(
        self,
        frame: np.ndarray,
        hand_cards: Sequence[Detection],
        detections: Sequence[Detection],
        card_descriptions: List[Dict[str, Any]],
    ) -> None:
        if not self.card_tooltip_service:
            return

        missing_indices = [
            idx
            for idx, desc in enumerate(card_descriptions)
            if not desc.get('description_detected')
        ]

        if not missing_indices:
            return

        try:
            fallback_infos = self.card_tooltip_service.collect_card_infos(
                frame,
                list(hand_cards),
                detections=list(detections),
                auto_hover_missing=True,
                save_debug_images=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'Fallback card description collection failed: {exc}')
            return

        for idx in missing_indices:
            if idx < len(fallback_infos):
                card_descriptions[idx] = fallback_infos[idx]

