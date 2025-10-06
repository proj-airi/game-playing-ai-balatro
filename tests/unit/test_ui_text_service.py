"""Unit tests for UI text extraction service and snapshot integration."""

from __future__ import annotations

import numpy as np
import pytest

from ai_balatro.core.detection import Detection
from ai_balatro.ocr.engines import OcrResult
from ai_balatro.services.game_state_extraction import GameStateExtractionService
from ai_balatro.services.ui_text_service import UITextExtractionService


class StubRawResult:
    """Simple stub that mimics RapidOCR JSON output."""

    def __init__(self, score: float) -> None:
        self._score = score

    def to_json(self):  # type: ignore[override]
        return [{'score': self._score}]


class StubOCREngine:
    """Stub that captures calls and returns deterministic OCR results."""

    def __init__(self, text: str, score: float = 0.85) -> None:
        self.text = text
        self.score = score
        self.available = True
        self.calls = []

    def run(self, image):  # noqa: ANN001 - signature matches RapidOCREngine
        self.calls.append(image.shape)
        return OcrResult(
            name='stub',
            text=self.text,
            success=bool(self.text),
            init_time=0.0,
            ocr_time=0.01,
            total_time=0.01,
            raw_results=StubRawResult(self.score),
        )


class EmptyRawResult:
    """Stub raw payload whose to_json returns None."""

    def to_json(self):  # type: ignore[override]
        return None


class StubScreenCapture:
    """Screen capture stub returning pre-defined frames."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0

    def capture_once(self):
        if not self._frames:
            return None
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        if self._index < len(self._frames) - 1:
            self._index += 1
        return frame

    def get_capture_region(self):  # pragma: no cover - not required for tests
        return None


class StubMultiDetector:
    """Multi-detector stub returning fixed detections."""

    def __init__(self, entities, ui):
        self._entities = list(entities)
        self._ui = list(ui)

    def detect_entities(self, _image, *args, **kwargs):  # noqa: ANN001
        return list(self._entities)

    def detect_ui(self, _image, *args, **kwargs):  # noqa: ANN001
        return list(self._ui)


class EmptyPayloadEngine(StubOCREngine):
    """Engine returning success with empty payload to exercise fallbacks."""

    def __init__(self):
        super().__init__(text='value', score=0.0)

    def run(self, image):
        self.calls.append(image.shape)
        return OcrResult(
            name='stub',
            text='value',
            success=True,
            init_time=0.0,
            ocr_time=0.0,
            total_time=0.0,
            raw_results=EmptyRawResult(),
        )


def test_ui_text_service_extracts_text_with_stub_engine():
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    detection = Detection(
        class_id=0,
        class_name='ui_score_chips',
        confidence=0.92,
        bbox=(5, 6, 25, 18),
    )

    engine = StubOCREngine(text='12345', score=0.9)
    service = UITextExtractionService(ocr_engine=engine, scale_factor=1.0)

    results = service.extract(frame, [detection])

    assert len(results) == 1
    result = results[0]
    assert result.class_name == 'ui_score_chips'
    assert result.text == '12345'
    assert pytest.approx(result.ocr_confidence, rel=1e-3) == 0.9
    assert engine.calls, 'Expected OCR engine to be invoked'


def test_extract_game_state_populates_ui_text():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    detections = [
        Detection(
            class_id=1,
            class_name='ui_score_current',
            confidence=0.88,
            bbox=(10, 5, 40, 20),
        )
    ]

    engine = StubOCREngine(text='777', score=0.95)
    service = UITextExtractionService(ocr_engine=engine, scale_factor=1.0)

    extractor = GameStateExtractionService(
        screen_capture=StubScreenCapture([frame]),
        multi_detector=StubMultiDetector([], detections),
        mouse_controller=None,
        card_tooltip_service=None,
        ui_text_service=service,
        hover_dwell_time=0.1,
        sweep_move_duration=0.1,
    )

    game_state = extractor.capture_state(
        frame=frame,
        entities_detection=[],
        ui_detection=detections,
        capture_card_descriptions=False,
    )

    assert game_state['ui_text_elements'], 'Expected UI text elements in game state'
    ui_entry = game_state['ui_text_elements'][0]
    assert ui_entry['class_name'] == 'ui_score_current'
    assert ui_entry['text'] == '777'

    tracked = game_state['ui_text_values']['ui_score_current']
    assert tracked['text'] == '777'
    assert pytest.approx(tracked['ocr_confidence'], rel=1e-3) == 0.95


def test_game_state_extractor_without_card_services():
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    card_detection = Detection(
        class_id=2,
        class_name='poker_card_front',
        confidence=0.9,
        bbox=(2, 5, 22, 30),
    )

    extractor = GameStateExtractionService(
        screen_capture=StubScreenCapture([frame]),
        multi_detector=StubMultiDetector([card_detection], []),
        mouse_controller=None,
        card_tooltip_service=None,
        ui_text_service=UITextExtractionService(),
    )

    state = extractor.capture_state(
        frame=frame,
        entities_detection=[card_detection],
        ui_detection=[],
        capture_card_descriptions=True,
    )

    assert state['cards'], 'Card list should be populated'
    assert state['card_descriptions'] == []


def test_handles_empty_ocr_payload():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    detection = Detection(
        class_id=1,
        class_name='ui_score_mult',
        confidence=0.5,
        bbox=(0, 0, 5, 5),
    )

    engine = EmptyPayloadEngine()
    service = UITextExtractionService(ocr_engine=engine, scale_factor=1.0)

    results = service.extract(frame, [detection])

    assert results
    assert results[0].ocr_confidence == 0.0
    assert engine.calls, 'Expected OCR engine to be invoked'
