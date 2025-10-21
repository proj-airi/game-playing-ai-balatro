"""Service for extracting dynamic UI text values using OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.detection import Detection
from ..ocr.engines import RapidOCREngine, OcrResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


DYNAMIC_UI_CLASSES = (
    'ui_card_value',
    'ui_data_cash',
    'ui_data_discards_left',
    'ui_data_hands_left',
    'ui_round_ante_current',
    'ui_round_ante_left',
    'ui_round_round_current',
    'ui_round_round_left',
    'ui_score_chips',
    'ui_score_current',
    'ui_score_mult',
    'ui_score_round_score',
    'ui_score_target_score',
)


@dataclass
class UITextExtraction:
    """Represents OCR results for a UI detection."""

    class_name: str
    bbox: Tuple[int, int, int, int]
    detection_confidence: float
    text: str
    ocr_success: bool
    ocr_confidence: float
    ocr_result: Optional[OcrResult]


class UITextExtractionService:
    """Runs OCR over detected UI elements to surface dynamic values."""

    def __init__(
        self,
        target_classes: Optional[Sequence[str]] = None,
        ocr_engine: Optional[RapidOCREngine] = None,
        scale_factor: float = 2.0,
    ) -> None:
        self.target_classes = (
            tuple(target_classes) if target_classes else DYNAMIC_UI_CLASSES
        )
        self.target_class_set = {cls.lower() for cls in self.target_classes}
        self.scale_factor = max(scale_factor, 1.0)

        if ocr_engine is not None:
            self.ocr_engine = ocr_engine
            self._owns_engine = False
        else:
            engine = RapidOCREngine()
            engine.init()
            self.ocr_engine = engine
            self._owns_engine = True

        self.ocr_available = getattr(self.ocr_engine, 'available', False)
        if not self.ocr_available:
            logger.warning('RapidOCR engine unavailable, UI text extraction disabled')

    def extract(
        self,
        frame: np.ndarray,
        detections: Iterable[Detection],
    ) -> List[UITextExtraction]:
        """Extract OCR text for target UI detections."""
        if frame is None:
            logger.warning('UI text extraction skipped: frame is None')
            return []

        if frame.size == 0:
            logger.warning('UI text extraction skipped: empty frame data')
            return []

        if not self.ocr_available:
            return []

        results: List[UITextExtraction] = []
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            if detection.class_name.lower() not in self.target_class_set:
                continue

            crop = self._crop_region(frame, detection.bbox, frame_width, frame_height)
            if crop is None:
                continue

            processed = self._preprocess_crop(crop)

            try:
                ocr_result = self.ocr_engine.run(processed)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    'RapidOCR failed on %s bbox=%s: %s',
                    detection.class_name,
                    detection.bbox,
                    exc,
                )
                continue

            text = ocr_result.text.strip() if ocr_result.text else ''
            confidence = self._infer_confidence(ocr_result)

            results.append(
                UITextExtraction(
                    class_name=detection.class_name,
                    bbox=detection.bbox,
                    detection_confidence=detection.confidence,
                    text=text,
                    ocr_success=ocr_result.success,
                    ocr_confidence=confidence,
                    ocr_result=ocr_result,
                )
            )

        return results

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Upscale crop to improve OCR accuracy."""
        if self.scale_factor == 1.0:
            return crop

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return crop

        new_size = (
            max(int(width * self.scale_factor), 1),
            max(int(height * self.scale_factor), 1),
        )

        return cv2.resize(crop, new_size, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _crop_region(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            logger.debug('Skipping UI crop with non-positive size: %s', bbox)
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            logger.debug('Skipping UI crop with empty data: %s', bbox)
            return None

        return np.ascontiguousarray(crop)

    @staticmethod
    def _infer_confidence(ocr_result: OcrResult) -> float:
        """Approximate OCR confidence from backend metadata."""
        if not ocr_result.success:
            return 0.0

        raw = getattr(ocr_result, 'raw_results', None)
        if raw is None:
            return 0.0

        try:
            json_payload = raw.to_json() if hasattr(raw, 'to_json') else raw
        except Exception:  # noqa: BLE001 - backend specific
            logger.debug('Failed to read OCR confidence payload', exc_info=True)
            return 0.0

        if not json_payload:
            return 0.0

        if isinstance(json_payload, dict):
            iterable = [json_payload]
        elif isinstance(json_payload, Sequence) and not isinstance(
            json_payload, (str, bytes)
        ):
            iterable = json_payload
        else:
            return 0.0

        confidences: List[float] = []
        for block in iterable:
            if not isinstance(block, dict):
                continue

            score = block.get('score')
            if isinstance(score, (int, float)):
                confidences.append(float(score))

            rec_scores = block.get('rec_scores')
            if isinstance(rec_scores, Sequence):
                for value in rec_scores:
                    if isinstance(value, (int, float)):
                        confidences.append(float(value))

        if not confidences:
            return 0.0

        return float(sum(confidences) / len(confidences))

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        if getattr(self, '_owns_engine', False):
            try:
                del self.ocr_engine
            except Exception:
                pass
