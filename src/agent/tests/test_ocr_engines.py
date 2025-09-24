"""Smoke tests for OCR engines to validate image I/O plumbing."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from ..ocr.engines import (
    BaseEngine,
    PaddleOCREngine,
    RapidOCREngine,
    TesseractEngine,
)

@pytest.fixture(name='sample_text_image')
def fixture_sample_text_image() -> np.ndarray:
    """Create a simple synthetic image with a short text snippet."""

    canvas = np.full((160, 480, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, 'Balatro', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return canvas


@pytest.mark.parametrize(
    'engine_cls',
    [PaddleOCREngine, TesseractEngine, RapidOCREngine],
)
def test_ocr_engine_smoke(
    engine_cls: type[BaseEngine], sample_text_image: np.ndarray
) -> None:
    """Each engine should either succeed or surface a clear availability error."""

    engine = engine_cls()
    engine.init()
    result = engine.run(sample_text_image)

    if engine.available:
        assert result.success, f'{engine.name} failed with text={result.text!r}'
        assert result.text, f'{engine.name} returned empty text'
    else:
        assert not result.success
        assert result.text.upper().startswith('ERROR'), result.text
