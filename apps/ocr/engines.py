"""OCR engine wrappers with a common interface.

Engines implemented (optional imports):
- PaddleOCR
- Tesseract (pytesseract)
- RapidOCR

Each engine exposes:
- name: str
- init(): measures init time and prepares model
- run(image): returns dict with text, success, timings and raw details

Note: PaddleOCR is optional and will be skipped if not installed.
"""

from __future__ import annotations

from PIL import ImageFile
from dataclasses import dataclass
from typing import Any, Dict
import time
import cv2
from rapidocr import RapidOCR

from utils.logger import get_logger

logger = get_logger(__name__)

def _to_rgb(image):
    if image is None:
        return image
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

@dataclass
class OcrResult:
    name: str
    text: str
    success: bool
    init_time: float
    ocr_time: float
    total_time: float
    raw_results: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "success": self.success,
            "init_time": self.init_time,
            "ocr_time": self.ocr_time,
            "total_time": self.total_time,
            "raw_results": self.raw_results,
        }

class BaseEngine:
    name = "base"

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._inited = False
        self._init_time = 0.0
        self._init_error: str | None = None

    def init(self) -> None:
        self._inited = True
        self._init_error = None

    def run(self, image: ImageFile.ImageFile | str) -> OcrResult:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return self._inited and self._init_error is None

    @property
    def init_error(self) -> str | None:
        return self._init_error

class PaddleOCREngine(BaseEngine):
    name = "PaddleOCR"

    def init(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore

            t0 = time.time()

            self._ocr = PaddleOCR()
            self._init_time = time.time() - t0
            self._inited = True
            self._init_error = None
        except Exception as e:  # noqa: BLE001
            logger.error(f"PaddleOCR init error: {e}")
            self._init_time = time.time() - t0 if "t0" in locals() else 0.0
            self._ocr = None
            self._inited = False
            self._init_error = str(e)

    def run(self, image) -> OcrResult:
        if not getattr(self, "_ocr", None):
            return OcrResult(
                name=self.name,
                text=(f"ERROR: paddleocr not available ({getattr(self, '_init_error', 'not installed')})"),
                success=False,
                init_time=self._init_time,
                ocr_time=0.0,
                total_time=self._init_time,
                raw_results=[],
            )

        t0 = time.time()
        results = None
        try:
            results = self._ocr.predict(image)
        except Exception as e:  # noqa: BLE001
            logger.error(f"PaddleOCR run error: {e}")
            ocr_time = time.time() - t0
            return OcrResult(
                name=self.name,
                text=f"ERROR: {e}",
                success=False,
                init_time=self._init_time,
                ocr_time=ocr_time,
                total_time=self._init_time + ocr_time,
                raw_results=None,
            )

        ocr_time = time.time() - t0
        texts = []
        for res in results:
            texts.append("\n".join(res.json['res']['rec_texts']).strip())
        text_joined = " ".join(texts).strip()

        return OcrResult(
            name=self.name,
            text=text_joined,
            success=bool(text_joined),
            init_time=self._init_time,
            ocr_time=ocr_time,
            total_time=self._init_time + ocr_time,
            raw_results=results,
        )

class TesseractEngine(BaseEngine):
    name = "Tesseract"

    def init(self) -> None:
        # pytesseract has no heavy init, but record as 0
        try:
            import pytesseract  # noqa: F401

            t0 = time.time()
            # Ensure native binary is reachable; raises if missing
            pytesseract.get_tesseract_version()

            self._init_time = time.time() - t0
            self._inited = True
            self._init_error = None
        except Exception as e:  # noqa: BLE001
            self._init_time = time.time() - t0 if "t0" in locals() else 0.0
            self._inited = False
            self._init_error = str(e)

    def run(self, image) -> OcrResult:
        try:
            import pytesseract  # type: ignore
        except Exception as e:  # noqa: BLE001
            logger.error(f"Tesseract init error: {e}")
            return OcrResult(
                name=self.name,
                text=f"ERROR: pytesseract not available ({e})",
                success=False,
                init_time=self._init_time,
                ocr_time=0.0,
                total_time=self._init_time,
                raw_results={},
            )

        def _map_langs(spec: str | None) -> str:
            if not spec:
                return "eng"
            mapping = {
                "en": "eng",
                "english": "eng",
                "ch_sim": "chi_sim",
                "zh_cn": "chi_sim",
                "zh": "chi_sim",
                "ch_tra": "chi_tra",
                "zh_tw": "chi_tra",
            }
            parts = [s.strip() for s in spec.split(",") if s.strip()]
            conv = [mapping.get(p, p) for p in parts]
            return "+".join(conv) if conv else "eng"

        rgb = _to_rgb(image)
        t0 = time.time()
        config = "--oem 3 --psm 6"  # good default for UI text blocks
        lang_param = _map_langs(self.lang)
        try:
            text = pytesseract.image_to_string(rgb, config=config, lang=lang_param).strip()
            ocr_time = time.time() - t0
            return OcrResult(
                name=self.name,
                text=text,
                success=bool(text),
                init_time=self._init_time,
                ocr_time=ocr_time,
                total_time=self._init_time + ocr_time,
                raw_results={"config": config, "lang": lang_param},
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Tesseract run error: {e}")
            ocr_time = time.time() - t0
            return OcrResult(
                name=self.name,
                text=f"ERROR: {e}",
                success=False,
                init_time=self._init_time,
                ocr_time=ocr_time,
                total_time=self._init_time + ocr_time,
                raw_results={"config": config, "lang": lang_param},
            )

class RapidOCREngine(BaseEngine):
    name = "RapidOCR"

    def init(self) -> None:
        try:
            t0 = time.time()
            self._ocr = RapidOCR()
            self._init_time = time.time() - t0
            self._inited = True
            self._init_error = None
        except Exception as e:  # noqa: BLE001
            self._init_time = time.time() - t0 if "t0" in locals() else 0.0
            self._ocr = None
            self._inited = False
            self._init_error = str(e)

    def run(self, image) -> OcrResult:
        if not getattr(self, "_ocr", None):
            return OcrResult(
                name=self.name,
                text=(f"ERROR: rapidocr not available ({getattr(self, '_init_error', 'not installed')})"),
                success=False,
                init_time=self._init_time,
                ocr_time=0.0,
                total_time=self._init_time,
                raw_results=[],
            )

        t0 = time.time()
        results = None
        try:
            results = self._ocr(image)
        except Exception as e:  # noqa: BLE001
            ocr_time = time.time() - t0
            return OcrResult(
                name=self.name,
                text=f"ERROR: {e}",
                success=False,
                init_time=self._init_time,
                ocr_time=ocr_time,
                total_time=self._init_time + ocr_time,
                raw_results=None,
            )

        ocr_time = time.time() - t0
        texts = []

        for line in results.to_json():
            for text in line['txt']:
                texts.append(text)
            texts.append("\n")

        text_joined = "".join(texts).strip()

        return OcrResult(
            name=self.name,
            text=text_joined,
            success=bool(text_joined),
            init_time=self._init_time,
            ocr_time=ocr_time,
            total_time=self._init_time + ocr_time,
            raw_results=results,
        )

def available_engines(lang: str = "en") -> list[BaseEngine]:
    """Instantiate engines that can at least import.

    We always return instances, but their `run` will report an error message if
    the backend is missing. This keeps the benchmark comparable across envs.
    """
    engines: list[BaseEngine] = [PaddleOCREngine(lang), TesseractEngine(lang), RapidOCREngine(lang)]
    for e in engines:
        e.init()
    return engines
