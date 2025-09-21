"""
Legacy YOLO detector module - DEPRECATED

This file is kept for backward compatibility.
New code should use: from core.yolo_detector import YOLODetector
                     from core.detection import Detection
"""

import warnings
import sys
from pathlib import Path

# Add path and import from new location
sys.path.insert(0, str(Path(__file__).parent))
from core.yolo_detector import YOLODetector as NewYOLODetector
from core.detection import Detection as NewDetection

warnings.warn(
    "Importing from yolo_detector.py is deprecated. Use 'from core.yolo_detector import YOLODetector' "
    "and 'from core.detection import Detection' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for compatibility
YOLODetector = NewYOLODetector
Detection = NewDetection

# Legacy test function
def test_yolo_detector():
    """Test YOLO detector (legacy)."""
    warnings.warn("Use the new API instead", DeprecationWarning)
    
    from utils.path_utils import find_model_file
    from config.settings import settings
    
    model_path = find_model_file(settings.model_search_paths)
    if not model_path:
        print(f"Model file not found")
        return
    
    # Create detector
    detector = YOLODetector(model_path, use_onnx=False)
    print(f"Detector initialized with {len(detector.class_names)} classes")


if __name__ == "__main__":
    test_yolo_detector()
