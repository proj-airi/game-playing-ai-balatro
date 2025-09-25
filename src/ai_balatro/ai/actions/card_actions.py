"""Card action engine for executing card-based actions in Balatro.

This module has been refactored into smaller, more focused modules:
- detection_models.py: Data models for detection results
- card_detector.py: Card position detection functionality  
- visualizer.py: Detection result visualization
- button_detector.py: Button detection and recognition
- mouse_controller.py: Mouse control and movement
- card_action_engine.py: Main card action engine

This file serves as a compatibility layer by re-exporting the main classes.
"""

# Re-export main classes for backward compatibility
from .detection_models import ButtonDetection
from .card_detector import CardPositionDetector
from .visualizer import DetectionVisualizer
from .button_detector import ButtonDetector
from .mouse_controller import MouseController
from .card_action_engine import CardActionEngine

# Re-export schemas
from .schemas import CardAction, BUTTON_CONFIG

__all__ = [
    'ButtonDetection',
    'CardPositionDetector', 
    'DetectionVisualizer',
    'ButtonDetector',
    'MouseController',
    'CardActionEngine',
    'CardAction',
    'BUTTON_CONFIG',
]
