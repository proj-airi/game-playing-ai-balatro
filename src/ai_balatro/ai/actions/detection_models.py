"""Data models for detection results in card actions."""

from dataclasses import dataclass
from ...core.detection import Detection


@dataclass
class ButtonDetection(Detection):
    """Button detection result, extends Detection class with button type information."""

    button_type: str = 'unknown'
