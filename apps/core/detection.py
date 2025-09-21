"""Detection data structures and utilities."""

from typing import Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection result data class."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def center(self) -> Tuple[int, int]:
        """Get bounding box center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Get bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        """Get bounding box width."""
        x1, _, x2, _ = self.bbox
        return x2 - x1

    @property
    def height(self) -> int:
        """Get bounding box height."""
        _, y1, _, y2 = self.bbox
        return y2 - y1

    def __repr__(self) -> str:
        return f'Detection({self.class_name}, {self.confidence:.3f}, {self.bbox})'
