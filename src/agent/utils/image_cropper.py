"""Generic image cropping utilities for computer vision tasks."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from ..core.detection import Detection
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ImageCropper:
    """Generic utility for cropping detected regions from images."""

    def __init__(self, padding_pixels: int = 10):
        """
        Initialize image cropper.

        Args:
            padding_pixels: Default extra pixels to include around detected regions
        """
        self.padding_pixels = padding_pixels

    def crop_detection_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: Optional[int] = None,
    ) -> np.ndarray:
        """
        Crop a bounding box region from image.

        Args:
            image: Source image (BGR format)
            bbox: Bounding box as (x1, y1, x2, y2)
            padding: Extra pixels around bbox (uses default if None)

        Returns:
            Cropped image region
        """
        if padding is None:
            padding = self.padding_pixels

        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Add padding while staying within image bounds
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        # Crop the region
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        return cropped

    def crop_detection(
        self, image: np.ndarray, detection: Detection, padding: Optional[int] = None
    ) -> np.ndarray:
        """
        Crop a detected region from image.

        Args:
            image: Source image
            detection: Detection object with bounding box
            padding: Extra pixels around detection

        Returns:
            Cropped image region
        """
        return self.crop_detection_bbox(image, detection.bbox, padding)

    def crop_multiple_regions(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        padding: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Crop multiple bounding box regions.

        Args:
            image: Source image
            bboxes: List of bounding boxes as (x1, y1, x2, y2)
            padding: Extra pixels around each bbox

        Returns:
            List of cropped image regions
        """
        crops = []
        for bbox in bboxes:
            try:
                crop = self.crop_detection_bbox(image, bbox, padding)
                crops.append(crop)
            except Exception as e:
                logger.warning(f'Failed to crop bbox {bbox}: {e}')
                crops.append(None)

        return crops

    def create_combined_crop(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        padding: int = 10,
    ) -> np.ndarray:
        """
        Create a crop that encompasses multiple bounding boxes.

        Args:
            image: Source image
            bboxes: List of bounding boxes to include
            padding: Extra padding around the combined area

        Returns:
            Cropped image containing all bboxes
        """
        if not bboxes:
            return image

        # Find bounding box that encompasses all regions
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[2] for bbox in bboxes)
        max_y = max(bbox[3] for bbox in bboxes)

        combined_bbox = (min_x, min_y, max_x, max_y)
        return self.crop_detection_bbox(image, combined_bbox, padding)

    def convert_bgr_to_rgb_pil(self, bgr_image: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR image to PIL RGB Image."""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)


class RegionMatcher:
    """Generic utility for matching detected regions based on spatial proximity."""

    @staticmethod
    def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)

    @staticmethod
    def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_nearby_regions(
        self,
        target_bbox: Tuple[int, int, int, int],
        candidate_bboxes: List[Tuple[int, int, int, int]],
        max_distance: float = 200.0,
        include_distance: bool = False,
    ) -> List[Any]:
        """
        Find regions near a target region.

        Args:
            target_bbox: Target bounding box
            candidate_bboxes: List of candidate bounding boxes
            max_distance: Maximum distance to consider "nearby"
            include_distance: Whether to include distance in results

        Returns:
            List of nearby bboxes (or tuples with distance if include_distance=True)
        """
        target_center = self.get_bbox_center(target_bbox)
        nearby_regions = []

        for bbox in candidate_bboxes:
            bbox_center = self.get_bbox_center(bbox)
            distance = self.calculate_distance(target_center, bbox_center)

            if distance <= max_distance:
                if include_distance:
                    nearby_regions.append((bbox, distance))
                else:
                    nearby_regions.append(bbox)

        # Sort by distance if including distance
        if include_distance:
            nearby_regions.sort(key=lambda x: x[1])

        return nearby_regions

    def match_regions_by_proximity(
        self,
        target_bboxes: List[Tuple[int, int, int, int]],
        candidate_bboxes: List[Tuple[int, int, int, int]],
        max_distance: float = 200.0,
    ) -> List[Dict[str, Any]]:
        """
        Match each target region to its closest candidate region.

        Args:
            target_bboxes: List of target bounding boxes
            candidate_bboxes: List of candidate bounding boxes
            max_distance: Maximum distance for matching

        Returns:
            List of match dictionaries with 'target', 'match', 'distance'
        """
        matches = []
        used_candidates = set()

        for target_bbox in target_bboxes:
            # Find available nearby candidates
            available_candidates = [
                bbox
                for i, bbox in enumerate(candidate_bboxes)
                if i not in used_candidates
            ]

            nearby = self.find_nearby_regions(
                target_bbox, available_candidates, max_distance, include_distance=True
            )

            if nearby:
                # Get closest match
                closest_bbox, distance = nearby[0]
                closest_idx = candidate_bboxes.index(closest_bbox)
                used_candidates.add(closest_idx)

                matches.append(
                    {'target': target_bbox, 'match': closest_bbox, 'distance': distance}
                )

        return matches
