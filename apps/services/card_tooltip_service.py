"""Balatro-specific service for matching card tooltips to cards."""

from typing import List, Dict, Any, Optional
import numpy as np

from core.detection import Detection
from utils.image_cropper import ImageCropper, RegionMatcher
from utils.logger import get_logger

logger = get_logger(__name__)


class CardTooltipService:
    """Service for matching Balatro card description tooltips to their cards."""

    def __init__(self, image_cropper: Optional[ImageCropper] = None):
        """
        Initialize service.

        Args:
            image_cropper: Image cropper utility (creates default if None)
        """
        self.image_cropper = image_cropper or ImageCropper(padding_pixels=15)
        self.region_matcher = RegionMatcher()

        # Balatro-specific class mappings
        self.tooltip_classes = {'card_description', 'poker_card_description'}
        self.card_classes = {
            'poker_card_front', 'joker_card', 'planet_card',
            'tarot_card', 'spectral_card', 'poker_card_stack'
        }

    def separate_tooltips_and_cards(
        self,
        detections: List[Detection]
    ) -> Dict[str, List[Detection]]:
        """
        Separate detections into tooltips and cards.

        Args:
            detections: All detections from YOLO

        Returns:
            Dictionary with 'tooltips' and 'cards' lists
        """
        tooltips = []
        cards = []

        for detection in detections:
            if detection.class_name in self.tooltip_classes:
                tooltips.append(detection)
            elif detection.class_name in self.card_classes:
                cards.append(detection)

        return {
            'tooltips': tooltips,
            'cards': cards
        }

    def match_tooltips_to_cards(
        self,
        detections: List[Detection],
        max_distance: float = 300.0
    ) -> List[Dict[str, Any]]:
        """
        Match card tooltips to their corresponding cards.

        Args:
            detections: All detections from YOLO
            max_distance: Maximum distance for tooltip-card matching

        Returns:
            List of match dictionaries
        """
        separated = self.separate_tooltips_and_cards(detections)
        tooltips = separated['tooltips']
        cards = separated['cards']

        logger.info(f"Matching {len(tooltips)} tooltips to {len(cards)} cards")

        matches = []

        for tooltip in tooltips:
            # Find nearby cards for this tooltip
            nearby_cards = self._find_nearby_cards(tooltip, cards, max_distance)

            if nearby_cards:
                closest_card, distance = nearby_cards[0]  # Already sorted by distance

                match = {
                    'tooltip': tooltip,
                    'card': closest_card,
                    'distance': distance,
                    'tooltip_bbox': tooltip.bbox,
                    'card_bbox': closest_card.bbox
                }

                matches.append(match)
                logger.debug(
                    f"Matched {tooltip.class_name} to {closest_card.class_name} "
                    f"(distance: {distance:.1f}px)"
                )

            else:
                logger.warning(f"No nearby cards found for {tooltip.class_name}")

        return matches

    def create_tooltip_card_crops(
        self,
        image: np.ndarray,
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create image crops for tooltip-card pairs.

        Args:
            image: Source image
            matches: List of tooltip-card matches

        Returns:
            List of match dictionaries with added crop data
        """
        enhanced_matches = []

        for match in matches:
            try:
                # Crop individual regions
                tooltip_crop = self.image_cropper.crop_detection(image, match['tooltip'])
                card_crop = self.image_cropper.crop_detection(image, match['card'])

                # Create combined context crop
                combined_crop = self.image_cropper.create_combined_crop(
                    image, [match['tooltip_bbox'], match['card_bbox']], padding=20
                )

                enhanced_match = match.copy()
                enhanced_match.update({
                    'tooltip_crop': tooltip_crop,
                    'card_crop': card_crop,
                    'context_crop': combined_crop,
                    'tooltip_crop_shape': tooltip_crop.shape,
                    'card_crop_shape': card_crop.shape,
                    'context_crop_shape': combined_crop.shape
                })

                enhanced_matches.append(enhanced_match)

            except Exception as e:
                logger.error(f"Failed to create crops for match: {e}")
                continue

        logger.info(f"Created crops for {len(enhanced_matches)} matches")
        return enhanced_matches

    def _find_nearby_cards(
        self,
        tooltip: Detection,
        cards: List[Detection],
        max_distance: float
    ) -> List[tuple]:
        """
        Find cards near a tooltip detection.

        Args:
            tooltip: Tooltip detection
            cards: List of card detections
            max_distance: Maximum distance to consider

        Returns:
            List of (card_detection, distance) tuples, sorted by distance
        """
        card_bboxes = [card.bbox for card in cards]

        nearby_with_distance = self.region_matcher.find_nearby_regions(
            tooltip.bbox, card_bboxes, max_distance, include_distance=True
        )

        # Convert back to Detection objects with distance
        nearby_cards = []
        for bbox, distance in nearby_with_distance:
            # Find the original Detection object
            for card in cards:
                if card.bbox == bbox:
                    nearby_cards.append((card, distance))
                    break

        return nearby_cards

    def get_processing_stats(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the tooltip-card matching process.

        Args:
            matches: List of matches from match_tooltips_to_cards

        Returns:
            Dictionary with statistics
        """
        if not matches:
            return {
                'total_matches': 0,
                'average_distance': 0,
                'min_distance': 0,
                'max_distance': 0,
                'tooltip_types': {},
                'card_types': {}
            }

        distances = [match['distance'] for match in matches]
        tooltip_types = {}
        card_types = {}

        for match in matches:
            tooltip_class = match['tooltip'].class_name
            card_class = match['card'].class_name

            tooltip_types[tooltip_class] = tooltip_types.get(tooltip_class, 0) + 1
            card_types[card_class] = card_types.get(card_class, 0) + 1

        return {
            'total_matches': len(matches),
            'average_distance': np.mean(distances),
            'min_distance': min(distances),
            'max_distance': max(distances),
            'tooltip_types': tooltip_types,
            'card_types': card_types
        }