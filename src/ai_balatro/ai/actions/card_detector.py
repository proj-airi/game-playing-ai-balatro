"""Card position detection and sorting functionality."""

from typing import List
from ...core.detection import Detection
from ...utils.logger import get_logger

logger = get_logger(__name__)


class CardPositionDetector:
    """Detects and sorts cards by position from left to right."""

    def __init__(self):
        """Initialize card position detector."""
        # Priority: playable card types
        self.playable_card_classes = [
            'poker_card_front',  # Poker card front (highest priority)
            'joker_card',  # Joker card
            'tarot_card',  # Tarot card
            'planet_card',  # Planet card
            'spectral_card',  # Spectral card
        ]

    def get_hand_cards(self, detections: List[Detection]) -> List[Detection]:
        """
        Extract hand cards from detection results and sort them from left to right.

        Args:
            detections: YOLO detection results

        Returns:
            Sorted list of hand card Detection objects
        """
        # Filter playable cards
        hand_cards = []

        for detection in detections:
            if self._is_playable_card(detection):
                hand_cards.append(detection)

        if not hand_cards:
            logger.warning('No playable hand cards detected')
            return []

        # Sort by x-coordinate (left to right)
        hand_cards.sort(key=lambda card: card.bbox[0])  # x1 coordinate

        logger.info(f'Detected {len(hand_cards)} hand cards:')
        for i, card in enumerate(hand_cards):
            logger.info(
                f'  Position {i}: {card.class_name} at {card.center} (confidence: {card.confidence:.3f})'
            )

        return hand_cards

    def _is_playable_card(self, detection: Detection) -> bool:
        """Check if the detection is a playable card."""
        class_name = detection.class_name.lower()

        # Check if it's a playable card type
        for card_class in self.playable_card_classes:
            if card_class in class_name:
                # Exclude descriptions and card backs
                if 'description' not in class_name and 'back' not in class_name:
                    return True

        return False
