"""Tests for card tooltip service with visual debugging."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.detection import Detection
from services.card_tooltip_service import CardTooltipService
from tests.test_utils import TestOutputManager


class TestCardTooltipService:
    """Test card tooltip service with visual outputs."""

    @pytest.fixture(scope="class")
    def test_output_dir(self):
        """Create standardized output directory for test outputs."""
        with TestOutputManager("card_tooltip_service", keep_outputs=True) as manager:
            yield manager.get_output_dir()

    @pytest.fixture
    def tooltip_service(self):
        """Create CardTooltipService instance."""
        return CardTooltipService()

    @pytest.fixture
    def mock_balatro_image(self, test_output_dir):
        """Create mock Balatro game image with cards and tooltips."""
        # Create 800x600 game-like image
        image = np.zeros((600, 800, 3), dtype=np.uint8)

        # Game background (dark blue)
        image[:, :] = [50, 30, 10]

        # Draw mock cards in hand area
        card_positions = [
            (100, 400, 180, 500),  # Card 1
            (200, 400, 280, 500),  # Card 2
            (300, 400, 380, 500),  # Card 3
            (500, 200, 580, 300),  # Joker card (different area)
        ]

        card_colors = [
            [0, 0, 200],    # Red-ish for poker cards
            [0, 0, 200],
            [0, 0, 200],
            [0, 150, 200]   # Orange for joker
        ]

        for (x1, y1, x2, y2), color in zip(card_positions, card_colors):
            image[y1:y2, x1:x2] = color
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw mock tooltips (appearing near some cards)
        tooltip_positions = [
            (120, 300, 250, 380),  # Tooltip for card 1
            (520, 100, 650, 180),  # Tooltip for joker
        ]

        for x1, y1, x2, y2 in tooltip_positions:
            # Tooltip background (light gray)
            image[y1:y2, x1:x2] = [200, 200, 200]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # Add mock text lines
            for i in range(3):
                line_y = y1 + 20 + i * 20
                cv2.line(image, (x1+10, line_y), (x2-10, line_y), (0, 0, 0), 2)

        # Save the mock image
        cv2.imwrite(str(test_output_dir / "mock_balatro_game.png"), image)

        return image

    @pytest.fixture
    def mock_detections(self):
        """Create mock YOLO detections for testing."""
        detections = [
            # Poker cards
            Detection(0, "poker_card_front", 0.95, (100, 400, 180, 500)),
            Detection(0, "poker_card_front", 0.92, (200, 400, 280, 500)),
            Detection(0, "poker_card_front", 0.89, (300, 400, 380, 500)),

            # Joker card
            Detection(1, "joker_card", 0.88, (500, 200, 580, 300)),

            # Card descriptions/tooltips
            Detection(2, "card_description", 0.85, (120, 300, 250, 380)),  # Near card 1
            Detection(3, "poker_card_description", 0.82, (520, 100, 650, 180)),  # Near joker
        ]
        return detections

    def test_separate_tooltips_and_cards(self, tooltip_service, mock_detections):
        """Test separation of tooltips and cards with counts."""
        separated = tooltip_service.separate_tooltips_and_cards(mock_detections)

        assert 'tooltips' in separated
        assert 'cards' in separated

        tooltips = separated['tooltips']
        cards = separated['cards']

        # Should have 2 tooltips and 4 cards
        assert len(tooltips) == 2
        assert len(cards) == 4

        # Check tooltip classes
        tooltip_classes = {t.class_name for t in tooltips}
        expected_tooltip_classes = {'card_description', 'poker_card_description'}
        assert tooltip_classes == expected_tooltip_classes

        # Check card classes
        card_classes = {c.class_name for c in cards}
        expected_card_classes = {'poker_card_front', 'joker_card'}
        assert card_classes == expected_card_classes

        print(f"✓ Separated: {len(tooltips)} tooltips, {len(cards)} cards")

    def test_match_tooltips_to_cards_with_stats(self, tooltip_service, mock_detections, test_output_dir):
        """Test tooltip-card matching with detailed statistics."""
        matches = tooltip_service.match_tooltips_to_cards(mock_detections, max_distance=200)

        # Should find matches for both tooltips
        assert len(matches) >= 1, "Should find at least one tooltip-card match"

        # Validate match structure
        for match in matches:
            required_keys = ['tooltip', 'card', 'distance', 'tooltip_bbox', 'card_bbox']
            for key in required_keys:
                assert key in match, f"Match should contain {key}"

            assert isinstance(match['distance'], float)
            assert match['distance'] >= 0

        # Get and save statistics
        stats = tooltip_service.get_processing_stats(matches)

        stats_file = test_output_dir / "tooltip_matching_stats.txt"
        with open(stats_file, 'w') as f:
            f.write("Tooltip-Card Matching Statistics\n")
            f.write("================================\n")
            f.write(f"Total matches: {stats['total_matches']}\n")
            f.write(f"Average distance: {stats['average_distance']:.2f}px\n")
            f.write(f"Min distance: {stats['min_distance']:.2f}px\n")
            f.write(f"Max distance: {stats['max_distance']:.2f}px\n")
            f.write(f"Tooltip types: {stats['tooltip_types']}\n")
            f.write(f"Card types: {stats['card_types']}\n")

        print(f"✓ Matching stats saved: {stats}")

        # Don't return matches to avoid pytest warning

    def test_create_tooltip_card_crops_with_saving(
        self, tooltip_service, mock_balatro_image, mock_detections, test_output_dir
    ):
        """Test crop creation with detailed image saving."""
        matches = tooltip_service.match_tooltips_to_cards(mock_detections)
        enhanced_matches = tooltip_service.create_tooltip_card_crops(mock_balatro_image, matches)

        assert len(enhanced_matches) >= 1, "Should create crops for matches"

        # Save each crop with detailed info
        for i, match in enumerate(enhanced_matches):
            match_dir = test_output_dir / f"match_{i:02d}"
            match_dir.mkdir(exist_ok=True)

            # Save individual crops
            cv2.imwrite(str(match_dir / "tooltip_crop.png"), match['tooltip_crop'])
            cv2.imwrite(str(match_dir / "card_crop.png"), match['card_crop'])
            cv2.imwrite(str(match_dir / "context_crop.png"), match['context_crop'])

            # Create detailed info file
            info_file = match_dir / "match_info.txt"
            with open(info_file, 'w') as f:
                f.write(f"Match {i} Details\n")
                f.write(f"================\n")
                f.write(f"Tooltip class: {match['tooltip'].class_name}\n")
                f.write(f"Card class: {match['card'].class_name}\n")
                f.write(f"Distance: {match['distance']:.2f}px\n")
                f.write(f"Tooltip bbox: {match['tooltip_bbox']}\n")
                f.write(f"Card bbox: {match['card_bbox']}\n")
                f.write(f"Tooltip crop shape: {match['tooltip_crop_shape']}\n")
                f.write(f"Card crop shape: {match['card_crop_shape']}\n")
                f.write(f"Context crop shape: {match['context_crop_shape']}\n")

            # Validate crop properties
            assert match['tooltip_crop'].shape[2] == 3, "Tooltip crop should be color image"
            assert match['card_crop'].shape[2] == 3, "Card crop should be color image"
            assert match['context_crop'].shape[2] == 3, "Context crop should be color image"

            # Validate crop sizes are reasonable
            assert match['tooltip_crop'].size > 0, "Tooltip crop should not be empty"
            assert match['card_crop'].size > 0, "Card crop should not be empty"
            assert match['context_crop'].size > 0, "Context crop should not be empty"

            print(f"✓ Match {i} crops saved to {match_dir}")

    def test_create_visualization_with_annotations(
        self, tooltip_service, mock_balatro_image, mock_detections, test_output_dir
    ):
        """Create comprehensive visualization of the matching process."""
        matches = tooltip_service.match_tooltips_to_cards(mock_detections)

        # Create visualization image
        vis_image = mock_balatro_image.copy()

        # Colors for different types
        tooltip_color = (0, 255, 255)  # Yellow
        card_color = (255, 0, 0)       # Blue
        match_line_color = (0, 255, 0) # Green

        # Draw all detections first
        separated = tooltip_service.separate_tooltips_and_cards(mock_detections)

        # Draw cards
        for card in separated['cards']:
            x1, y1, x2, y2 = card.bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), card_color, 2)
            cv2.putText(vis_image, f"CARD: {card.class_name}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, card_color, 1)

        # Draw tooltips
        for tooltip in separated['tooltips']:
            x1, y1, x2, y2 = tooltip.bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), tooltip_color, 2)
            cv2.putText(vis_image, f"TOOLTIP: {tooltip.class_name}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, tooltip_color, 1)

        # Draw matches
        for i, match in enumerate(matches):
            # Get centers
            tooltip_center = tooltip_service.region_matcher.get_bbox_center(match['tooltip_bbox'])
            card_center = tooltip_service.region_matcher.get_bbox_center(match['card_bbox'])

            # Draw connecting line
            cv2.line(vis_image, tooltip_center, card_center, match_line_color, 2)

            # Add match info
            mid_x = (tooltip_center[0] + card_center[0]) // 2
            mid_y = (tooltip_center[1] + card_center[1]) // 2
            cv2.putText(vis_image, f"Match {i}: {match['distance']:.1f}px",
                       (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, match_line_color, 1)

        # Save comprehensive visualization
        cv2.imwrite(str(test_output_dir / "complete_matching_visualization.png"), vis_image)

        # Add legend
        legend_height = 100
        legend_image = np.zeros((legend_height, vis_image.shape[1], 3), dtype=np.uint8)

        legend_items = [
            ("Cards", card_color),
            ("Tooltips", tooltip_color),
            ("Matches", match_line_color)
        ]

        for i, (label, color) in enumerate(legend_items):
            y_pos = 20 + i * 25
            cv2.rectangle(legend_image, (10, y_pos-10), (30, y_pos+10), color, -1)
            cv2.putText(legend_image, label, (40, y_pos+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Combine image with legend
        final_image = np.vstack([vis_image, legend_image])
        cv2.imwrite(str(test_output_dir / "final_visualization_with_legend.png"), final_image)

        print(f"✓ Comprehensive visualization saved")

    def test_pixel_level_validation(self, tooltip_service, mock_balatro_image, mock_detections, test_output_dir):
        """Test pixel-level properties of cropped regions."""
        matches = tooltip_service.match_tooltips_to_cards(mock_detections)
        enhanced_matches = tooltip_service.create_tooltip_card_crops(mock_balatro_image, matches)

        pixel_stats = []

        for i, match in enumerate(enhanced_matches):
            tooltip_crop = match['tooltip_crop']
            card_crop = match['card_crop']

            # Calculate pixel statistics
            tooltip_stats = {
                'mean_brightness': np.mean(tooltip_crop),
                'std_brightness': np.std(tooltip_crop),
                'dominant_color': np.mean(tooltip_crop, axis=(0, 1)).astype(int),
                'pixel_count': tooltip_crop.size
            }

            card_stats = {
                'mean_brightness': np.mean(card_crop),
                'std_brightness': np.std(card_crop),
                'dominant_color': np.mean(card_crop, axis=(0, 1)).astype(int),
                'pixel_count': card_crop.size
            }

            match_pixel_stats = {
                'match_id': i,
                'tooltip': tooltip_stats,
                'card': card_stats
            }

            pixel_stats.append(match_pixel_stats)

            # Validate basic pixel properties
            assert tooltip_stats['mean_brightness'] > 0, "Tooltip should have some brightness"
            assert card_stats['mean_brightness'] > 0, "Card should have some brightness"
            assert tooltip_stats['pixel_count'] > 100, "Tooltip should have reasonable pixel count"
            assert card_stats['pixel_count'] > 100, "Card should have reasonable pixel count"

        # Save pixel statistics
        pixel_stats_file = test_output_dir / "pixel_statistics.txt"
        with open(pixel_stats_file, 'w') as f:
            f.write("Pixel-Level Statistics\n")
            f.write("=====================\n\n")

            for stats in pixel_stats:
                f.write(f"Match {stats['match_id']}:\n")
                f.write(f"  Tooltip - Brightness: {stats['tooltip']['mean_brightness']:.1f} ± {stats['tooltip']['std_brightness']:.1f}\n")
                f.write(f"  Tooltip - Dominant Color (BGR): {stats['tooltip']['dominant_color']}\n")
                f.write(f"  Tooltip - Pixel Count: {stats['tooltip']['pixel_count']}\n")
                f.write(f"  Card - Brightness: {stats['card']['mean_brightness']:.1f} ± {stats['card']['std_brightness']:.1f}\n")
                f.write(f"  Card - Dominant Color (BGR): {stats['card']['dominant_color']}\n")
                f.write(f"  Card - Pixel Count: {stats['card']['pixel_count']}\n\n")

        print(f"✓ Pixel statistics saved for {len(pixel_stats)} matches")


if __name__ == "__main__":
    print("Running card tooltip service tests with visual debugging...")
    pytest.main([__file__, "-v", "-s"])