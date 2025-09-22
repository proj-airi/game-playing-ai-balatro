"""Tests for image cropping utilities with debugging output."""

import pytest
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_cropper import ImageCropper, RegionMatcher


class TestImageCropper:
    """Test cases for ImageCropper with visual debugging."""

    @pytest.fixture(scope="class")
    def test_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp(prefix="test_cropper_")
        yield Path(temp_dir)
        # Cleanup after all tests
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_image(self, test_output_dir):
        """Create a test image with known patterns."""
        # Create 300x300 test image with distinct colored regions
        image = np.zeros((300, 300, 3), dtype=np.uint8)

        # Background (blue)
        image[:, :] = [100, 50, 0]  # Blue background

        # Red square (top-left)
        image[50:150, 50:150] = [0, 0, 255]  # Red in BGR

        # Green square (bottom-right)
        image[200:280, 200:280] = [0, 255, 0]  # Green in BGR

        # Yellow rectangle (middle)
        image[120:180, 100:200] = [0, 255, 255]  # Yellow in BGR

        # Save test image for reference
        cv2.imwrite(str(test_output_dir / "test_input.png"), image)
        return image

    @pytest.fixture
    def cropper(self):
        """Create ImageCropper instance."""
        return ImageCropper(padding_pixels=10)

    def test_crop_detection_bbox_basic(self, cropper, test_image, test_output_dir):
        """Test basic bbox cropping with pixel validation."""
        # Crop the red square region
        bbox = (50, 50, 150, 150)  # Red square
        cropped = cropper.crop_detection_bbox(test_image, bbox, padding=5)

        # Save cropped image
        cv2.imwrite(str(test_output_dir / "crop_red_square.png"), cropped)

        # Validate dimensions
        expected_size = (110, 110, 3)  # 100x100 + 5px padding each side
        assert cropped.shape == expected_size

        # Pixel-level validation: center should be red
        center_y, center_x = cropped.shape[0] // 2, cropped.shape[1] // 2
        center_pixel = cropped[center_y, center_x]

        # In BGR format, red is [0, 0, 255]
        assert center_pixel[2] > 200  # High red channel
        assert center_pixel[1] < 50   # Low green channel
        assert center_pixel[0] < 50   # Low blue channel

        print(f"✓ Red square crop saved: {cropped.shape}, center pixel: {center_pixel}")

    def test_crop_multiple_regions_with_validation(self, cropper, test_image, test_output_dir):
        """Test cropping multiple regions with pixel validation."""
        bboxes = [
            (50, 50, 150, 150),    # Red square
            (200, 200, 280, 280),  # Green square
            (100, 120, 200, 180)   # Yellow rectangle
        ]

        colors = ["red", "green", "yellow"]
        expected_bgr = [
            [0, 0, 255],    # Red
            [0, 255, 0],    # Green
            [0, 255, 255]   # Yellow
        ]

        crops = cropper.crop_multiple_regions(test_image, bboxes, padding=5)

        assert len(crops) == 3
        assert all(crop is not None for crop in crops)

        for i, (crop, color, expected_pixel) in enumerate(zip(crops, colors, expected_bgr)):
            # Save each crop
            filename = f"crop_{color}_{i}.png"
            cv2.imwrite(str(test_output_dir / filename), crop)

            # Validate center pixel color
            center_y, center_x = crop.shape[0] // 2, crop.shape[1] // 2
            center_pixel = crop[center_y, center_x]

            # Check dominant color channel
            dominant_channel = np.argmax(expected_pixel)
            assert center_pixel[dominant_channel] > 200, f"{color} crop should have high {dominant_channel} channel"

            print(f"✓ {color} crop saved: {crop.shape}, center pixel: {center_pixel}")

    def test_create_combined_crop_with_stats(self, cropper, test_image, test_output_dir):
        """Test combined crop with statistical validation."""
        bboxes = [
            (50, 50, 150, 150),   # Red square
            (200, 200, 280, 280)  # Green square
        ]

        combined = cropper.create_combined_crop(test_image, bboxes, padding=10)

        # Save combined crop
        cv2.imwrite(str(test_output_dir / "crop_combined.png"), combined)

        # Should encompass both regions plus padding
        min_expected_height = (280 - 50) + 20  # Height span + padding
        min_expected_width = (280 - 50) + 20   # Width span + padding

        assert combined.shape[0] >= min_expected_height
        assert combined.shape[1] >= min_expected_width

        # Calculate color statistics
        red_pixels = np.sum((combined[:, :, 2] > 200) & (combined[:, :, 1] < 100))
        green_pixels = np.sum((combined[:, :, 1] > 200) & (combined[:, :, 2] < 100))

        print(f"✓ Combined crop: {combined.shape}, red pixels: {red_pixels}, green pixels: {green_pixels}")
        assert red_pixels > 1000, "Should contain substantial red region"
        assert green_pixels > 1000, "Should contain substantial green region"

    def test_edge_case_cropping(self, cropper, test_image, test_output_dir):
        """Test cropping near image edges."""
        # Test cases for edge handling
        edge_cases = [
            ("top_left", (0, 0, 50, 50)),
            ("top_right", (250, 0, 300, 50)),
            ("bottom_left", (0, 250, 50, 300)),
            ("bottom_right", (250, 250, 300, 300))
        ]

        for case_name, bbox in edge_cases:
            cropped = cropper.crop_detection_bbox(test_image, bbox, padding=20)

            # Save edge case crop
            cv2.imwrite(str(test_output_dir / f"crop_edge_{case_name}.png"), cropped)

            # Validate that crop is not empty and within reasonable bounds
            assert cropped.shape[0] > 0, f"{case_name} crop should not be empty (height)"
            assert cropped.shape[1] > 0, f"{case_name} crop should not be empty (width)"
            assert cropped.shape[0] <= 90, f"{case_name} crop should not exceed expected max height"
            assert cropped.shape[1] <= 90, f"{case_name} crop should not exceed expected max width"

            print(f"✓ Edge case {case_name}: {cropped.shape}")


class TestRegionMatcher:
    """Test RegionMatcher with visual debugging."""

    @pytest.fixture(scope="class")
    def test_output_dir_regions(self):
        """Create temporary directory for region matcher tests."""
        temp_dir = tempfile.mkdtemp(prefix="test_regions_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def matcher(self):
        """Create RegionMatcher instance."""
        return RegionMatcher()

    @pytest.fixture
    def test_regions_image(self, test_output_dir_regions):
        """Create test image with labeled regions for matching tests."""
        image = np.zeros((400, 400, 3), dtype=np.uint8)

        # Create distinct regions with labels
        regions = [
            ((50, 50, 100, 100), [255, 0, 0], "target_1"),      # Red
            ((150, 50, 200, 100), [0, 255, 0], "candidate_1"),  # Green
            ((300, 50, 350, 100), [0, 0, 255], "candidate_2"),  # Blue
            ((50, 200, 100, 250), [255, 255, 0], "target_2"),   # Yellow
            ((150, 200, 200, 250), [255, 0, 255], "candidate_3") # Magenta
        ]

        for (x1, y1, x2, y2), color, label in regions:
            image[y1:y2, x1:x2] = color
            # Add text label
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(test_output_dir_regions / "regions_test_input.png"), image)
        return image, regions

    def test_find_nearby_regions_with_visualization(self, matcher, test_regions_image, test_output_dir_regions):
        """Test region matching with visual output."""
        image, regions = test_regions_image

        # Extract target and candidate bboxes
        target_bbox = regions[0][0]  # Red square
        candidate_bboxes = [region[0] for region in regions[1:4]]  # Green, Blue, Yellow

        nearby = matcher.find_nearby_regions(
            target_bbox, candidate_bboxes, max_distance=150, include_distance=True
        )

        # Create visualization
        vis_image = image.copy()

        # Draw target in thick red
        x1, y1, x2, y2 = target_bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(vis_image, "TARGET", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw nearby regions with distance labels
        for i, (bbox, distance) in enumerate(nearby):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"d:{distance:.1f}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(str(test_output_dir_regions / "nearby_regions_visualization.png"), vis_image)

        # Validate results
        assert len(nearby) >= 1, "Should find at least one nearby region"

        # Check that distances are sorted
        distances = [item[1] for item in nearby]
        assert distances == sorted(distances), "Results should be sorted by distance"

        print(f"✓ Found {len(nearby)} nearby regions with distances: {distances}")

    def test_match_regions_with_detailed_output(self, matcher, test_regions_image, test_output_dir_regions):
        """Test region matching with detailed statistics."""
        image, regions = test_regions_image

        targets = [regions[0][0], regions[3][0]]  # Red and Yellow
        candidates = [region[0] for region in regions[1:4]]  # Green, Blue, Magenta

        matches = matcher.match_regions_by_proximity(targets, candidates, max_distance=200)

        # Create detailed visualization
        vis_image = image.copy()

        colors = [(0, 0, 255), (0, 255, 255)]  # Red, Yellow for targets

        for i, match in enumerate(matches):
            target_bbox = match['target']
            match_bbox = match['match']
            distance = match['distance']

            # Draw target
            x1, y1, x2, y2 = target_bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[i], 3)

            # Draw match
            x1, y1, x2, y2 = match_bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[i], 2)

            # Draw connecting line
            target_center = matcher.get_bbox_center(target_bbox)
            match_center = matcher.get_bbox_center(match_bbox)
            cv2.line(vis_image, target_center, match_center, colors[i], 2)

            # Add distance label
            mid_x = (target_center[0] + match_center[0]) // 2
            mid_y = (target_center[1] + match_center[1]) // 2
            cv2.putText(vis_image, f"{distance:.1f}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        cv2.imwrite(str(test_output_dir_regions / "region_matches_visualization.png"), vis_image)

        # Generate statistics
        stats = {
            'total_matches': len(matches),
            'average_distance': np.mean([m['distance'] for m in matches]) if matches else 0,
            'distances': [m['distance'] for m in matches]
        }

        print(f"✓ Matching stats: {stats}")

        # Save stats to file
        stats_file = test_output_dir_regions / "matching_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Region Matching Statistics\n")
            f.write(f"========================\n")
            f.write(f"Total matches: {stats['total_matches']}\n")
            f.write(f"Average distance: {stats['average_distance']:.2f}\n")
            f.write(f"Individual distances: {stats['distances']}\n")

        assert len(matches) >= 1, "Should find at least one match"


if __name__ == "__main__":
    # Run tests with output directory info
    print("Running image cropper tests with visual debugging...")
    print("Test outputs will be saved to temporary directory")
    pytest.main([__file__, "-v", "-s"])