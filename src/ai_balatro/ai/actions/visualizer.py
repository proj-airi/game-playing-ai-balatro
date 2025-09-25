"""YOLO detection result visualization tools."""

from typing import List
import numpy as np
from ...core.detection import Detection
from ...utils.logger import get_logger

logger = get_logger(__name__)


class DetectionVisualizer:
    """YOLO detection result visualization tool."""

    def __init__(self):
        """Initialize visualization tool."""
        pass

    def show_detection_results(
        self,
        image: np.ndarray,
        detections: List[Detection],
        window_title: str = 'YOLO Detection Results',
    ) -> None:
        """
        Display visualization window for YOLO detection results.

        Args:
            image: Original image
            detections: YOLO detection result list
            window_title: Window title
        """
        try:
            import cv2

            # Create visualization image copy
            vis_image = image.copy()

            # Define color mapping
            colors = {
                'card': (0, 255, 0),  # Green - Cards
                'button': (255, 0, 0),  # Red - Buttons
                'ui': (0, 255, 255),  # Yellow - UI elements
                'other': (128, 128, 128),  # Gray - Others
            }

            logger.info(f'Displaying {len(detections)} detection results')

            # Draw bounding boxes and labels for each detection result
            for i, detection in enumerate(detections):
                # Get color
                color = self._get_detection_color(detection, colors)

                # Draw bounding box
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # Prepare label text
                label = f'{detection.class_name} ({detection.confidence:.2f})'

                # Calculate text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw text background
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

                # Draw index number
                center_x, center_y = detection.center
                cv2.circle(vis_image, (center_x, center_y), 12, (255, 255, 255), -1)
                cv2.putText(
                    vis_image,
                    str(i + 1),
                    (center_x - 6, center_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )

            # Add statistics information
            stats_text = [
                f'Total detections: {len(detections)}',
                f'Cards: {self._count_detections_by_type(detections, "card")}',
                f'Buttons: {self._count_detections_by_type(detections, "button")}',
                f'Others: {len(detections) - self._count_detections_by_type(detections, "card") - self._count_detections_by_type(detections, "button")}',
                'Press ESC to close window',
            ]

            for i, text in enumerate(stats_text):
                cv2.putText(
                    vis_image,
                    text,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # Display window
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, 1200, 800)
            cv2.imshow(window_title, vis_image)

            logger.info(f'Displaying detection result window: {window_title}')
            logger.info('Press ESC key to close window...')

            # Wait for user input
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window_title)

            if key == 27:  # ESC key
                logger.info('User closed detection result window')

        except Exception as e:
            logger.error(f'Error occurred while displaying detection results: {e}')

    def _get_detection_color(self, detection: Detection, colors: dict) -> tuple:
        """Get color based on detection type."""
        class_name = detection.class_name.lower()

        if 'card' in class_name or 'poker' in class_name or 'joker' in class_name:
            return colors['card']
        elif 'button' in class_name:
            return colors['button']
        elif any(kw in class_name for kw in ['ui', 'menu', 'text', 'score']):
            return colors['ui']
        else:
            return colors['other']

    def _count_detections_by_type(
        self, detections: List[Detection], detection_type: str
    ) -> int:
        """Count the number of detections of a specified type."""
        count = 0
        for detection in detections:
            class_name = detection.class_name.lower()
            if detection_type == 'card':
                if (
                    'card' in class_name
                    or 'poker' in class_name
                    or 'joker' in class_name
                ):
                    count += 1
            elif detection_type == 'button':
                if 'button' in class_name:
                    count += 1
        return count
