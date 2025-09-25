"""Main card action engine for executing card-based actions in Balatro."""

import time
from typing import List, Optional
import numpy as np
from ...core.detection import Detection
from ...core.yolo_detector import YOLODetector
from ...core.multi_yolo_detector import MultiYOLODetector
from ...core.screen_capture import ScreenCapture
from ...utils.logger import get_logger
from .schemas import CardAction
from .detection_models import ButtonDetection
from .card_detector import CardPositionDetector
from .button_detector import ButtonDetector
from .visualizer import DetectionVisualizer
from .mouse_controller import MouseController

logger = get_logger(__name__)


class CardActionEngine:
    """Engine for executing card actions based on position arrays."""

    def __init__(
        self,
        yolo_detector: Optional[YOLODetector] = None,
        screen_capture: Optional[ScreenCapture] = None,
        multi_detector: Optional[MultiYOLODetector] = None,
    ):
        """
        Initialize card action engine.

        Args:
            yolo_detector: Traditional single YOLO detector (backward compatibility)
            screen_capture: Screen capture instance
            multi_detector: Multi-model YOLO detector (recommended)
        """
        # Validate required parameters
        if screen_capture is None:
            raise ValueError('screen_capture is required')

        self.screen_capture = screen_capture

        # Support backward compatible single detector or new multi-model detector
        if multi_detector is not None:
            self.multi_detector = multi_detector
            self.yolo_detector = None  # Not needed when using multi-model
            logger.info('Using multi-model YOLO detector')
        elif yolo_detector is not None:
            self.yolo_detector = yolo_detector
            self.multi_detector = None
            logger.info('Using traditional single YOLO detector (backward compatibility mode)')
        else:
            # Default to creating multi-model detector
            self.multi_detector = MultiYOLODetector()
            self.yolo_detector = None
            logger.info('Created default multi-model YOLO detector')

        # Initialize sub-components
        self.position_detector = CardPositionDetector()
        self.button_detector = ButtonDetector(self.multi_detector)
        self.visualizer = DetectionVisualizer()
        self.mouse_controller = MouseController(self.screen_capture)

        logger.info('CardActionEngine initialization complete')

    def set_mouse_animation_params(
        self,
        move_duration: float = 0.5,
        move_steps: int = 20,
        click_hold_duration: float = 0.1,
    ) -> None:
        """
        Set mouse movement animation parameters.

        Args:
            move_duration: Mouse movement duration (seconds)
            move_steps: Movement steps (more steps = smoother)
            click_hold_duration: Click hold time (seconds)
        """
        self.mouse_controller.set_animation_params(
            move_duration, move_steps, click_hold_duration
        )

    def execute_card_action(
        self,
        positions: List[int],
        description: str = '',
        show_visualization: bool = False,
    ) -> bool:
        """
        Execute card action.

        Args:
            positions: Position array, like [1, 1, 1, 0] or [-1, -1, 0, 0]
            description: Action description
            show_visualization: Whether to show visualization window

        Returns:
            Whether execution was successful
        """
        logger.info(f'Executing card action: {positions} - {description}')

        # Create CardAction object
        action = CardAction.from_array(positions, description)

        try:
            # 1. Capture current screen
            frame = self.screen_capture.capture_once()
            if frame is None:
                logger.error('Screen capture failed')
                return False

            # 2. Detect hand cards
            if self.multi_detector is not None:
                # Use multi-model detector's entity model to detect cards
                detections = self.multi_detector.detect_entities(frame)
                logger.info(f'Multi-model detector (entity model) detected {len(detections)} objects')
            elif self.yolo_detector is not None:
                # Backward compatibility: use single detector
                detections = self.yolo_detector.detect(frame)
                logger.info(f'Single detector detected {len(detections)} objects')
            else:
                logger.error('No detector available')
                return False

            hand_cards = self.position_detector.get_hand_cards(detections)

            if not hand_cards:
                logger.error('No hand cards detected')
                return False

            # 3. Validate position array length
            if len(positions) > len(hand_cards):
                logger.warning(
                    f'Position array length ({len(positions)}) exceeds hand card count ({len(hand_cards)})'
                )
                # Truncate position array
                positions = positions[: len(hand_cards)]

            # 4. Show visualization (if enabled)
            if show_visualization:
                self._show_card_action_visualization(
                    frame, hand_cards, action, positions
                )

            # 5. Ensure game window focus
            logger.info('Ensuring game window has focus...')
            focus_success = self.mouse_controller.ensure_game_window_focus()
            if focus_success:
                logger.info('✓ Game window focus ready')
            else:
                logger.warning('! Window focus handling failed, continuing with operation')

            # 6. Execute click operations
            success = self._execute_clicks(hand_cards, action)

            if success:
                # 7. Execute confirmation action (play or discard)
                time.sleep(0.5)  # Action delay
                success = self._execute_confirm_action(action, show_visualization)

            return success

        except Exception as e:
            logger.error(f'Error occurred while executing card action: {e}')
            return False

    def hover_card(self, card_index: int, duration: float = 1.0) -> bool:
        """
        Hover over card at specified position.

        Args:
            card_index: Card position index (starting from 0)
            duration: Hover duration

        Returns:
            Whether execution was successful
        """
        try:
            logger.info(f'Hovering over card at position {card_index} for {duration} seconds')

            # Capture screen and detect hand cards
            frame = self.screen_capture.capture_once()
            if frame is None:
                return False

            # Detect hand cards
            if self.multi_detector is not None:
                detections = self.multi_detector.detect_entities(frame)
            elif self.yolo_detector is not None:
                detections = self.yolo_detector.detect(frame)
            else:
                logger.error('No detector available')
                return False

            hand_cards = self.position_detector.get_hand_cards(detections)

            if card_index >= len(hand_cards):
                logger.error(f'Position {card_index} exceeds hand card range (total {len(hand_cards)} cards)')
                return False

            card = hand_cards[card_index]

            # Calculate screen coordinates (reuse logic from _click_card)
            center_x, center_y = card.center
            capture_region = self.screen_capture.get_capture_region()
            if not capture_region:
                return False

            current_frame = self.screen_capture.capture_once()
            if current_frame is None:
                return False

            actual_height, actual_width = current_frame.shape[:2]
            scale_x = capture_region['width'] / actual_width
            scale_y = capture_region['height'] / actual_height

            screen_x = capture_region['left'] + (center_x * scale_x)
            screen_y = capture_region['top'] + (center_y * scale_y)

            # Smoothly move mouse to card
            if not self.mouse_controller.smooth_move_to(int(screen_x), int(screen_y)):
                logger.error('Smooth mouse movement failed')
                return False

            logger.info(f'Mouse hovering at ({screen_x:.1f}, {screen_y:.1f})')

            # Hover for specified time
            time.sleep(duration)

            return True

        except Exception as e:
            logger.error(f'Hover operation failed: {e}')
            return False

    def _execute_clicks(self, hand_cards: List[Detection], action: CardAction) -> bool:
        """Execute click operations to select cards."""
        try:
            clicked_cards = []

            # Select cards to click based on action type
            if action.is_play_action:
                # Play action: click cards with position = 1
                target_indices = action.selected_indices
                logger.info(f'Selecting cards to play: positions {target_indices}')
            elif action.is_discard_action:
                # Discard action: click cards with position = -1
                target_indices = action.discard_indices
                logger.info(f'Selecting cards to discard: positions {target_indices}')
            else:
                logger.warning('Unrecognized action type')
                return False

            # Click target cards
            for index in target_indices:
                if index < len(hand_cards):
                    card = hand_cards[index]
                    success = self._click_card(card, index)
                    if success:
                        clicked_cards.append(card)
                        time.sleep(self.mouse_controller.click_interval)
                    else:
                        logger.error(f'Failed to click card at position {index}')
                        return False
                else:
                    logger.error(f'Position {index} exceeds hand card range')
                    return False

            logger.info(f'Successfully selected {len(clicked_cards)} cards')
            return True

        except Exception as e:
            logger.error(f'Error occurred while executing click operations: {e}')
            return False

    def _click_card(self, card: Detection, index: int) -> bool:
        """Click specified card."""
        try:
            # Get card center point
            center_x, center_y = card.center

            # Get capture region information for coordinate conversion
            capture_region = self.screen_capture.get_capture_region()
            if not capture_region:
                logger.error('Cannot get capture region information')
                return False

            # Get current frame for coordinate scaling calculation
            current_frame = self.screen_capture.capture_once()
            if current_frame is None:
                logger.error('Cannot get current frame')
                return False

            # Calculate coordinate conversion
            actual_height, actual_width = current_frame.shape[:2]
            region_width = capture_region['width']
            region_height = capture_region['height']

            scale_x = region_width / actual_width
            scale_y = region_height / actual_height

            # Convert to screen coordinates
            screen_x = capture_region['left'] + (center_x * scale_x)
            screen_y = capture_region['top'] + (center_y * scale_y)

            logger.info(f'Clicking card at position {index}: {card.class_name}')
            logger.info(f'  Detection center: ({center_x}, {center_y})')
            logger.info(f'  Screen coordinates: ({screen_x:.1f}, {screen_y:.1f})')

            # Use mouse controller to click
            return self.mouse_controller.click_at(int(screen_x), int(screen_y))

        except Exception as e:
            logger.error(f'Error occurred while clicking card: {e}')
            return False

    def _execute_confirm_action(
        self, action: CardAction, show_visualization: bool = False
    ) -> bool:
        """Execute confirmation action (play or discard button)."""
        try:
            if action.is_play_action:
                button_type = 'play'
                logger.info('Finding and clicking play button')
            elif action.is_discard_action:
                button_type = 'discard'
                logger.info('Finding and clicking discard button')
            else:
                logger.info('No confirmation action required')
                return True

            # Capture current screen to find button
            frame = self.screen_capture.capture_once()
            if frame is None:
                logger.error('Cannot capture screen to find confirmation button')
                return False

            # Use UI model to detect button
            logger.info(f'Using UI model to detect {button_type} button...')

            # Directly use button_detector to detect button
            target_button = self.button_detector.find_best_button(frame, button_type)

            if target_button is None:
                logger.warning(f'{button_type} button not found, trying to find all buttons...')
                all_buttons = self.button_detector.find_buttons(frame)

                if all_buttons:
                    logger.info(
                        f'Found {len(all_buttons)} available buttons, but no {button_type} button'
                    )

                    # Show visualization of all buttons (if enabled)
                    if show_visualization:
                        logger.info('Displaying all detected buttons...')
                        # Create detection list containing only buttons for visualization
                        button_detections: List[Detection] = [
                            btn for btn in all_buttons
                        ]
                        self.visualizer.show_detection_results(
                            frame,
                            button_detections,
                            f'All Detected Buttons (Target: {button_type})',
                        )

                    # Could consider using other suitable buttons as alternatives
                    logger.info('Please manually click the appropriate button to complete the operation')
                else:
                    logger.error('No buttons detected')

                    # Show UI detection results for debugging
                    if show_visualization and self.multi_detector is not None:
                        logger.info('Displaying UI model detection results...')
                        ui_detections = self.multi_detector.detect_ui(frame)
                        self.visualizer.show_detection_results(
                            frame, ui_detections, 'UI Model Detection Results'
                        )

                return False

            # Click found button
            logger.info(f'Found {button_type} button, preparing to click...')
            success = self._click_button(target_button)

            if success:
                logger.info(f'✓ Successfully clicked {button_type} button')
            else:
                logger.error(f'Failed to click {button_type} button')

            return success

        except Exception as e:
            logger.error(f'Error occurred while executing confirmation action: {e}')
            return False

    def _click_button(self, button: ButtonDetection) -> bool:
        """Click specified button."""
        try:
            # Get button center point
            center_x, center_y = button.center

            # Get capture region information for coordinate conversion
            capture_region = self.screen_capture.get_capture_region()
            if not capture_region:
                logger.error('Cannot get capture region information')
                return False

            # Get current frame for coordinate scaling calculation
            current_frame = self.screen_capture.capture_once()
            if current_frame is None:
                logger.error('Cannot get current frame')
                return False

            # Calculate coordinate conversion
            actual_height, actual_width = current_frame.shape[:2]
            region_width = capture_region['width']
            region_height = capture_region['height']

            scale_x = region_width / actual_width
            scale_y = region_height / actual_height

            # Convert to screen coordinates
            screen_x = capture_region['left'] + (center_x * scale_x)
            screen_y = capture_region['top'] + (center_y * scale_y)

            logger.info(f'Clicking {button.button_type} button: {button.class_name}')
            logger.info(f'  Detection center: ({center_x}, {center_y})')
            logger.info(f'  Screen coordinates: ({screen_x:.1f}, {screen_y:.1f})')
            logger.info(f'  Confidence: {button.confidence:.3f}')

            # Use mouse controller to click
            return self.mouse_controller.click_at(int(screen_x), int(screen_y))

        except Exception as e:
            logger.error(f'Error occurred while clicking button: {e}')
            return False

    def _show_card_action_visualization(
        self,
        image: np.ndarray,
        hand_cards: List[Detection],
        action: CardAction,
        positions: List[int],
    ) -> None:
        """
        Show visualization interface for card action.

        Args:
            image: Original image
            hand_cards: List of detected hand cards
            action: Card action object
            positions: Position array
        """
        try:
            import cv2

            # Create visualization image copy
            vis_image = image.copy()

            # Define colors
            colors = {
                'play': (0, 255, 0),  # Green - play
                'discard': (0, 0, 255),  # Red - discard
                'neutral': (128, 128, 128),  # Gray - no action
            }

            # Draw bounding boxes and labels for each card
            for i, card in enumerate(hand_cards):
                if i < len(positions):
                    pos_value = positions[i]
                    if pos_value == 1:
                        color = colors['play']
                        action_text = 'Play'
                    elif pos_value == -1:
                        color = colors['discard']
                        action_text = 'Discard'
                    else:
                        color = colors['neutral']
                        action_text = 'No Action'
                else:
                    color = colors['neutral']
                    action_text = 'No Action'

                # Draw bounding box
                x1, y1, x2, y2 = card.bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

                # Draw position label
                label = f'Position {i}: {action_text}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw text background
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - text_height - baseline - 10),
                    (x1 + text_width, y1),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - baseline - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

                # Draw position number in card center
                center_x, center_y = card.center
                cv2.circle(vis_image, (center_x, center_y), 15, (255, 255, 255), -1)
                cv2.putText(
                    vis_image,
                    str(i),
                    (center_x - 8, center_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2,
                )

            # Add title information
            title_text = f'Action Preview: {action.action_type.value} - {positions}'
            cv2.putText(
                vis_image,
                title_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Add instruction text
            instruction = 'Press any key to continue operation, ESC to cancel'
            cv2.putText(
                vis_image,
                instruction,
                (10, vis_image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )

            # Show window
            window_name = 'Balatro Card Action Preview'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.imshow(window_name, vis_image)

            # Wait for user input
            logger.info('Displaying action preview window, press any key to continue...')
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window_name)

            if key == 27:  # ESC key
                logger.info('User cancelled operation')
                raise KeyboardInterrupt('User cancelled operation')

        except Exception as e:
            logger.error(f'Error occurred while displaying visualization: {e}')
            # Continue execution, don't interrupt operation due to visualization error
