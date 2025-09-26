"""Main card action engine for executing card-based actions in Balatro."""

import time
from typing import List, Optional
import numpy as np
from ...core.detection import Detection
from ...core.yolo_detector import YOLODetector
from ...core.multi_yolo_detector import MultiYOLODetector
from ...core.screen_capture import ScreenCapture
from ...utils.logger import get_logger
from ...services.card_tooltip_service import CardTooltipService
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
        self.card_tooltip_service = CardTooltipService(
            screen_capture=self.screen_capture,
            mouse_controller=self.mouse_controller,
            multi_detector=self.multi_detector
        )

        # Card hover settings
        self.enable_card_hovering = True  # Enable card hovering by default
        self.hover_before_action = True  # Hover cards before Play/Discard actions

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

    def set_card_hover_settings(
        self,
        enable_hovering: bool = True,
        hover_before_action: bool = True,
        save_debug_images: bool = False
    ) -> None:
        """
        Configure card hovering settings.

        Args:
            enable_hovering: Whether to enable card hovering
            hover_before_action: Whether to hover cards before Play/Discard actions
            save_debug_images: Whether to save debug images during hovering
        """
        self.enable_card_hovering = enable_hovering
        self.hover_before_action = hover_before_action
        self.save_debug_images = save_debug_images

        logger.info(f"Card hover settings updated: hovering={enable_hovering}, before_action={hover_before_action}, debug_images={save_debug_images}")

    def execute_card_action(
        self,
        positions: List[int],
        description: str = '',
        show_visualization: bool = False,
        return_card_descriptions: bool = False,
    ) -> dict:
        """
        Execute card action.

        Args:
            positions: Position array, like [1, 1, 1, 0] or [-1, -1, 0, 0]
            description: Action description
            show_visualization: Whether to show visualization window
            return_card_descriptions: Whether to return card descriptions from hovering

        Returns:
            Dictionary with execution results and optionally card descriptions
        """
        logger.info(f'Executing card action: {positions} - {description}')

        # Create CardAction object
        action = CardAction.from_array(positions, description)

        result = {
            'success': False,
            'card_descriptions': [],
            'action_executed': False,
            'error_message': ''
        }

        try:
            # 1. Capture current screen
            frame = self.screen_capture.capture_once()
            if frame is None:
                result['error_message'] = 'Screen capture failed'
                logger.error('Screen capture failed')
                return result

            # 2. Detect hand cards
            combined_detections: List[Detection]
            if self.multi_detector is not None:
                # Use multi-model detector's entity and UI models
                entity_detections = self.multi_detector.detect_entities(frame)
                ui_detections = self.multi_detector.detect_ui(frame)
                detections = entity_detections
                combined_detections = entity_detections + ui_detections
                logger.info(
                    'Multi-model detector found %d entities and %d UI elements',
                    len(entity_detections),
                    len(ui_detections),
                )
            elif self.yolo_detector is not None:
                # Backward compatibility: use single detector
                detections = self.yolo_detector.detect(frame)
                combined_detections = detections
                logger.info(f'Single detector detected {len(detections)} objects')
            else:
                result['error_message'] = 'No detector available'
                logger.error('No detector available')
                return result

            hand_cards = self.position_detector.get_hand_cards(detections)

            if not hand_cards:
                result['error_message'] = 'No hand cards detected'
                logger.error('No hand cards detected')
                return result

            # 3. Validate position array length
            if len(positions) > len(hand_cards):
                logger.warning(
                    f'Position array length ({len(positions)}) exceeds hand card count ({len(hand_cards)})'
                )
                # Truncate position array
                positions = positions[: len(hand_cards)]

            # 4. Hover cards to capture descriptions (NEW FEATURE)
            if self.enable_card_hovering and self.hover_before_action:
                logger.info('Capturing card descriptions before action...')

                card_descriptions = self.card_tooltip_service.collect_card_infos(
                    frame,
                    hand_cards,
                    detections=combined_detections,
                    auto_hover_missing=True,
                    save_debug_images=getattr(self, 'save_debug_images', False)
                )

                result['card_descriptions'] = card_descriptions
                logger.info(f'Successfully captured descriptions for {len(card_descriptions)} cards')

                for i, card_desc in enumerate(card_descriptions):
                    if card_desc['description_detected'] and card_desc['description_text']:
                        truncated = (
                            card_desc['description_text'][:50] + '...'
                            if len(card_desc['description_text']) > 50
                            else card_desc['description_text']
                        )
                        logger.info(f'  Card {i}: {truncated}')
                    else:
                        logger.info(f'  Card {i}: No description captured')
            else:
                logger.info('Card hovering disabled, skipping description capture')

            # 5. Show visualization (if enabled)
            if show_visualization:
                self._show_card_action_visualization(
                    frame, hand_cards, action, positions
                )

            # 6. Ensure game window focus
            logger.info('Ensuring game window has focus...')
            focus_success = self.mouse_controller.ensure_game_window_focus()
            if focus_success:
                logger.info('✓ Game window focus ready')
            else:
                logger.warning('! Window focus handling failed, continuing with operation')

            # 7. Execute click operations
            click_success = self._execute_clicks(hand_cards, action)

            if click_success:
                # 8. Execute confirmation action (play or discard)
                time.sleep(0.5)  # Action delay
                confirm_success = self._execute_confirm_action(action, show_visualization)

                if confirm_success:
                    result['success'] = True
                    result['action_executed'] = True
                    logger.info('✓ Card action executed successfully')
                else:
                    result['error_message'] = 'Failed to execute confirmation action (Play/Discard button)'
                    logger.error('Failed to execute confirmation action')
            else:
                result['error_message'] = 'Failed to execute card click operations'
                logger.error('Failed to execute card click operations')

            return result

        except Exception as e:
            result['error_message'] = f'Error occurred while executing card action: {e}'
            logger.error(f'Error occurred while executing card action: {e}')
            return result

    def get_card_descriptions(self, save_debug_images: bool = False) -> List[dict]:
        """
        Get card descriptions by hovering over all cards in hand.

        Args:
            save_debug_images: Whether to save debug images during hovering

        Returns:
            List of card description dictionaries
        """
        logger.info('Capturing card descriptions by hovering over hand cards...')

        try:
            # 1. Capture current screen
            frame = self.screen_capture.capture_once()
            if frame is None:
                logger.error('Screen capture failed')
                return []

            # 2. Detect hand cards
            if self.multi_detector is not None:
                entity_detections = self.multi_detector.detect_entities(frame)
                ui_detections = self.multi_detector.detect_ui(frame)
                detections = entity_detections + ui_detections
                logger.info(
                    'Multi-model detector found %d entities and %d UI elements',
                    len(entity_detections),
                    len(ui_detections),
                )
            elif self.yolo_detector is not None:
                detections = self.yolo_detector.detect(frame)
                logger.info(f'Single detector detected {len(detections)} objects')
            else:
                logger.error('No detector available')
                return []

            hand_cards = self.position_detector.get_hand_cards(detections)

            if not hand_cards:
                logger.error('No hand cards detected')
                return []

            card_descriptions = self.card_tooltip_service.collect_card_infos(
                frame,
                hand_cards,
                detections=detections,
                auto_hover_missing=True,
                save_debug_images=save_debug_images
            )

            logger.info(f'Successfully captured descriptions for {len(card_descriptions)} cards')
            return card_descriptions

        except Exception as e:
            logger.error(f'Error getting card descriptions: {e}')
            return []

    def get_card_descriptions_from_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        auto_hover_missing: bool = True,
        save_debug_images: bool = False
    ) -> List[dict]:
        """Collect card descriptions from an existing frame and detections."""

        hand_cards = self.position_detector.get_hand_cards(detections)

        if not hand_cards:
            logger.warning('No hand cards available when collecting descriptions')
            return []

        return self.card_tooltip_service.collect_card_infos(
            frame,
            hand_cards,
            detections=detections,
            auto_hover_missing=auto_hover_missing,
            save_debug_images=save_debug_images
        )

    def format_card_descriptions_for_llm(self, card_descriptions: List[dict]) -> str:
        """
        Format card descriptions for LLM prompt.

        Args:
            card_descriptions: List of card description dictionaries

        Returns:
            Formatted string for LLM prompt
        """
        return self.card_tooltip_service.format_card_info_for_llm(card_descriptions)

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
