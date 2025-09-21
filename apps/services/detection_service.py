"""Detection service for coordinating screen capture and YOLO detection."""

import time
from typing import Optional
import cv2

from core.screen_capture import ScreenCapture
from core.yolo_detector import YOLODetector
from services.auto_click_service import AutoClickService
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DetectionService:
    """Service coordinating screen capture, detection, and auto-click functionality."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_onnx: bool = False,
        enable_auto_click: bool = False,
    ):
        """
        Initialize detection service.

        Args:
            model_path: Path to YOLO model file
            use_onnx: Whether to use ONNX model format
            enable_auto_click: Whether to enable auto-click functionality
        """
        logger.info('Initializing detection service...')

        # Initialize components
        self.screen_capture = ScreenCapture()
        self.yolo_detector = YOLODetector(model_path, use_onnx)

        # Auto-click service (optional)
        self.auto_click_service = (
            AutoClickService(self.screen_capture) if enable_auto_click else None
        )

        # Detection parameters
        self.confidence_threshold = settings.detection_confidence_threshold
        self.iou_threshold = settings.detection_iou_threshold

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()

        logger.info('Detection service initialized successfully')
        if enable_auto_click:
            logger.info('Auto-click functionality enabled')

    def set_detection_params(self, confidence: float = 0.5, iou: float = 0.45) -> None:
        """
        Set detection parameters.

        Args:
            confidence: Confidence threshold
            iou: IoU threshold
        """
        self.confidence_threshold = confidence
        self.iou_threshold = iou
        logger.info(f'Detection parameters: confidence={confidence}, IoU={iou}')

    def set_auto_click_cooldown(self, cooldown: float = 1.0) -> None:
        """
        Set auto-click cooldown time.

        Args:
            cooldown: Cooldown time in seconds
        """
        if self.auto_click_service:
            self.auto_click_service.set_cooldown(cooldown)

    def select_detection_region(self) -> bool:
        """
        Select detection region (auto-detects Balatro window).

        Returns:
            True if region selected successfully, False otherwise
        """
        logger.info('Auto-detecting Balatro window...')
        return self.screen_capture.select_region_interactive()

    def run_single_detection(self, save_result: bool = True) -> bool:
        """
        Run single detection cycle.

        Args:
            save_result: Whether to save detection results to files

        Returns:
            True to continue, False to exit
        """
        logger.info('Capturing screen...')

        # Capture screen
        frame = self.screen_capture.capture_once()
        if frame is None:
            logger.error('Screen capture failed')
            return False

        logger.info(f'Capture successful, image size: {frame.shape}')

        # Log capture region for debugging
        capture_region = self.screen_capture.get_capture_region()
        if capture_region:
            logger.info(f'Capture region: {capture_region}')
            logger.info(
                f'Image vs Region size: {frame.shape[:2]} vs ({capture_region["width"]}, {capture_region["height"]})'
            )

            # Calculate scaling factors
            scale_x = frame.shape[1] / capture_region['width']
            scale_y = frame.shape[0] / capture_region['height']
            logger.info(f'Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}')

        # Run detection
        logger.info('Running YOLO detection...')
        detections = self.yolo_detector.detect(
            frame,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )

        logger.info(f'Detected {len(detections)} objects:')
        for i, det in enumerate(detections):
            logger.info(
                f'  {i + 1}. {det.class_name} (confidence: {det.confidence:.3f}) position: {det.bbox}'
            )

        # Auto-click first card if enabled
        if self.auto_click_service and detections:
            logger.info(
                '🖱️ Auto-click service is enabled, attempting to click first card...'
            )
            clicked = self.auto_click_service.click_first_card(detections)
            if clicked:
                logger.info('✅ Auto-clicked first card successfully')
            else:
                logger.info(
                    '⚠️ Auto-click was not performed (cooldown, no cards, or error)'
                )
        elif self.auto_click_service and not detections:
            logger.info('🔍 Auto-click enabled but no detections found')
        elif not self.auto_click_service:
            logger.debug('🖱️ Auto-click service disabled')

        # Visualize results
        vis_frame = self.yolo_detector.visualize_detections(frame, detections)

        # Add information overlay
        info_text = [
            f'Objects detected: {len(detections)}',
            f'Confidence threshold: {self.confidence_threshold}',
            f'IoU threshold: {self.iou_threshold}',
            f'Model: {"ONNX" if self.yolo_detector.use_onnx else "PyTorch"}',
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                vis_frame,
                text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Display results (position window to avoid game window overlap)
        window_name = settings.ui_window_names.get(
            'single_detection', 'Balatro Detection Result'
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Position window based on game window location
        self._position_display_window(window_name, 'single_detection')

        cv2.imshow(window_name, vis_frame)

        # Save results if requested
        if save_result:
            timestamp = int(time.time())
            original_filename = f'screen_capture_{timestamp}.png'
            detection_filename = f'detection_result_{timestamp}.png'

            cv2.imwrite(original_filename, frame)
            cv2.imwrite(detection_filename, vis_frame)

            logger.info('Results saved:')
            logger.info(f'  Original image: {original_filename}')
            logger.info(f'  Detection result: {detection_filename}')

        logger.info("\nPress any key to continue, press 'q' to exit...")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        return key != ord('q')

    def run_continuous_detection(self, fps: int = 2) -> None:
        """
        Run continuous detection.

        Args:
            fps: Detection frame rate
        """
        logger.info(f'Starting continuous detection (FPS: {fps})...')
        logger.info('Control keys:')
        logger.info("  'q' - Exit")
        logger.info("  's' - Save current frame")
        logger.info("  '+' - Increase confidence threshold")
        logger.info("  '-' - Decrease confidence threshold")
        logger.info('  Space - Pause/Resume')
        if self.auto_click_service:
            logger.info("  'c' - Manually trigger click on first card")
            logger.info("  'i' - Show auto-click status info")
            logger.info("  'p' - Test mouse permissions")

        frame_time = 1.0 / fps
        paused = False

        while True:
            if not paused:
                loop_start = time.time()

                # Capture screen
                frame = self.screen_capture.capture_once()
                if frame is None:
                    continue

                self.frame_count += 1

                # Run detection
                detections = self.yolo_detector.detect(
                    frame,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                )

                self.detection_count += len(detections)

                # Auto-click first card if enabled (now works in continuous mode too)
                if self.auto_click_service and detections:
                    logger.debug(
                        '🖱️ Auto-click service enabled in continuous mode, attempting to click first card...'
                    )
                    clicked = self.auto_click_service.click_first_card(detections)
                    if clicked:
                        logger.info(
                            '✅ Auto-clicked first card successfully in continuous mode'
                        )
                    else:
                        logger.debug(
                            '⚠️ Auto-click not performed (cooldown, no cards, or error)'
                        )
                elif self.auto_click_service and not detections:
                    logger.debug(
                        '🔍 Auto-click enabled but no detections found in continuous mode'
                    )

                # Visualize results
                vis_frame = self.yolo_detector.visualize_detections(frame, detections)

                # Add real-time information
                runtime = time.time() - self.start_time
                avg_fps = self.frame_count / runtime if runtime > 0 else 0
                avg_detections = (
                    self.detection_count / self.frame_count
                    if self.frame_count > 0
                    else 0
                )

                auto_click_status = 'Enabled' if self.auto_click_service else 'Disabled'
                info_text = [
                    f'Objects detected: {len(detections)}',
                    f'Confidence: {self.confidence_threshold:.2f}',
                    f'Average FPS: {avg_fps:.1f}',
                    f'Average detections: {avg_detections:.1f}',
                    f'Total frames: {self.frame_count}',
                    f'Auto-click: {auto_click_status}',
                    'Space:Pause q:Exit s:Save +/-:Adjust confidence'
                    + (' c:Click i:Info p:Test' if self.auto_click_service else ''),
                ]

                for i, text in enumerate(info_text):
                    color = (0, 255, 0) if i < 4 else (255, 255, 255)
                    cv2.putText(
                        vis_frame,
                        text,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        1,
                    )

                # Display results (position window to avoid game window overlap)
                window_name = settings.ui_window_names.get(
                    'continuous_detection', 'Balatro Real-time Detection'
                )

                # Only set window position on first frame
                if self.frame_count == 1:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    self._position_display_window(window_name, 'continuous_detection')

                cv2.imshow(window_name, vis_frame)

                # Control frame rate
                elapsed = time.time() - loop_start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = f'realtime_detection_{timestamp}.png'
                cv2.imwrite(filename, vis_frame)
                logger.info(f'Saved: {filename}')
            elif key == ord('+') or key == ord('='):
                # Increase confidence
                self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                logger.info(f'Confidence threshold: {self.confidence_threshold:.2f}')
            elif key == ord('-'):
                # Decrease confidence
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                logger.info(f'Confidence threshold: {self.confidence_threshold:.2f}')
            elif key == ord(' '):
                # Pause/Resume
                paused = not paused
                status = 'Paused' if paused else 'Resumed'
                logger.info(f'{status}')
            elif key == ord('c') and self.auto_click_service:
                # Manual click trigger
                if not paused:
                    logger.info('🖱️ Manual click triggered by user...')
                    frame = self.screen_capture.capture_once()
                    if frame is not None:
                        detections = self.yolo_detector.detect(
                            frame,
                            confidence_threshold=self.confidence_threshold,
                            iou_threshold=self.iou_threshold,
                        )
                        logger.info(
                            f'🔍 Fresh detection for manual click: {len(detections)} objects found'
                        )
                        if detections:
                            clicked = self.auto_click_service.manual_click_first_card(
                                detections
                            )
                            if clicked:
                                logger.info('✅ Manual click successful')
                            else:
                                logger.info('❌ Manual click failed')
                        else:
                            logger.info('🔍 No objects detected for manual click')
                    else:
                        logger.error('❌ Failed to capture frame for manual click')
                else:
                    logger.info('⏸️ Please resume detection first (press Space)')
            elif key == ord('i') and self.auto_click_service:
                # Show auto-click status info
                logger.info('ℹ️ Auto-click status requested by user')
                self.auto_click_service.print_click_status()
            elif key == ord('p') and self.auto_click_service:
                # Test mouse permissions
                logger.info('🧪 Testing mouse permissions...')
                self.auto_click_service.test_click_permissions()

        cv2.destroyAllWindows()

        # Display statistics
        runtime = time.time() - self.start_time
        logger.info('\nDetection statistics:')
        logger.info(f'  Runtime: {runtime:.1f}s')
        logger.info(f'  Total frames: {self.frame_count}')
        logger.info(f'  Total detections: {self.detection_count}')
        logger.info(f'  Average FPS: {self.frame_count / runtime:.1f}')
        logger.info(
            f'  Average detections/frame: {self.detection_count / self.frame_count:.1f}'
        )

    def _position_display_window(self, window_name: str, window_type: str) -> None:
        """
        Position display window to avoid overlapping with game window.

        Args:
            window_name: Name of the display window
            window_type: Type of window ('single_detection' or 'continuous_detection')
        """
        window_info = self.screen_capture.get_window_info()
        if not window_info:
            return

        bounds = window_info['bounds']
        game_x = int(bounds['X'])
        game_y = int(bounds['Y'])
        game_width = int(bounds['Width'])
        game_height = int(bounds['Height'])

        # Get screen size
        try:
            import Quartz

            main_display = Quartz.CGMainDisplayID()
            screen_width = Quartz.CGDisplayPixelsWide(main_display)
            screen_height = Quartz.CGDisplayPixelsHigh(main_display)
        except ImportError:
            try:
                monitor = self.screen_capture.sct.monitors[0]
                screen_width, screen_height = monitor['width'], monitor['height']
            except Exception:
                screen_width, screen_height = 1920, 1080

        logger.info(f'Screen size: {screen_width}x{screen_height}')

        # Position detection window to the right of game window, or below if no space
        if game_x + game_width + 400 < screen_width:
            # Enough space on the right
            display_x = game_x + game_width + 20
            display_y = game_y
        else:
            # Not enough space on right, place below
            display_x = game_x
            display_y = game_y + game_height + 20

        # Get window size from settings
        window_sizes = settings.ui_window_sizes
        if window_type in window_sizes:
            width, height = window_sizes[window_type]
        else:
            width, height = 600, 450  # Default size

        cv2.moveWindow(window_name, display_x, display_y)
        cv2.resizeWindow(window_name, width, height)
        logger.info(
            f'Display window position: ({display_x}, {display_y}), '
            f'Game window: ({game_x}, {game_y}) {game_width}x{game_height}'
        )

    def get_statistics(self) -> dict:
        """
        Get detection statistics.

        Returns:
            Dictionary with statistics information
        """
        runtime = time.time() - self.start_time
        return {
            'runtime': runtime,
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'avg_fps': self.frame_count / runtime if runtime > 0 else 0,
            'avg_detections_per_frame': self.detection_count / self.frame_count
            if self.frame_count > 0
            else 0,
        }
