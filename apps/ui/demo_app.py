"""Demo application UI for Balatro detection system."""

from typing import Optional

from services.detection_service import DetectionService
from config.settings import settings
from utils.logger import get_logger
from utils.path_utils import find_model_file

logger = get_logger(__name__)


class BalatroDetectionDemo:
    """Main demo application for Balatro detection system."""

    def __init__(self):
        """Initialize demo application."""
        self.detection_service: Optional[DetectionService] = None
        logger.info('Balatro Detection Demo initialized')

    def setup(self) -> bool:
        """
        Set up the demo application.

        Returns:
            True if setup successful, False otherwise
        """
        logger.info('🃏 Balatro Screen Detection Demo')
        logger.info('=' * 50)

        # Find model file
        model_path = find_model_file(settings.model_search_paths)
        if not model_path:
            logger.error('YOLO model file not found')
            logger.info('\nPlease ensure model file exists at one of these locations:')
            for path in settings.model_search_paths:
                logger.info(f'  - {path}')
            return False

        logger.info(f'Found model: {model_path}')

        # Ask about auto-click
        logger.info('\n🖱️ Auto-click settings:')
        auto_click_input = (
            input('Enable auto-click on first card? (y/N): ').strip().lower()
        )
        auto_click = auto_click_input in ['y', 'yes']

        # Create detection service
        try:
            self.detection_service = DetectionService(
                model_path=model_path, use_onnx=False, enable_auto_click=auto_click
            )
        except Exception as e:
            logger.error(f'Initialization failed: {e}')
            return False

        # Auto-detect Balatro window
        if not self.detection_service.select_detection_region():
            logger.error('Balatro window not detected')
            logger.info('💡 Please ensure:')
            logger.info('   1. Balatro game is running')
            logger.info('   2. Game window is visible and not obscured')
            logger.info("   3. Window title contains 'Balatro' keyword")
            return False

        # Configure auto-click parameters
        if auto_click:
            logger.info('\n🖱️ Auto-click parameter setup:')
            cooldown_input = input(
                'Click cooldown time (seconds, default 1.0): '
            ).strip()
            if cooldown_input:
                try:
                    cooldown = float(cooldown_input)
                    self.detection_service.set_auto_click_cooldown(cooldown)
                except ValueError:
                    logger.warning('Invalid input, using default value')

        # Configure detection parameters
        logger.info('\n🎯 Detection parameter setup...')
        confidence = input('Confidence threshold (0.1-0.9, default 0.5): ').strip()
        if confidence:
            try:
                self.detection_service.set_detection_params(
                    confidence=float(confidence)
                )
            except ValueError:
                logger.warning('Invalid input, using default value')

        return True

    def run(self) -> None:
        """Run the demo application."""
        if not self.detection_service:
            logger.error('Detection service not initialized')
            return

        # Select run mode
        logger.info('\n🚀 Select run mode:')
        logger.info('  1. Single detection')
        logger.info('  2. Continuous detection')

        mode = input('Please select (1/2, default 2): ').strip()

        if mode == '1':
            # Single detection mode
            logger.info('\n📸 Single detection mode')
            while True:
                if not self.detection_service.run_single_detection():
                    break
        else:
            # Continuous detection mode
            logger.info('\n🎥 Continuous detection mode')
            fps = input('Detection frame rate (1-10, default 2): ').strip()
            if fps:
                try:
                    fps = int(fps)
                    fps = max(1, min(10, fps))
                except ValueError:
                    fps = 2
            else:
                fps = 2

            self.detection_service.run_continuous_detection(fps=fps)

        logger.info('\n👋 Detection demo ended')

    def get_statistics(self) -> dict:
        """
        Get detection statistics.

        Returns:
            Statistics dictionary
        """
        if self.detection_service:
            return self.detection_service.get_statistics()
        return {}
