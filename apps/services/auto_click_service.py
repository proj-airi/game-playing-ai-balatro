"""Auto-click service for automated card interaction."""

import time
import platform
from typing import List, Optional
from pynput.mouse import Button
from pynput import mouse

from core.detection import Detection
from core.screen_capture import ScreenCapture
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class AutoClickService:
    """Service for automated clicking on detected cards."""

    def __init__(self, screen_capture: ScreenCapture):
        """
        Initialize auto-click service.

        Args:
            screen_capture: Screen capture instance for coordinate conversion
        """
        self.screen_capture = screen_capture
        self.mouse_controller = mouse.Controller()
        self.last_click_time = 0
        self.cooldown_seconds = settings.auto_click_cooldown
        self.last_clicked_card = None  # Prevent duplicate clicks
        self.card_keywords = settings.auto_click_card_keywords

        logger.info(f"Auto-click service initialized with {self.cooldown_seconds}s cooldown")

        # Check permissions on macOS
        if platform.system() == "Darwin":
            self._check_macos_permissions()

    def set_cooldown(self, cooldown_seconds: float) -> None:
        """
        Set click cooldown time.

        Args:
            cooldown_seconds: Cooldown time in seconds
        """
        self.cooldown_seconds = cooldown_seconds
        logger.info(f'Click cooldown set to {cooldown_seconds}s')

    def find_first_card(self, detections: List[Detection]) -> Optional[Detection]:
        """
        Find the first card (leftmost card) from detections.

        Args:
            detections: List of detection results

        Returns:
            First card detection or None if no cards found
        """
        # Filter card-type detections with more precise logic
        card_detections = []

        logger.info(f"🔍 Analyzing {len(detections)} detections for cards:")

        for i, det in enumerate(detections):
            class_name_lower = det.class_name.lower()

            # More precise card filtering - prioritize actual playable cards
            is_playable_card = False
            card_type = "unknown"

            # Prioritize joker cards and poker cards (front-facing)
            if "joker" in class_name_lower:
                is_playable_card = True
                card_type = "joker_card"
            elif "poker_card_front" in class_name_lower:
                is_playable_card = True
                card_type = "poker_card"
            elif "tarot" in class_name_lower:
                is_playable_card = True
                card_type = "tarot_card"
            elif "planet" in class_name_lower:
                is_playable_card = True
                card_type = "planet_card"
            elif "spectral" in class_name_lower:
                is_playable_card = True
                card_type = "spectral_card"
            # Fallback to general card keywords but with lower priority
            elif any(keyword in class_name_lower for keyword in self.card_keywords):
                # Exclude card descriptions and backs
                if "description" not in class_name_lower and "back" not in class_name_lower:
                    is_playable_card = True
                    card_type = "generic_card"

            logger.info(f"  {i+1}. {det.class_name} -> {card_type} ({'✓' if is_playable_card else '✗'})")
            logger.info(f"      Position: {det.bbox}, Center: {det.center}, Confidence: {det.confidence:.3f}")

            if is_playable_card:
                card_detections.append((det, card_type))

        if not card_detections:
            logger.warning("❌ No playable cards found in detections")
            return None

        logger.info(f"📋 Found {len(card_detections)} playable cards")

        # Sort by x-coordinate (leftmost first), then by priority
        def card_sort_key(card_tuple):
            det, card_type = card_tuple
            x_pos = det.bbox[0]  # x1 coordinate

            # Priority: joker > poker > tarot > planet > spectral > generic
            priority_map = {
                "joker_card": 0,
                "poker_card": 1,
                "tarot_card": 2,
                "planet_card": 3,
                "spectral_card": 4,
                "generic_card": 5
            }
            priority = priority_map.get(card_type, 6)

            return (x_pos, priority)

        card_detections.sort(key=card_sort_key)

        # Log sorted results
        logger.info("🎯 Cards sorted by position (left to right):")
        for i, (det, card_type) in enumerate(card_detections):
            marker = "👆 SELECTED" if i == 0 else ""
            logger.info(f"  {i+1}. {det.class_name} ({card_type}) at x={det.bbox[0]} {marker}")

        selected_card = card_detections[0][0]
        logger.info(f"✅ Selected first card: {selected_card.class_name} at {selected_card.center}")

        return selected_card

    def click_first_card(self, detections: List[Detection]) -> bool:
        """
        Automatically click the first card.

        Args:
            detections: List of detection results

        Returns:
            True if click was successful, False otherwise
        """
        logger.info("🖱️ Starting click_first_card process...")

        # Check cooldown
        current_time = time.time()
        time_since_last = current_time - self.last_click_time
        if time_since_last < self.cooldown_seconds:
            logger.info(f"⏰ Click on cooldown: {time_since_last:.1f}s < {self.cooldown_seconds}s")
            return False

        # Find first card
        first_card = self.find_first_card(detections)
        if not first_card:
            logger.warning("❌ No suitable card found for clicking")
            return False

        # Check if it's the same card (avoid duplicate clicks)
        card_signature = (
            f'{first_card.class_name}_{first_card.bbox[0]}_{first_card.bbox[1]}'
        )
        if self.last_clicked_card == card_signature:
            logger.info(f"🔄 Skipping duplicate click on same card: {card_signature}")
            return False

        # Calculate click position (card center)
        center_x, center_y = first_card.center

        # Get window information for coordinate conversion
        capture_region = self.screen_capture.get_capture_region()
        if not capture_region:
            logger.error("❌ Cannot get capture region information")
            return False

        # Get the actual captured image to determine scaling
        # We need to capture a frame to get the actual image dimensions
        current_frame = self.screen_capture.capture_once()
        if current_frame is None:
            logger.error("❌ Cannot capture current frame for coordinate conversion")
            return False

        # Calculate scaling factors between detection coordinates and capture region
        actual_height, actual_width = current_frame.shape[:2]
        region_width = capture_region['width']
        region_height = capture_region['height']

        scale_x = region_width / actual_width
        scale_y = region_height / actual_height

        logger.info("🔍 Coordinate conversion details:")
        logger.info(f"   Actual image size: {actual_width} x {actual_height}")
        logger.info(f"   Capture region size: {region_width} x {region_height}")
        logger.info(f"   Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

        # Convert detection coordinates to capture region coordinates
        region_x = center_x * scale_x
        region_y = center_y * scale_y

        logger.info(f"   Detection center: ({center_x}, {center_y})")
        logger.info(f"   Region center: ({region_x:.1f}, {region_y:.1f})")

        # Validate converted coordinates are within capture region
        if region_x < 0 or region_y < 0 or region_x > region_width or region_y > region_height:
            logger.error("❌ Converted coordinates out of capture region bounds!")
            logger.error(f"   Converted center: ({region_x:.1f}, {region_y:.1f})")
            logger.error(f"   Capture region size: {region_width} x {region_height}")
            return False

        # Convert to screen coordinates
        screen_x = capture_region['left'] + region_x
        screen_y = capture_region['top'] + region_y

        # Validate final screen coordinates are reasonable
        if screen_x < 0 or screen_y < 0 or screen_x > 3000 or screen_y > 2000:
            logger.error(f"❌ Invalid screen coordinates: ({screen_x}, {screen_y})")
            logger.error(f"   Card center: ({center_x}, {center_y})")
            logger.error(f"   Capture region: {capture_region}")
            return False

        try:
            # Detailed logging before click
            logger.info("🎯 Click target information:")
            logger.info(f"   📋 Card: {first_card.class_name}")
            logger.info(f"   📦 Bounding box: {first_card.bbox}")
            logger.info(f"   🎯 Detection center: ({center_x}, {center_y})")
            logger.info(f"   📐 Scaled center: ({region_x:.1f}, {region_y:.1f})")
            logger.info(f"   🖥️ Capture region: {capture_region}")
            logger.info(f"   🖱️ Final screen coordinates: ({screen_x:.1f}, {screen_y:.1f})")
            logger.info(f"   🔑 Card signature: {card_signature}")

            # Get current mouse position for reference
            current_mouse_pos = self.mouse_controller.position
            logger.info(f"   📍 Current mouse position: {current_mouse_pos}")

            # Perform click
            logger.info("🖱️ Moving mouse to target position...")
            self.mouse_controller.position = (int(screen_x), int(screen_y))
            time.sleep(0.1)  # Brief delay to ensure mouse movement

            # Verify mouse moved to correct position
            actual_mouse_pos = self.mouse_controller.position
            logger.info(f"   ✅ Mouse moved to: {actual_mouse_pos}")

            logger.info("🖱️ Performing left click...")
            self.mouse_controller.click(Button.left, 1)

            # Record successful click
            self.last_click_time = current_time
            self.last_clicked_card = card_signature

            logger.info("✅ Click completed successfully!")
            logger.info(f"   ⏰ Next click available after: {self.cooldown_seconds}s")

            return True

        except Exception as e:
            logger.error(f"❌ Auto-click failed: {e}")
            logger.error(f"   Card: {first_card.class_name}")
            logger.error(f"   Target coordinates: ({screen_x}, {screen_y})")
            return False

    def manual_click_first_card(self, detections: List[Detection]) -> bool:
        """
        Manually trigger click (bypasses cooldown and duplicate check).

        Args:
            detections: List of detection results

        Returns:
            True if click was successful, False otherwise
        """
        # Temporarily reset click restrictions
        old_last_click_time = self.last_click_time
        old_last_clicked_card = self.last_clicked_card
        self.last_click_time = 0
        self.last_clicked_card = None

        success = self.click_first_card(detections)

        if not success:
            # Restore previous state if click failed
            self.last_click_time = old_last_click_time
            self.last_clicked_card = old_last_clicked_card

        return success

    def get_click_status(self) -> dict:
        """
        Get current click status information.

        Returns:
            Dictionary with click status information
        """
        current_time = time.time()
        time_since_last_click = current_time - self.last_click_time
        can_click = time_since_last_click >= self.cooldown_seconds

        return {
            'last_click_time': self.last_click_time,
            'time_since_last_click': time_since_last_click,
            'cooldown_seconds': self.cooldown_seconds,
            'can_click': can_click,
            'last_clicked_card': self.last_clicked_card,
            'card_keywords': self.card_keywords
        }

    def print_click_status(self) -> None:
        """Print detailed click status information."""
        status = self.get_click_status()
        logger.info("🖱️ Auto-click Status:")
        logger.info(f"   ⏰ Cooldown: {status['cooldown_seconds']}s")
        logger.info(f"   🕐 Time since last click: {status['time_since_last_click']:.1f}s")
        logger.info(f"   ✅ Can click: {'Yes' if status['can_click'] else 'No'}")
        logger.info(f"   🎯 Last clicked card: {status['last_clicked_card'] or 'None'}")
        logger.info(f"   🔍 Card keywords: {status['card_keywords']}")

        # Get current mouse position
        current_pos = self.mouse_controller.position
        logger.info(f"   📍 Current mouse position: {current_pos}")

        # Get capture region info
        capture_region = self.screen_capture.get_capture_region()
        if capture_region:
            logger.info(f"   🖥️ Capture region: {capture_region}")
        else:
            logger.warning("   ❌ No capture region set")

    def _check_macos_permissions(self) -> None:
        """Check macOS accessibility permissions for mouse control."""
        try:
            # Try to get current mouse position to test permissions
            current_pos = self.mouse_controller.position
            logger.info(f"✅ macOS accessibility permissions OK - current mouse position: {current_pos}")
        except Exception as e:
            logger.error("❌ macOS accessibility permissions may be missing!")
            logger.error("   Please enable accessibility permissions for this application:")
            logger.error("   1. Go to System Preferences > Security & Privacy > Privacy")
            logger.error("   2. Select 'Accessibility' on the left")
            logger.error("   3. Add this application to the list and enable it")
            logger.error(f"   Error details: {e}")

    def test_click_permissions(self) -> bool:
        """
        Test if mouse clicking permissions are available.

        Returns:
            True if permissions are available, False otherwise
        """
        try:
            # Get current position
            current_pos = self.mouse_controller.position
            logger.info(f"📍 Testing click permissions - current mouse at: {current_pos}")

            # Try to move mouse slightly (this tests if we have control)
            test_x, test_y = current_pos
            self.mouse_controller.position = (test_x + 1, test_y)
            time.sleep(0.05)
            self.mouse_controller.position = current_pos  # Move back

            logger.info("✅ Mouse control permissions are working")
            return True

        except Exception as e:
            logger.error(f"❌ Mouse control permissions test failed: {e}")
            logger.error("   Please check macOS accessibility permissions")
            return False
