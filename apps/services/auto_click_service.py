"""Auto-click service for automated card interaction."""

import time
from typing import List, Optional, Tuple
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
    
    def set_cooldown(self, cooldown_seconds: float) -> None:
        """
        Set click cooldown time.
        
        Args:
            cooldown_seconds: Cooldown time in seconds
        """
        self.cooldown_seconds = cooldown_seconds
        logger.info(f"Click cooldown set to {cooldown_seconds}s")
    
    def find_first_card(self, detections: List[Detection]) -> Optional[Detection]:
        """
        Find the first card (leftmost card) from detections.
        
        Args:
            detections: List of detection results
            
        Returns:
            First card detection or None if no cards found
        """
        # Filter card-type detections
        card_detections = []
        
        for det in detections:
            class_name_lower = det.class_name.lower()
            if any(keyword in class_name_lower for keyword in self.card_keywords):
                card_detections.append(det)
        
        if not card_detections:
            return None
        
        # Sort by x-coordinate, find leftmost card
        card_detections.sort(key=lambda d: d.bbox[0])  # Sort by x1 coordinate
        return card_detections[0]
    
    def click_first_card(self, detections: List[Detection]) -> bool:
        """
        Automatically click the first card.
        
        Args:
            detections: List of detection results
            
        Returns:
            True if click was successful, False otherwise
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_click_time < self.cooldown_seconds:
            return False
        
        # Find first card
        first_card = self.find_first_card(detections)
        if not first_card:
            return False
        
        # Check if it's the same card (avoid duplicate clicks)
        card_signature = f"{first_card.class_name}_{first_card.bbox[0]}_{first_card.bbox[1]}"
        if self.last_clicked_card == card_signature:
            return False
        
        # Calculate click position (card center)
        center_x, center_y = first_card.center
        
        # Get window information for coordinate conversion
        capture_region = self.screen_capture.get_capture_region()
        if not capture_region:
            logger.error("Cannot get capture region information")
            return False
        
        # Convert to screen coordinates
        screen_x = capture_region['left'] + center_x
        screen_y = capture_region['top'] + center_y
        
        try:
            # Perform click
            logger.info(f"Preparing to click first card: {first_card.class_name}")
            logger.info(f"   Detection coordinates: ({center_x}, {center_y})")
            logger.info(f"   Screen coordinates: ({screen_x}, {screen_y})")
            logger.info(f"   Capture region: {capture_region}")
            
            self.mouse_controller.position = (screen_x, screen_y)
            time.sleep(0.1)  # Brief delay to ensure mouse movement
            self.mouse_controller.click(Button.left, 1)
            
            self.last_click_time = current_time
            self.last_clicked_card = card_signature  # Record clicked card
            logger.info("Click completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Auto-click failed: {e}")
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
            'last_clicked_card': self.last_clicked_card
        }
