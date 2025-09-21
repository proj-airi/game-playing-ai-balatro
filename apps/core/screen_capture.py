"""Screen capture module for real-time game interface capture."""

import time
import threading
from typing import Optional, Tuple, Callable, Dict, Any, List
import numpy as np
import cv2
import mss
from PIL import Image

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ScreenCapture:
    """Screen capture system supporting real-time screenshot and region selection."""
    
    def __init__(self):
        """Initialize screen capture system."""
        self.sct = mss.mss()
        self.capture_region: Optional[Dict[str, int]] = None
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_callback: Optional[Callable] = None
        self.fps = settings.screen_capture_fps
        self.balatro_window_info: Optional[Dict[str, Any]] = None
        
        # Try to auto-detect Balatro window on initialization
        try:
            import Quartz
            logger.info("Auto-detecting Balatro window on initialization...")
            self._detect_balatro_window()
        except ImportError:
            logger.warning("pyobjc not available, manual selection mode will be used")
    
    def set_capture_region(self, x: int, y: int, width: int, height: int) -> None:
        """
        Set capture region.
        
        Args:
            x: Left coordinate
            y: Top coordinate
            width: Region width
            height: Region height
        """
        self.capture_region = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }
        logger.info(f"Set capture region: x={x}, y={y}, width={width}, height={height}")
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get screen dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        monitor = self.sct.monitors[1]  # Primary monitor
        return monitor["width"], monitor["height"]
    
    def capture_once(self, region: Optional[Dict[str, int]] = None) -> Optional[np.ndarray]:
        """
        Capture single screenshot.
        
        Args:
            region: Capture region. If None, uses current capture region or full screen.
            
        Returns:
            Captured image as numpy array, or None if capture failed
        """
        try:
            if region is None:
                region = self.capture_region or self.sct.monitors[1]
            
            # Capture using mss
            screenshot = self.sct.grab(region)
            
            # Convert to numpy array (BGRA -> BGR)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def select_region_interactive(self) -> bool:
        """
        Interactive region selection (prioritizes auto-detection).
        
        Returns:
            True if region selected successfully, False otherwise
        """
        # First try auto-detection
        if self._detect_balatro_window():
            return True
        
        # Fall back to manual selection
        logger.info("Auto-detection failed, please select game window region on screen...")
        logger.info("Press and drag mouse to select region, press ESC to cancel")
        
        # Full screen capture for selection
        full_screen = self.capture_once(self.sct.monitors[1])
        if full_screen is None:
            logger.error("Failed to capture full screen for region selection")
            return False
        
        # Create selection window
        clone = full_screen.copy()
        window_name = "Select Game Region"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        
        # Mouse callback variables
        selecting = False
        start_point = None
        end_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selecting, start_point, end_point, clone
            
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                clone = full_screen.copy()
                cv2.rectangle(clone, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow(window_name, clone)
                
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                end_point = (x, y)
        
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, clone)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                return False
            elif key == 13 and start_point and end_point:  # Enter key
                break
        
        cv2.destroyAllWindows()
        
        if start_point and end_point:
            # Calculate selection region
            x1, y1 = start_point
            x2, y2 = end_point
            
            # Ensure correct coordinates
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Scale to actual screen coordinates
            screen_width, screen_height = self.get_screen_size()
            display_height, display_width = full_screen.shape[:2]
            
            scale_x = screen_width / display_width
            scale_y = screen_height / display_height
            
            actual_x = int(x * scale_x)
            actual_y = int(y * scale_y)
            actual_width = int(width * scale_x)
            actual_height = int(height * scale_y)
            
            self.set_capture_region(actual_x, actual_y, actual_width, actual_height)
            return True
        
        return False
    
    def start_continuous_capture(self, callback: Callable[[np.ndarray], None], fps: int = 10) -> None:
        """
        Start continuous capture.
        
        Args:
            callback: Function to call with each captured frame
            fps: Capture frame rate
        """
        if self.is_capturing:
            logger.warning("Already capturing...")
            return
        
        self.frame_callback = callback
        self.fps = fps
        self.is_capturing = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"Started continuous capture at {fps} FPS")
    
    def stop_continuous_capture(self) -> None:
        """Stop continuous capture."""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        logger.info("Stopped continuous capture")
    
    def _capture_loop(self) -> None:
        """Capture loop for continuous mode."""
        frame_time = 1.0 / self.fps
        
        while self.is_capturing:
            start_time = time.time()
            
            try:
                frame = self.capture_once()
                if frame is not None and self.frame_callback:
                    self.frame_callback(frame)
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
            
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def save_screenshot(self, filename: str, region: Optional[Dict[str, int]] = None) -> bool:
        """
        Save screenshot to file.
        
        Args:
            filename: Output filename
            region: Capture region (optional)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            img = self.capture_once(region)
            if img is not None:
                cv2.imwrite(filename, img)
                logger.info(f"Screenshot saved: {filename}")
                return True
            else:
                logger.error("Failed to capture image for saving")
                return False
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return False
    
    def get_window_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current detected window information.
        
        Returns:
            Window information dictionary or None
        """
        return self.balatro_window_info
    
    def get_capture_region(self) -> Optional[Dict[str, int]]:
        """
        Get current capture region.
        
        Returns:
            Region information dictionary or None
        """
        return self.capture_region
    
    def list_all_windows(self) -> None:
        """Debug method: List all window information."""
        try:
            import Quartz
            
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID
            )
            
            logger.info("Current windows:")
            for i, window in enumerate(window_list):
                window_name = window.get('kCGWindowName', '')
                window_owner = window.get('kCGWindowOwnerName', '')
                bounds = window.get('kCGWindowBounds', {})
                
                if window_name or window_owner:  # Only show named windows
                    logger.info(f"  {i+1}. Name: '{window_name}' | App: '{window_owner}' | "
                              f"Size: {bounds.get('Width', 0)}x{bounds.get('Height', 0)}")
                    
        except Exception as e:
            logger.error(f"Failed to list windows: {e}")
    
    def _detect_balatro_window(self) -> bool:
        """
        Detect Balatro window using pyobjc.
        
        Returns:
            True if window detected successfully, False otherwise
        """
        try:
            import Quartz
            
            # Get all window information
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID
            )
            
            # Balatro game possible names
            balatro_keywords = settings.screen_capture_window_keywords
            excluded_apps = settings.screen_capture_excluded_apps
            
            candidates = []
            
            logger.info("Checking window matches...")
            for window in window_list:
                window_name = window.get('kCGWindowName', '')
                window_owner = window.get('kCGWindowOwnerName', '')
                bounds = window.get('kCGWindowBounds', {})
                
                # Skip windows without size information
                if not bounds or bounds.get('Width', 0) < 100 or bounds.get('Height', 0) < 100:
                    continue
                
                # Exclude non-game applications
                if any(excluded_app.lower() in window_owner.lower() for excluded_app in excluded_apps):
                    continue
                
                # Check for Balatro keywords
                for keyword in balatro_keywords:
                    if (keyword.lower() in window_name.lower() or 
                        keyword.lower() in window_owner.lower()):
                        
                        score = self._calculate_window_score(window_name, window_owner, bounds)
                        logger.info(f"   Found candidate window: '{window_name}' | '{window_owner}' | "
                                  f"Size: {bounds.get('Width', 0)}x{bounds.get('Height', 0)} | Score: {score}")
                        
                        candidates.append({
                            'window': window,
                            'name': window_name,
                            'owner': window_owner,
                            'bounds': bounds,
                            'score': score
                        })
                        break
            
            if not candidates:
                logger.warning("No Balatro window detected")
                logger.info("Current window list:")
                self.list_all_windows()
                return False
            
            # Select best candidate window (by score)
            best_candidate = max(candidates, key=lambda x: x['score'])
            
            window_info = best_candidate
            self.balatro_window_info = {
                'name': window_info['name'],
                'owner': window_info['owner'],
                'bounds': window_info['bounds'],
                'window_id': window_info['window'].get('kCGWindowNumber', 0)
            }
            
            # Set capture region
            bounds = window_info['bounds']
            self.set_capture_region(
                int(bounds['X']),
                int(bounds['Y']),
                int(bounds['Width']),
                int(bounds['Height'])
            )
            
            logger.info(f"Detected Balatro window: {window_info['name'] or window_info['owner']}")
            logger.info(f"   Position: ({bounds['X']}, {bounds['Y']})")
            logger.info(f"   Size: {bounds['Width']} x {bounds['Height']}")
            logger.info(f"   Score: {window_info['score']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Window detection failed: {e}")
            return False
    
    def _calculate_window_score(self, window_name: str, window_owner: str, bounds: Dict[str, Any]) -> int:
        """
        Calculate window matching score.
        
        Args:
            window_name: Window name
            window_owner: Window owner
            bounds: Window bounds
            
        Returns:
            Score (higher is better match)
        """
        score = 0
        
        # Exact window name match
        if 'balatro' in window_name.lower():
            score += 100
        
        # Exact application name match
        if 'balatro' in window_owner.lower():
            score += 50
        
        # Game windows usually have specific size ranges
        width = bounds.get('Width', 0)
        height = bounds.get('Height', 0)
        
        # Balatro game typical resolutions (relaxed constraints)
        if 600 <= width <= 1920 and 400 <= height <= 1080:
            score += 20
        
        # Prefer reasonably sized windows
        if width > 800 and height > 500:
            score += 10
        
        # Bonus for exact Balatro window sizes
        if 850 <= width <= 950 and 500 <= height <= 600:
            score += 30
        
        return score
