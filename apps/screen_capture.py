"""
Legacy screen capture module - DEPRECATED

This file is kept for backward compatibility.
New code should use: from core.screen_capture import ScreenCapture
"""

import warnings
import sys
from pathlib import Path

# Add path and import from new location
sys.path.insert(0, str(Path(__file__).parent))
from core.screen_capture import ScreenCapture as NewScreenCapture

warnings.warn(
    "Importing from screen_capture.py is deprecated. Use 'from core.screen_capture import ScreenCapture' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for compatibility
ScreenCapture = NewScreenCapture

# Legacy test function
def test_screen_capture():
    """Test screen capture functionality (legacy)."""
    warnings.warn("Use the new API instead", DeprecationWarning)
    capture = ScreenCapture()
    
    if capture.select_region_interactive():
        print("Region selection successful")
        capture.save_screenshot("test_capture.png")
    else:
        print("Region selection cancelled")


if __name__ == "__main__":
    test_screen_capture()
