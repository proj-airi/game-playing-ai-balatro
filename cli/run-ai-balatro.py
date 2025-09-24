"""
Main entry point for Balatro game detection system.
Real-time screen capture and YOLO model detection with annotation.
"""

import sys
from pathlib import Path

_cli_dir = Path(__file__).parent
_src_dir = _cli_dir.parent / 'src'

if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ai_balatro.ui.demo_app import BalatroDetectionDemo
from ai_balatro.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function."""
    try:
        # Create and setup demo application
        demo = BalatroDetectionDemo()

        if not demo.setup():
            logger.error('Demo setup failed')
            return 1

        # Run demo
        demo.run()

        # Show final statistics
        stats = demo.get_statistics()
        if stats:
            logger.info('\n📊 Final Statistics:')
            logger.info(f'  Runtime: {stats.get("runtime", 0):.1f}s')
            logger.info(f'  Total frames: {stats.get("frame_count", 0)}')
            logger.info(f'  Total detections: {stats.get("detection_count", 0)}')
            logger.info(f'  Average FPS: {stats.get("avg_fps", 0):.1f}')
            logger.info(
                f'  Average detections/frame: {stats.get("avg_detections_per_frame", 0):.1f}'
            )

        return 0

    except KeyboardInterrupt:
        logger.info('\nDemo interrupted by user')
        return 0
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
