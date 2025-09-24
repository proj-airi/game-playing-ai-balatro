import sys
from pathlib import Path

_tests_dir = Path(__file__).parent
_src_dir = _tests_dir.parent / 'src'

if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
