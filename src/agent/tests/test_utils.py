"""Utilities for standardized test output management."""

import tempfile
import shutil
from pathlib import Path
from typing import Optional
import pytest


class TestOutputManager:
    """Manages test output directories with standardized naming."""

    def __init__(self, test_name: str, keep_outputs: bool = True):
        """Initialize test output manager.

        Args:
            test_name: Name of the test (e.g., 'vlm_processing', 'image_cropping')
            keep_outputs: Whether to keep outputs after test completion
        """
        self.test_name = test_name
        self.keep_outputs = keep_outputs
        self.base_dir = Path('.output')
        self.test_dir = self.base_dir / test_name
        self._temp_dirs = []

    def get_output_dir(self, subtest_name: Optional[str] = None) -> Path:
        """Get output directory for a test or subtest.

        Args:
            subtest_name: Optional subtest name for organization

        Returns:
            Path to output directory
        """
        if subtest_name:
            output_dir = self.test_dir / subtest_name
        else:
            output_dir = self.test_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_temp_dir(self, prefix: Optional[str] = None) -> Path:
        """Get temporary directory that will be cleaned up.

        Args:
            prefix: Optional prefix for temp directory name

        Returns:
            Path to temporary directory
        """
        if prefix:
            temp_prefix = f'{self.test_name}_{prefix}_'
        else:
            temp_prefix = f'{self.test_name}_'

        temp_dir = Path(tempfile.mkdtemp(prefix=temp_prefix))
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.keep_outputs:
            self.cleanup()


@pytest.fixture
def output_manager():
    """Pytest fixture for test output management."""

    def _get_manager(test_name: str, keep_outputs: bool = True):
        return TestOutputManager(test_name, keep_outputs)

    return _get_manager


def get_standard_test_output_dir(test_category: str, test_name: str) -> Path:
    """Get standardized test output directory.

    Args:
        test_category: Category like 'vlm_processing', 'image_cropping'
        test_name: Specific test name

    Returns:
        Path to output directory
    """
    base_dir = Path(__file__).parent / 'outputs' / test_category / test_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def cleanup_old_test_outputs(max_age_days: int = 7):
    """Clean up old test outputs to prevent disk bloat.

    Args:
        max_age_days: Remove outputs older than this many days
    """
    import time

    outputs_dir = Path(__file__).parent / 'outputs'
    if not outputs_dir.exists():
        return

    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

    for item in outputs_dir.rglob('*'):
        if item.is_file() and item.stat().st_mtime < cutoff_time:
            try:
                item.unlink()
            except OSError:
                pass
        elif item.is_dir() and not any(item.iterdir()):
            try:
                item.rmdir()
            except OSError:
                pass
