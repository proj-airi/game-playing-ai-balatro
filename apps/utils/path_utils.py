"""Path utilities for Balatro detection system."""

import os
from pathlib import Path
from typing import Optional, List

from utils.logger import get_logger

logger = get_logger(__name__)


def find_model_file(search_paths: List[str]) -> Optional[str]:
    """
    Find the first existing model file from a list of paths.

    Args:
        search_paths: List of paths to search

    Returns:
        Path to the first existing model file, or None if not found
    """
    for path in search_paths:
        if os.path.exists(path):
            logger.info(f'Found model file: {path}')
            return path

    logger.warning('No model file found in search paths')
    return None


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def resolve_path(path: str, relative_to: Optional[str] = None) -> str:
    """
    Resolve a path, handling relative paths.

    Args:
        path: Path to resolve
        relative_to: Base path for relative resolution. If None, uses project root.

    Returns:
        Resolved absolute path
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return str(path_obj)

    if relative_to is None:
        relative_to = get_project_root()

    return str(Path(relative_to) / path_obj)
