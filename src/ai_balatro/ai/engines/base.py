"""Base classes for engines - computational resources."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class EngineType(Enum):
    """Types of computational engines."""
    LOCAL = 'local'
    API_PROVIDER = 'api_provider'
    TRANSFORMERS = 'transformers'


@dataclass
class EngineConfig:
    """Configuration for engines."""
    engine_type: EngineType
    device: Optional[str] = None  # cuda, mps, cpu
    timeout: int = 30
    max_retries: int = 3
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseEngine(ABC):
    """Abstract base class for computational engines."""

    def __init__(self, name: str, config: EngineConfig):
        self.name = name
        self.config = config
        self.is_initialized = False
        self._last_error = None

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the engine resources."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up engine resources."""
        pass

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error

    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize engine {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
