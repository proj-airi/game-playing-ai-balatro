"""Base classes for AI providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..engines.base import BaseEngine
from ..llm.base import ProcessingResult


class ProviderType(Enum):
    """Types of AI providers."""

    LLM = 'llm'
    VLM = 'vlm'
    EMBEDDING = 'embedding'
    TTS = 'tts'
    STT = 'stt'


@dataclass
class ProviderConfig:
    """Configuration for providers."""

    provider_type: ProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, name: str, config: ProviderConfig, engine: BaseEngine):
        self.name = name
        self.config = config
        self.engine = engine
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the provider."""
        pass

    @abstractmethod
    def generate_text(
        self, prompt: str, context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Generate text response."""
        pass

    @abstractmethod
    def function_call(
        self, prompt: str, functions: List[Dict], context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Generate function calls."""
        pass

    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError(f'Failed to initialize provider {self.name}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if hasattr(self, 'shutdown'):
            self.shutdown()


class LLMProvider(BaseProvider):
    """Base class for Large Language Model providers."""

    def __init__(self, name: str, config: ProviderConfig, engine: BaseEngine):
        config.provider_type = ProviderType.LLM
        super().__init__(name, config, engine)

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass

    def set_model(self, model_name: str) -> bool:
        """Change the model."""
        self.config.model_name = model_name
        return True


class VLMProvider(BaseProvider):
    """Base class for Vision-Language Model providers."""

    def __init__(self, name: str, config: ProviderConfig, engine: BaseEngine):
        config.provider_type = ProviderType.VLM
        super().__init__(name, config, engine)

    @abstractmethod
    def analyze_image(
        self, image_data: bytes, prompt: str, context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Analyze image with text prompt."""
        pass
