"""Base classes and interfaces for AI/LLM processing components."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time


class ProcessorType(Enum):
    """Types of processors in the AI pipeline."""

    TEXT = 'text'
    VISUAL = 'visual'
    CONTEXT = 'context'
    DECISION = 'decision'
    ACTION = 'action'


@dataclass
class ProcessingResult:
    """Standard result format for all processors."""

    success: bool
    data: Any
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class GameState:
    """Comprehensive game state representation."""

    # Visual elements
    detected_objects: List[Dict] = field(default_factory=list)

    # Textual information
    extracted_texts: List[Dict] = field(default_factory=list)

    # Game-specific data
    cards_in_hand: List[Dict] = field(default_factory=list)
    selected_cards: List[Dict] = field(default_factory=list)
    current_blind: Optional[Dict] = None
    score_info: Dict = field(default_factory=dict)
    round_info: Dict = field(default_factory=dict)

    # Context
    turn_history: List[Dict] = field(default_factory=list)
    game_phase: str = 'unknown'

    # Metadata
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0
    raw_data: Dict = field(default_factory=dict)


class BaseProcessor(ABC):
    """Abstract base class for all processors."""

    def __init__(self, name: str, processor_type: ProcessorType):
        self.name = name
        self.processor_type = processor_type
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the processor. Return True if successful."""
        pass

    @abstractmethod
    def process(
        self, input_data: Any, context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Process input data and return structured result."""
        pass

    def __call__(
        self, input_data: Any, context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Make processor callable."""
        if not self.is_initialized:
            if not self.initialize():
                return ProcessingResult(
                    success=False,
                    data=None,
                    errors=[f'Failed to initialize {self.name}'],
                )

        start_time = time.time()
        result = self.process(input_data, context)
        result.processing_time = time.time() - start_time

        return result


class BaseEngine(ABC):
    """Abstract base class for AI engines (LLM, VLM, etc.)."""

    def __init__(self, name: str, model_config: Dict[str, Any]):
        self.name = name
        self.model_config = model_config
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the AI engine."""
        pass

    @abstractmethod
    def generate(self, prompt: str, context: Optional[Dict] = None) -> ProcessingResult:
        """Generate response from the AI model."""
        pass

    @abstractmethod
    def function_call(
        self, prompt: str, functions: List[Dict], context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Generate function calls from the AI model."""
        pass


class BaseComposer(ABC):
    """Abstract base class for composing multiple processors/engines."""

    def __init__(self, name: str):
        self.name = name
        self.components: List[Union[BaseProcessor, BaseEngine]] = []

    def add_component(self, component: Union[BaseProcessor, BaseEngine]):
        """Add a component to the composition."""
        self.components.append(component)

    @abstractmethod
    def compose(
        self, input_data: Any, context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Compose results from multiple components."""
        pass


@dataclass
class Decision:
    """Represents a decision made by the AI system."""

    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    alternatives: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionMaker(ABC):
    """Abstract base class for decision making components."""

    @abstractmethod
    def decide(self, game_state: GameState, context: Optional[Dict] = None) -> Decision:
        """Make a decision based on current game state."""
        pass
