---
title: "Improved Game AI Architecture - General Framework Design"
date: "2025-09-25"
coding_agents:
  authors: ["neko", "Claude Code"]
  project: "proj-airi/game-playing-ai-balatro"
  context: "Complete architectural redesign for general game AI framework"
  technologies: ["Python", "Abstract Base Classes", "Dependency Injection", "Plugin Architecture"]
tags: ["architecture", "refactoring", "game-ai", "abstractions"]
---

# Improved Game AI Architecture - General Framework Design

## Executive Summary

This document outlines a comprehensive architectural redesign that transforms the current Balatro-specific codebase into a general-purpose game AI framework. The new architecture supports multiple game types, input modalities, and AI backends while maintaining clean separation of concerns and proper abstraction layers.

## Current Architecture Problems

### 1. **Architectural Inconsistencies**
- Duplicate `BaseEngine` classes in `ai/engines/base.py` and `ai/llm/base.py`
- Unclear separation between Processors, Providers, and Engines
- Mixed responsibilities (ActionExecutor inheriting from BaseProcessor)

### 2. **Tight Coupling to Balatro**
- Chinese text hardcoded in action schemas (`schemas.py:67-77`)
- Balatro-specific action types and button configurations
- Game-specific class names embedded throughout YOLO detection

### 3. **Missing General Abstractions**
- No support for different game types (turn-based vs real-time)
- No multi-modal input handling (audio, haptic, multiple monitors)
- No temporal sequence modeling or hierarchical state representation
- No planning algorithms or risk assessment frameworks

## New Architecture Design

### Core Principles
1. **Game Agnostic Core**: Framework works for any game type
2. **Plugin Architecture**: Easy to add new games, AI models, input/output systems
3. **Proper Separation of Concerns**: Clear boundaries between components
4. **Dependency Injection**: All components are testable and swappable
5. **Multi-Modal Support**: Vision, audio, text, haptic feedback
6. **Temporal Modeling**: Support for sequence learning and planning

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Game AI Framework                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Game Layer    │  │   AI Core       │  │   I/O Layer │  │
│  │   (Adapters)    │  │   (Agents)      │  │  (Modalities│  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  Perception     │  │   Reasoning     │  │   Action    │  │
│  │  (Vision, NLP)  │  │   (LLM, Plans)  │  │  (Execute)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│           Foundation Layer (Engines, Memory, Config)        │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Component Design

### 1. Foundation Layer (`framework/core/`)

#### Unified Engine Architecture
```python
# framework/core/engines/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass

class EngineCapability(Enum):
    """What capabilities an engine provides."""
    TEXT_GENERATION = "text_generation"
    VISION_PROCESSING = "vision_processing"
    AUDIO_PROCESSING = "audio_processing"
    FUNCTION_CALLING = "function_calling"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"

class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU_CUDA = "gpu_cuda"
    GPU_MPS = "gpu_mps"  # Apple Silicon
    GPU_ROCM = "gpu_rocm"  # AMD
    TPU = "tpu"
    API_ENDPOINT = "api_endpoint"

@dataclass
class EngineMetrics:
    """Performance metrics for engines."""
    latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    cost_per_operation: float = 0.0

class BaseEngine(ABC):
    """Unified base class for all computational engines."""

    def __init__(self,
                 name: str,
                 capabilities: List[EngineCapability],
                 resource_type: ResourceType,
                 config: Dict[str, Any]):
        self.name = name
        self.capabilities = capabilities
        self.resource_type = resource_type
        self.config = config
        self.metrics = EngineMetrics()
        self._is_initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize engine resources asynchronously."""
        pass

    @abstractmethod
    async def process(self,
                     input_data: Any,
                     capability: EngineCapability,
                     **kwargs) -> Dict[str, Any]:
        """Process input using specified capability."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of engine resources."""
        pass

    def supports(self, capability: EngineCapability) -> bool:
        """Check if engine supports a capability."""
        return capability in self.capabilities
```

#### Game Interface Abstractions
```python
# framework/core/game/interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import time

class GameType(Enum):
    """Classification of game types."""
    TURN_BASED = "turn_based"
    REAL_TIME = "real_time"
    HYBRID = "hybrid"
    SIMULATION = "simulation"

class GameGenre(Enum):
    """Game genre classifications."""
    CARD_GAME = "card_game"
    STRATEGY = "strategy"
    ACTION = "action"
    RPG = "rpg"
    PUZZLE = "puzzle"
    SPORTS = "sports"
    RACING = "racing"

class InputModality(Enum):
    """Types of input the game accepts."""
    VISUAL = "visual"           # Screen content
    AUDIO = "audio"             # Sound/music
    KEYBOARD = "keyboard"       # Keyboard input
    MOUSE = "mouse"             # Mouse/touch
    GAMEPAD = "gamepad"         # Controller input
    HAPTIC = "haptic"           # Force feedback
    VOICE = "voice"             # Voice commands

class OutputModality(Enum):
    """Types of output the game provides."""
    VISUAL = "visual"           # Screen display
    AUDIO = "audio"             # Sound output
    HAPTIC = "haptic"           # Vibration/force
    TEXT = "text"               # Text information

@dataclass
class GameState:
    """Universal game state representation."""
    # Core state
    timestamp: float = field(default_factory=time.time)
    game_phase: str = "unknown"

    # Multi-modal observations
    visual_observations: Dict[str, Any] = field(default_factory=dict)
    audio_observations: Dict[str, Any] = field(default_factory=dict)
    text_observations: Dict[str, Any] = field(default_factory=dict)

    # Game entities (generic)
    entities: List[Dict[str, Any]] = field(default_factory=list)

    # Player/agent state
    agent_state: Dict[str, Any] = field(default_factory=dict)

    # Temporal context
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    confidence: float = 1.0
    raw_data: Dict[str, Any] = field(default_factory=dict)

class GameAction(ABC):
    """Abstract base class for all game actions."""

    @abstractmethod
    def execute(self, game_interface: 'GameInterface') -> bool:
        """Execute this action through the game interface."""
        pass

    @abstractmethod
    def validate(self, game_state: GameState) -> bool:
        """Check if action is valid in current state."""
        pass

class GameInterface(ABC):
    """Abstract interface for any game."""

    @property
    @abstractmethod
    def game_type(self) -> GameType:
        """Get the type of game."""
        pass

    @property
    @abstractmethod
    def supported_input_modalities(self) -> List[InputModality]:
        """Get supported input types."""
        pass

    @property
    @abstractmethod
    def supported_output_modalities(self) -> List[OutputModality]:
        """Get available output types."""
        pass

    @abstractmethod
    async def observe(self) -> GameState:
        """Observe current game state."""
        pass

    @abstractmethod
    async def act(self, action: GameAction) -> bool:
        """Execute an action in the game."""
        pass

    @abstractmethod
    def get_valid_actions(self, state: GameState) -> List[GameAction]:
        """Get valid actions for current state."""
        pass
```

### 2. Perception Layer (`framework/perception/`)

#### Multi-Modal Perception
```python
# framework/perception/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

class PerceptionModality(Enum):
    """Types of perception."""
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"

@dataclass
class PerceptionResult:
    """Result from perception processing."""
    success: bool
    modality: PerceptionModality
    entities: List[Dict[str, Any]]
    features: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class BasePerceptionProcessor(ABC):
    """Base class for all perception processors."""

    @abstractmethod
    async def process(self,
                     input_data: Any,
                     context: Optional[Dict[str, Any]] = None) -> PerceptionResult:
        """Process perceptual input."""
        pass
```

#### Computer Vision Pipeline
```python
# framework/perception/vision/pipeline.py
from typing import List, Dict, Any
from .detectors import ObjectDetector
from .extractors import TextExtractor, FeatureExtractor

class VisionPipeline:
    """Configurable computer vision processing pipeline."""

    def __init__(self):
        self.detectors: List[ObjectDetector] = []
        self.text_extractors: List[TextExtractor] = []
        self.feature_extractors: List[FeatureExtractor] = []

    def add_detector(self, detector: ObjectDetector) -> 'VisionPipeline':
        """Add object detector to pipeline."""
        self.detectors.append(detector)
        return self

    def add_text_extractor(self, extractor: TextExtractor) -> 'VisionPipeline':
        """Add text extractor to pipeline."""
        self.text_extractors.append(extractor)
        return self

    def add_feature_extractor(self, extractor: FeatureExtractor) -> 'VisionPipeline':
        """Add feature extractor to pipeline."""
        self.feature_extractors.append(extractor)
        return self

    async def process_frame(self, image: Any) -> Dict[str, Any]:
        """Process a single frame through the pipeline."""
        results = {
            "detections": [],
            "text_extractions": [],
            "features": {}
        }

        # Run object detection
        for detector in self.detectors:
            detection_result = await detector.detect(image)
            results["detections"].extend(detection_result.objects)

        # Extract text from regions
        for extractor in self.text_extractors:
            text_result = await extractor.extract(image, results["detections"])
            results["text_extractions"].extend(text_result.texts)

        # Extract features
        for extractor in self.feature_extractors:
            feature_result = await extractor.extract(image, results["detections"])
            results["features"].update(feature_result.features)

        return results
```

### 3. AI Core Layer (`framework/ai/`)

#### Universal Agent Framework
```python
# framework/ai/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..core.game.interface import GameState, GameAction, GameInterface

class AgentCapability(Enum):
    """Capabilities an agent can have."""
    PLANNING = "planning"
    LEARNING = "learning"
    REASONING = "reasoning"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"

@dataclass
class AgentDecision:
    """A decision made by an agent."""
    action: GameAction
    confidence: float
    reasoning: str
    alternatives: List[Dict[str, Any]]
    expected_outcome: Optional[Dict[str, Any]] = None

class BaseGameAgent(ABC):
    """Universal base class for game-playing agents."""

    def __init__(self,
                 name: str,
                 capabilities: List[AgentCapability],
                 game_interface: GameInterface):
        self.name = name
        self.capabilities = capabilities
        self.game_interface = game_interface
        self.memory: List[Dict[str, Any]] = []

    @abstractmethod
    async def perceive(self) -> GameState:
        """Perceive current game state."""
        pass

    @abstractmethod
    async def decide(self, state: GameState) -> AgentDecision:
        """Make a decision based on game state."""
        pass

    @abstractmethod
    async def act(self, decision: AgentDecision) -> bool:
        """Execute the decided action."""
        pass

    async def play_turn(self) -> bool:
        """Execute one complete turn."""
        try:
            # Perceive → Decide → Act cycle
            state = await self.perceive()
            decision = await self.decide(state)
            success = await self.act(decision)

            # Store in memory
            self.memory.append({
                "timestamp": time.time(),
                "state": state,
                "decision": decision,
                "success": success
            })

            return success
        except Exception as e:
            logger.error(f"Error in play_turn: {e}")
            return False
```

### 4. Game-Specific Adapters (`games/`)

#### Balatro Game Adapter
```python
# games/balatro/adapter.py
from framework.core.game.interface import GameInterface, GameType, InputModality, OutputModality
from framework.core.game.interface import GameState, GameAction
from .actions import BalatroCardAction, BalatroButtonAction
from .state_extractor import BalatroStateExtractor
from .ui_controller import BalatroUIController

class BalatroGameInterface(GameInterface):
    """Balatro-specific game interface implementation."""

    def __init__(self):
        self.state_extractor = BalatroStateExtractor()
        self.ui_controller = BalatroUIController()

    @property
    def game_type(self) -> GameType:
        return GameType.TURN_BASED

    @property
    def supported_input_modalities(self) -> List[InputModality]:
        return [InputModality.VISUAL, InputModality.MOUSE, InputModality.KEYBOARD]

    @property
    def supported_output_modalities(self) -> List[OutputModality]:
        return [InputModality.VISUAL, InputModality.AUDIO]

    async def observe(self) -> GameState:
        """Extract Balatro-specific game state."""
        return await self.state_extractor.extract_state()

    async def act(self, action: GameAction) -> bool:
        """Execute Balatro-specific actions."""
        if isinstance(action, BalatroCardAction):
            return await self.ui_controller.execute_card_action(action)
        elif isinstance(action, BalatroButtonAction):
            return await self.ui_controller.execute_button_action(action)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def get_valid_actions(self, state: GameState) -> List[GameAction]:
        """Get valid Balatro actions for current state."""
        actions = []

        # Extract cards from state
        cards = state.entities.get("cards", [])

        # Generate card selection actions
        for combination in self._get_valid_card_combinations(cards):
            actions.append(BalatroCardAction(combination))

        # Add button actions based on game phase
        if state.game_phase == "playing":
            actions.extend([
                BalatroButtonAction("play"),
                BalatroButtonAction("discard")
            ])

        return actions
```

## Implementation Plan

### Phase 1: Foundation Refactoring
1. **Consolidate base classes** - Remove duplicate BaseEngine definitions
2. **Create unified engine architecture** - Single engine interface supporting multiple capabilities
3. **Implement dependency injection container** - IoC pattern for better testing
4. **Extract game-agnostic interfaces** - Move Balatro-specific code to games/ folder

### Phase 2: Multi-Modal Perception
1. **Create perception pipeline framework** - Configurable processing chains
2. **Implement modality-specific processors** - Vision, audio, text processing
3. **Add temporal sequence modeling** - Support for learning from game history
4. **Create universal state representation** - Game-agnostic state format

### Phase 3: Advanced AI Core
1. **Implement planning algorithms** - MCTS, A*, minimax for different game types
2. **Add learning frameworks** - Reinforcement learning, imitation learning
3. **Create multi-objective optimization** - Balance multiple competing goals
4. **Implement uncertainty quantification** - Confidence scoring for decisions

### Phase 4: Game Adapter Ecosystem
1. **Refactor Balatro implementation** - Use new adapter pattern
2. **Create chess adapter** - Demonstrate framework generality
3. **Add real-time game adapter** - Support for action games
4. **Implement puzzle game adapter** - Support for different game genres

## Benefits of New Architecture

### 1. **True Generalization**
- Framework supports any game type, not just card games
- Multi-modal input/output support
- Temporal reasoning and planning capabilities

### 2. **Proper Separation of Concerns**
- Clear boundaries between perception, reasoning, and action
- Game-specific code isolated in adapter layer
- AI algorithms separated from game mechanics

### 3. **Extensibility**
- Plugin architecture for new games, AI models, sensors
- Composable perception pipelines
- Swappable reasoning engines

### 4. **Production Ready**
- Proper async/await support for performance
- Comprehensive error handling and logging
- Performance metrics and monitoring
- Easy testing with dependency injection

### 5. **Multi-Game Support**
```python
# Example: Same agent code works for different games
async def run_multi_game_agent():
    # Chess agent
    chess_interface = ChessGameInterface()
    chess_agent = PlanningAgent("chess_master", chess_interface)

    # Balatro agent
    balatro_interface = BalatroGameInterface()
    balatro_agent = PlanningAgent("balatro_master", balatro_interface)

    # Same agent framework, different games
    await chess_agent.play_turn()
    await balatro_agent.play_turn()
```

## Use Case Analysis - Comprehensive Examples

### Use Case 1: Multi-Game Tournament Agent

**Scenario**: Create an AI that can compete in tournaments across different game types.

```python
async def run_tournament_agent():
    """Agent that adapts to different games in a tournament."""

    # Initialize container with shared components
    container = GameAIContainer()
    container.register(BaseEngine, OpenAIEngine(api_key="sk-..."))
    container.register(BaseEngine, CUDAEngine(model_path="./models/llama2-7b"))

    # Game interfaces
    games = {
        "chess": ChessGameInterface(),
        "balatro": BalatroGameInterface(),
        "starcraft": RTSGameInterface()
    }

    # Specialized agents for each game type
    agents = {
        "chess": PlanningGameAgent("chess_master", games["chess"], planning_depth=5),
        "balatro": LearningGameAgent("balatro_pro", games["balatro"]),
        "starcraft": ReactiveGameAgent("sc_commander", games["starcraft"], reaction_time_ms=50)
    }

    # Tournament loop
    tournament_results = {}

    for game_name, agent in agents.items():
        print(f"\nStarting {game_name} tournament...")

        # Each agent uses the same framework but different strategies
        game_results = []
        for match in range(10):  # 10 matches per game
            try:
                # Run complete game
                match_result = await run_complete_game(agent, games[game_name])
                game_results.append(match_result)

                # Agent learns from each match
                if hasattr(agent, 'learn_from_episode'):
                    await agent.learn_from_episode(match_result['episode_data'])

            except Exception as e:
                print(f"Match {match} failed: {e}")

        tournament_results[game_name] = {
            "win_rate": sum(r['won'] for r in game_results) / len(game_results),
            "avg_score": sum(r['score'] for r in game_results) / len(game_results),
            "total_matches": len(game_results)
        }

    return tournament_results

async def run_complete_game(agent: BaseGameAgent, game_interface: GameInterface) -> Dict:
    """Run a complete game and return results."""
    episode_data = []
    total_score = 0
    won = False

    while True:
        # Get current state
        state = await game_interface.observe()

        # Check if game is over
        if state.game_phase == "game_over":
            won = state.agent_state.get("won", False)
            break

        # Agent makes decision
        decision = await agent.decide(state)

        # Execute action
        success = await game_interface.act(decision.action)

        # Record for learning
        episode_data.append({
            "state": state,
            "action": decision.action,
            "reward": state.agent_state.get("score_change", 0),
            "success": success
        })

        total_score += state.agent_state.get("score", 0)

        # Prevent infinite loops
        if len(episode_data) > 1000:
            break

    return {
        "won": won,
        "score": total_score,
        "episode_data": episode_data,
        "total_moves": len(episode_data)
    }
```

### Use Case 2: Multi-Modal Learning System

**Scenario**: An AI system that learns to play games by watching human players and listening to commentary.

```python
class MultiModalLearningAgent(BaseGameAgent):
    """Agent that learns from visual, audio, and text inputs."""

    def __init__(self, name: str, game_interface: GameInterface):
        super().__init__(
            name=name,
            capabilities=[
                AgentCapability.LEARNING,
                AgentCapability.REASONING,
                AgentCapability.PREDICTION
            ],
            game_interface=game_interface
        )

        # Multi-modal processors
        self.vision_processor = VisionProcessor()
        self.audio_processor = AudioProcessor()
        self.text_processor = TextProcessor()

        # Learning components
        self.imitation_learner = ImitationLearner()
        self.pattern_matcher = PatternMatcher()

    async def learn_from_human_gameplay(self,
                                       video_file: str,
                                       audio_file: str,
                                       commentary_file: str) -> None:
        """Learn by observing human gameplay with commentary."""

        # Process all modalities in parallel
        vision_task = asyncio.create_task(
            self.vision_processor.process_video(video_file)
        )
        audio_task = asyncio.create_task(
            self.audio_processor.process_audio(audio_file)
        )
        text_task = asyncio.create_task(
            self.text_processor.process_commentary(commentary_file)
        )

        # Wait for all processing to complete
        vision_data, audio_data, text_data = await asyncio.gather(
            vision_task, audio_task, text_task
        )

        # Correlate multi-modal data
        learning_episodes = self._correlate_modalities(
            vision_data, audio_data, text_data
        )

        # Learn patterns
        for episode in learning_episodes:
            await self.imitation_learner.learn_from_episode(episode)

        print(f"Learned from {len(learning_episodes)} episodes")

    async def decide(self, state: GameState) -> AgentDecision:
        """Make decision using learned patterns."""

        # Check if we've seen similar situations
        similar_episodes = await self.pattern_matcher.find_similar(
            state, threshold=0.8
        )

        if similar_episodes:
            # Use imitation learning
            action = await self.imitation_learner.predict_action(
                state, similar_episodes
            )
            confidence = 0.9
            reasoning = f"Found {len(similar_episodes)} similar situations in training data"
        else:
            # Fall back to exploration
            valid_actions = self.game_interface.get_valid_actions(state)
            action = random.choice(valid_actions) if valid_actions else None
            confidence = 0.3
            reasoning = "Exploring - no similar situations found in training data"

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=[]
        )
```

### Use Case 3: Distributed Multi-Agent System

**Scenario**: Multiple AI agents collaborating in a team-based game.

```python
class TeamGameAgent(BaseGameAgent):
    """Agent designed for team-based games."""

    def __init__(self, name: str, role: str, team_id: str,
                 game_interface: GameInterface, communication_channel):
        super().__init__(name, [AgentCapability.REASONING], game_interface)
        self.role = role  # "scout", "attacker", "defender", etc.
        self.team_id = team_id
        self.communication = communication_channel
        self.team_strategy = None

    async def decide(self, state: GameState) -> AgentDecision:
        """Make decision considering team coordination."""

        # Get team information
        team_state = await self.communication.get_team_state()
        team_plan = await self.communication.get_team_plan()

        # Role-specific decision making
        if self.role == "scout":
            action = await self._scout_decision(state, team_state)
        elif self.role == "attacker":
            action = await self._attacker_decision(state, team_plan)
        elif self.role == "defender":
            action = await self._defender_decision(state, team_state)
        else:
            action = await self._general_decision(state)

        # Communicate decision to team
        await self.communication.broadcast_decision({
            "agent_id": self.name,
            "action": action,
            "role": self.role,
            "confidence": 0.8
        })

        return AgentDecision(
            action=action,
            confidence=0.8,
            reasoning=f"Team-coordinated {self.role} action",
            alternatives=[]
        )
```

## Performance Analysis & Optimization

### Current Performance Bottlenecks

1. **Vision Processing**: YOLO inference takes ~200ms per frame
2. **LLM Calls**: OpenRouter API calls average 1-2 seconds
3. **Action Execution**: UI automation adds ~100ms latency
4. **Memory Usage**: Conversation memory grows unbounded

### Optimization Strategies

```python
class OptimizedGameAgent(BaseGameAgent):
    """High-performance agent with optimized processing pipeline."""

    def __init__(self, name: str, game_interface: GameInterface):
        super().__init__(name, [AgentCapability.REASONING], game_interface)

        # Performance optimizations
        self.vision_cache = LRUCache(maxsize=100)  # Cache vision results
        self.decision_cache = LRUCache(maxsize=500)  # Cache decisions
        self.async_processor = AsyncProcessorPool(max_workers=4)

    async def play_optimized_turn(self) -> bool:
        """Optimized turn with parallel processing."""
        start_time = time.time()

        # Parallel processing pipeline
        tasks = [
            asyncio.create_task(self.perceive_with_cache()),
            asyncio.create_task(self.prepare_decision_context()),
            asyncio.create_task(self.preload_next_actions())
        ]

        # Wait for critical path (perception)
        state = await tasks[0]

        # Check decision cache first
        state_hash = self._hash_game_state(state)
        if state_hash in self.decision_cache:
            decision = self.decision_cache[state_hash]
            logger.debug("Using cached decision")
        else:
            # Make new decision while other tasks complete
            decision = await self.decide(state)
            self.decision_cache[state_hash] = decision

        # Execute action
        success = await self.act(decision)

        # Record metrics
        total_time = time.time() - start_time
        self.performance_metrics.record("turn_time", total_time)

        return success

    async def perceive_with_cache(self) -> GameState:
        """Optimized perception with caching."""
        # Take screenshot
        screenshot = await self.game_interface.capture_screen()

        # Check if screenshot is similar to cached ones
        screenshot_hash = self._hash_image(screenshot)
        if screenshot_hash in self.vision_cache:
            return self.vision_cache[screenshot_hash]

        # Process vision
        state = await self.game_interface.observe()
        self.vision_cache[screenshot_hash] = state

        return state
```

## Integration Examples

### Example 1: Balatro + Chess Hybrid Agent

```python
class HybridGameAgent(BaseGameAgent):
    """Agent that can switch between different games seamlessly."""

    def __init__(self, name: str):
        # No specific game interface - will be set dynamically
        super().__init__(name, [AgentCapability.REASONING], None)

        self.game_adapters = {
            "balatro": BalatroGameInterface(),
            "chess": ChessGameInterface(),
            "go": GoGameInterface()
        }

        self.current_game = None
        self.game_specific_memory = {}

    async def switch_game(self, game_name: str):
        """Switch to a different game type."""
        if game_name in self.game_adapters:
            # Save current game state
            if self.current_game:
                self.game_specific_memory[self.current_game] = self.memory.copy()
                self.memory.clear()

            # Switch to new game
            self.current_game = game_name
            self.game_interface = self.game_adapters[game_name]

            # Restore game-specific memory
            if game_name in self.game_specific_memory:
                self.memory = self.game_specific_memory[game_name]

            logger.info(f"Switched to {game_name}")
        else:
            raise ValueError(f"Unknown game: {game_name}")

    async def decide(self, state: GameState) -> AgentDecision:
        """Make game-specific decisions."""
        if not self.current_game:
            raise ValueError("No game selected")

        # Use game-specific decision logic
        if self.current_game == "balatro":
            return await self._decide_balatro(state)
        elif self.current_game == "chess":
            return await self._decide_chess(state)
        elif self.current_game == "go":
            return await self._decide_go(state)
        else:
            # Generic decision logic
            return await self._decide_generic(state)

    async def _decide_balatro(self, state: GameState) -> AgentDecision:
        """Balatro-specific decision logic."""
        # Analyze card combinations
        cards = state.entities.get("cards", [])
        best_combo = await self._analyze_poker_hands(cards)

        if best_combo:
            action = BalatroCardAction(best_combo["positions"])
            confidence = best_combo["strength"]
        else:
            action = BalatroButtonAction("discard")
            confidence = 0.3

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning="Balatro poker hand analysis",
            alternatives=[]
        )

    async def _decide_chess(self, state: GameState) -> AgentDecision:
        """Chess-specific decision logic."""
        # Use chess engine for analysis
        board_fen = state.raw_data.get("fen_position")

        # Run minimax or use chess engine
        best_move = await self._chess_engine_analysis(board_fen)

        return AgentDecision(
            action=ChessMove(best_move),
            confidence=0.8,
            reasoning="Chess engine analysis",
            alternatives=[]
        )
```

### Example 2: Production Deployment Architecture

```python
class ProductionGameAISystem:
    """Production-ready game AI system with monitoring and scaling."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.container = GameAIContainer()
        self.monitoring = MonitoringSystem()
        self.load_balancer = LoadBalancer()
        self.agents = {}

    async def initialize(self):
        """Initialize production system."""
        # Set up engines with load balancing
        engines = []

        # Add API engines
        for i in range(self.config["api_engines"]["count"]):
            engine = OpenRouterEngine(
                api_key=self.config["api_engines"]["api_key"],
                model=self.config["api_engines"]["model"]
            )
            engines.append(engine)

        # Add local GPU engines if available
        if self.config["local_engines"]["enabled"]:
            for gpu_id in range(self.config["local_engines"]["gpu_count"]):
                engine = CUDAEngine(
                    model_path=self.config["local_engines"]["model_path"],
                    device_id=gpu_id
                )
                engines.append(engine)

        # Set up load balancer
        self.load_balancer.add_engines(engines)

        # Initialize monitoring
        await self.monitoring.start()

    async def create_agent(self, agent_config: Dict[str, Any]) -> BaseGameAgent:
        """Create and configure an agent."""
        game_type = agent_config["game_type"]
        agent_type = agent_config["agent_type"]

        # Get game interface
        game_interface = self._create_game_interface(game_type)

        # Create agent based on type
        if agent_type == "planning":
            agent = PlanningGameAgent(
                name=agent_config["name"],
                game_interface=game_interface,
                planning_depth=agent_config.get("planning_depth", 3)
            )
        elif agent_type == "learning":
            agent = LearningGameAgent(
                name=agent_config["name"],
                game_interface=game_interface
            )
        elif agent_type == "reactive":
            agent = ReactiveGameAgent(
                name=agent_config["name"],
                game_interface=game_interface,
                reaction_time_ms=agent_config.get("reaction_time_ms", 100)
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Inject dependencies
        engine = await self.load_balancer.get_available_engine()
        agent.engine = engine

        # Store agent
        self.agents[agent_config["name"]] = agent

        return agent

    async def run_agent_session(self, agent_name: str, session_config: Dict[str, Any]):
        """Run an agent session with monitoring."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")

        # Start monitoring
        session_id = f"{agent_name}_{int(time.time())}"
        await self.monitoring.start_session(session_id, agent)

        try:
            # Run game session
            results = []
            for game_num in range(session_config.get("num_games", 1)):
                game_result = await run_complete_game(agent, agent.game_interface)
                results.append(game_result)

                # Log progress
                await self.monitoring.log_game_result(session_id, game_result)

        except Exception as e:
            await self.monitoring.log_error(session_id, str(e))
            raise

        finally:
            await self.monitoring.end_session(session_id)

        return results

# Usage example
async def run_production_system():
    """Run production game AI system."""

    config = {
        "api_engines": {
            "count": 3,
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "model": "anthropic/claude-3.5-sonnet"
        },
        "local_engines": {
            "enabled": True,
            "gpu_count": 2,
            "model_path": "./models/llama2-7b"
        }
    }

    system = ProductionGameAISystem(config)
    await system.initialize()

    # Create agents for different games
    agents_config = [
        {
            "name": "balatro_master",
            "game_type": "balatro",
            "agent_type": "learning",
        },
        {
            "name": "chess_grandmaster",
            "game_type": "chess",
            "agent_type": "planning",
            "planning_depth": 6
        },
        {
            "name": "starcraft_commander",
            "game_type": "starcraft",
            "agent_type": "reactive",
            "reaction_time_ms": 50
        }
    ]

    # Create all agents
    for agent_config in agents_config:
        await system.create_agent(agent_config)

    # Run concurrent sessions
    session_tasks = []
    for agent_config in agents_config:
        task = asyncio.create_task(
            system.run_agent_session(
                agent_config["name"],
                {"num_games": 10}
            )
        )
        session_tasks.append(task)

    # Wait for all sessions to complete
    results = await asyncio.gather(*session_tasks)

    print("All sessions completed!")
    for i, agent_results in enumerate(results):
        agent_name = agents_config[i]["name"]
        avg_score = sum(r["score"] for r in agent_results) / len(agent_results)
        win_rate = sum(1 for r in agent_results if r["won"]) / len(agent_results)

        print(f"{agent_name}: {win_rate:.1%} win rate, {avg_score:.1f} avg score")
```

## Benefits Summary

This comprehensive architecture provides:

1. **True Game Agnosticism**: Works with any game type - card games, board games, real-time strategy, first-person shooters, etc.

2. **Production Scalability**: Async/await architecture, load balancing, monitoring, and error handling for enterprise use.

3. **Multi-Modal Intelligence**: Processes vision, audio, text, and other sensor inputs for complete game understanding.

4. **Learning Capabilities**: Supports reinforcement learning, imitation learning, and pattern recognition across different game domains.

5. **Team Coordination**: Enables multi-agent collaboration for team-based games.

6. **Performance Optimization**: Caching, parallel processing, and efficient resource management.

7. **Extensible Plugin System**: Easy to add new games, AI models, and input/output modalities without changing core framework.

8. **Comprehensive Testing**: Dependency injection and mocking support for robust testing.

This transforms a Balatro-specific project into a comprehensive, reusable game AI framework while maintaining all existing functionality and dramatically expanding capabilities to support any type of game across multiple domains.