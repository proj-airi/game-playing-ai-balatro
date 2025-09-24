---
title: "OpenRouter + YOLO + OCR Implementation Specification"
date: "2025-09-24"
coding_agents:
  authors: ["neko", "Claude Code"]
  project: "proj-airi/game-playing-ai-balatro"
  context: "Implementation spec for LLM decision-making feedback loop"
  technologies: ["OpenRouter", "OpenAI SDK", "YOLO", "RapidOCR", "Function Calling"]
tags: ["implementation", "llm", "openrouter", "feedback-loop"]
---

# OpenRouter + YOLO + OCR Implementation Specification

## Core Loop (Today's Target)

```
Screen → YOLO → Hover → OCR → LLM → Actions → Repeat
```

## Components to Build

### 1. OpenRouter LLM Engine (`src/ai_balatro/ai/llm/openrouter_engine.py`)

```python
class OpenRouterEngine(BaseEngine):
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model

    def generate(self, prompt: str, context: Dict = None) -> ProcessingResult
    def function_call(self, prompt: str, functions: List[Dict]) -> ProcessingResult
```

### 2. Game Actions Schema (`src/ai_balatro/ai/actions/schemas.py`)

```python
GAME_ACTIONS = [
    {
        "name": "hover_card",
        "description": "Hover over a card to see its description",
        "parameters": {"bbox": {"type": "array", "items": {"type": "number"}}}
    },
    {
        "name": "play_cards",
        "description": "Play selected cards",
        "parameters": {"card_positions": {"type": "array"}}
    },
    {
        "name": "discard_cards",
        "description": "Discard selected cards",
        "parameters": {"card_positions": {"type": "array"}}
    },
    {
        "name": "select_cards",
        "description": "Select/deselect cards",
        "parameters": {"card_positions": {"type": "array"}}
    }
]
```

### 3. AI Decision Processor (`src/ai_balatro/ai/processors/decision_processor.py`)

```python
class DecisionProcessor(BaseProcessor):
    def __init__(self, llm_engine: OpenRouterEngine):
        self.llm_engine = llm_engine

    def process(self, game_state: GameState) -> Decision:
        # Convert YOLO detections + OCR text to LLM prompt
        # Get function call decision
        # Return structured decision
```

### 4. Game Loop Agent (`src/ai_balatro/ai/agent.py`)

```python
class BalatroAgent:
    def __init__(self):
        self.yolo_detector = YOLODetector()
        self.text_extractor = TextExtractor()
        self.llm_engine = OpenRouterEngine()
        self.decision_processor = DecisionProcessor()

    def run_game_loop(self):
        while True:
            # Capture screen
            # YOLO detect cards
            # Hover on interesting cards
            # OCR extract descriptions
            # LLM decide action
            # Execute action
            # Wait/repeat
```

## Implementation Order

1. **OpenRouter Engine** - Basic LLM integration with function calling
2. **Action Schemas** - Define available game actions
3. **Decision Processor** - Convert game state to LLM decisions
4. **Game Loop Agent** - Orchestrate the full pipeline
5. **Quick Test** - Run on test images, verify each step

## Key Implementation Details

### LLM Prompt Format
```
GAME STATE:
- Cards in hand: [OCR extracted card descriptions]
- Score: [detected score]
- Blind requirements: [OCR extracted requirements]

AVAILABLE ACTIONS:
[Function calling schema]

Analyze the situation and decide the best action.
```

### Error Handling
- API failures → retry with exponential backoff
- OCR failures → skip that card, continue
- YOLO detection failures → capture new screen
- Invalid actions → log and continue loop

### Configuration (`src/ai_balatro/config/settings.py`)
```python
# Add to existing settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"
DECISION_LOOP_DELAY = 2.0  # seconds between decisions
MAX_RETRIES = 3
```

## Testing Strategy

### Unit Tests
- OpenRouter engine response parsing
- Action schema validation
- Decision processor logic

### Integration Test
- Full pipeline with test images from `test/testdata/`
- Verify: YOLO → OCR → LLM → Action sequence
- Save debug images at each step

### Manual Verification
- Run agent on actual game
- Monitor decisions via logs
- Visual debugging output

## Success Criteria

✅ LLM receives game state from YOLO + OCR
✅ LLM returns valid function calls for game actions
✅ Actions execute via existing UI automation
✅ Loop runs continuously without crashes
✅ Decisions are contextually reasonable

## Files to Create/Modify

**New Files:**
- `src/ai_balatro/ai/llm/openrouter_engine.py`
- `src/ai_balatro/ai/actions/schemas.py`
- `src/ai_balatro/ai/processors/decision_processor.py`
- `src/ai_balatro/ai/agent.py`

**Modified Files:**
- `src/ai_balatro/config/settings.py` (add OpenRouter config)

**Test Files:**
- `tests/test_openrouter_engine.py`
- `tests/test_decision_processor.py`
- `tests/test_agent_integration.py`

## Environment Setup

```bash
export OPENROUTER_API_KEY="your-key-here"
pixi run python src/ai_balatro/ai/agent.py
```

That's it. Simple, focused, implementable today.