---
title: "Agent Framework Architecture Documentation"
date: "2025-09-24"
coding_agents:
  authors: ["neko", "Claude Code"]
  project: "proj-airi/game-playing-ai-balatro"
  context: "Complete agent framework with memory, templates, and orchestration"
  technologies: ["OpenRouter", "LLM", "Agent Framework", "Memory Management", "Prompt Templates"]
tags: ["architecture", "agents", "framework", "documentation"]
---

# Agent Framework Architecture Documentation

## Overview

Built a comprehensive, extensible agent framework for AI-driven game playing. The framework provides clean abstractions for memory management, prompt templating, and multi-agent orchestration with proper separation of concerns.

## Architecture Components

### 1. **Engine/Provider Pattern** (`ai/engines/`, `ai/providers/`)

**Design Philosophy**: Separate computational resources (engines) from AI services (providers).

#### Engines - Computational Resources
- `BaseEngine` - Abstract engine interface
- `APIProviderEngine` - HTTP API call handling with retries, backoff
- `LocalEngine` - Future: CUDA/MPS/CPU local inference
- `TransformersEngine` - Future: Local transformers integration

#### Providers - AI Services
- `BaseProvider` - Abstract provider interface
- `LLMProvider` - Base for language model providers
- `VLMProvider` - Base for vision-language models
- `OpenRouterProvider` - Production-ready OpenRouter integration

```python
# Usage
from ai_balatro.ai.providers import OpenRouterProvider

provider = OpenRouterProvider(model_name="anthropic/claude-3.5-sonnet")
with provider:
    result = provider.generate_text("Analyze this game state...")
    # or
    result = provider.function_call(prompt, GAME_ACTIONS)
```

### 2. **Memory Management** (`ai/memory/`)

**Problem Solved**: LLM conversation context management with token limits and persistence.

#### Core Components
- `Message` - Single conversation message with role, content, metadata
- `MessageRole` - Enum: SYSTEM, USER, ASSISTANT, FUNCTION, TOOL
- `ConversationState` - Manages message history with automatic token/count limits
- `ConversationMemory` - Handles multiple conversation sessions

#### Features
- **Automatic Limit Management**: Removes oldest messages when exceeding token/count limits
- **Token Estimation**: Rough 4-char = 1-token estimation
- **Persistence**: Save/load conversation state to/from JSON
- **Multiple Sessions**: Named conversation management

```python
# Usage
from ai_balatro.ai.memory import ConversationMemory, MessageRole

memory = ConversationMemory()
conversation = memory.create_conversation(
    "game_session_1",
    system_message="You are playing Balatro...",
    max_tokens=4000
)

conversation.add_user_message("Current game state: ...")
conversation.add_assistant_message("I recommend...")

# Get messages for API
api_messages = conversation.get_messages_for_api()
```

### 3. **Prompt Template System** (`ai/templates/`)

**Problem Solved**: Structured, reusable prompts with conditional sections and variable substitution.

#### Core Components
- `PromptTemplate` - Template definition with sections and variables
- `TemplateSection` - Individual template section with conditions
- `PromptTemplateManager` - Template registry and rendering
- `TemplateType` - Enum: SYSTEM, USER, ANALYSIS, PLANNING, EXECUTION

#### Features
- **Variable Substitution**: `{variable_name}` syntax
- **Conditional Sections**: Include/exclude based on context
- **Pre-built Templates**: Ready-to-use Balatro game templates
- **Complex Data Formatting**: Lists and dicts rendered nicely

```python
# Usage
from ai_balatro.ai.templates import render_prompt

context = {
    "game_state_summary": "5 cards visible, score: 1000",
    "card_list": ["Ace of Spades", "King of Hearts"],
    "card_descriptions": {"0": "High card", "1": "Face card"}
}

prompt = render_prompt("game_state_analysis", context)
```

#### Pre-built Templates
- `balatro_system` - Agent system message
- `game_state_analysis` - Game state analysis prompt
- `strategic_planning` - Strategic decision making
- `action_execution` - Action execution prompt

### 4. **Agent Framework** (`ai/agents/`)

**Problem Solved**: Structured agent execution flow with orchestration capabilities.

#### Core Components
- `BaseAgent` - Abstract agent with analyze → plan → execute flow
- `AgentContext` - Context passed between agent methods
- `AgentResult` - Standardized result format
- `AgentState` - Execution state tracking
- `AgentOrchestrator` - Multi-agent coordination

#### Agent Execution Flow
```
analyze_situation() → plan_action() → execute_action() → repeat
```

Each method returns `AgentResult` with success, reasoning, actions, and metadata.

#### Features
- **Conversation Integration**: Built-in LLM querying with conversation memory
- **Template Integration**: Easy prompt rendering within agents
- **State Management**: Track agent execution state
- **Error Handling**: Graceful failure handling and recovery
- **Orchestration**: Run multiple agents in sequence with shared context

```python
# Custom Agent Implementation
class GameAnalyzerAgent(BaseAgent):
    def analyze_situation(self, context: AgentContext) -> AgentResult:
        prompt = render_prompt("game_state_analysis", {
            "game_state_summary": context.game_state.get("summary"),
            "card_list": context.game_state.get("cards", [])
        })
        return self._llm_query(prompt)

    def plan_action(self, context: AgentContext) -> AgentResult:
        prompt = render_prompt("strategic_planning", {...})
        return self._llm_query(prompt, use_functions=True, functions=GAME_ACTIONS)

    def execute_action(self, action: dict, context: AgentContext) -> AgentResult:
        # Execute via game interface
        return AgentResult(success=True, action=action)

# Usage
agent = GameAnalyzerAgent(llm_provider)
result = agent.run(AgentContext(game_state={...}))
```

### 5. **Game Actions System** (`ai/actions/`)

**Problem Solved**: Structured function calling interface for game interactions.

#### Action Schemas
Index-based actions for integration with YOLO detection arrays:

- `hover_card(index)` - Hover over detected card
- `select_card(index)` - Select/deselect card
- `use_card(index)` - Use/activate card
- `sell_card(index)` - Sell card for money
- `purchase_card(index)` - Buy card from shop
- `play_cards()` - Play selected cards
- `discard_cards()` - Discard selected cards
- `click_button(button_name)` - Click UI button

```python
# Usage
from ai_balatro.ai.actions import GAME_ACTIONS, validate_action_parameters

# LLM function calling returns:
action = {
    "name": "select_card",
    "arguments": {"index": 2, "action": "select"}
}

if validate_action_parameters(action["name"], action["arguments"]):
    # Execute action...
```

## Integration Architecture

### Game State Flow
```
Screen Capture → YOLO Detection → Card Array (indexed) → LLM Context
                                         ↓
Agent Analysis → Strategic Planning → Function Calls → UI Automation
                                         ↓
                                  Game State Update → Memory → Next Iteration
```

### Multi-Agent Orchestration
```python
# Example: Three specialized agents
orchestrator = AgentOrchestrator()
orchestrator.register_agent(AnalyzerAgent(provider), 0)  # Analyze situation
orchestrator.register_agent(PlannerAgent(provider), 1)   # Strategic planning
orchestrator.register_agent(ExecutorAgent(provider), 2)  # Execute actions

# Run all agents with shared context
results = orchestrator.run_agents(initial_context)
```

## Implementation Benefits

### 1. **Extensibility**
- Easy to create new agent types by extending `BaseAgent`
- Plugin architecture for new engines/providers
- Template system allows domain-specific prompts

### 2. **Maintainability**
- Clean separation of concerns
- Standardized interfaces and result formats
- Comprehensive logging and error handling

### 3. **Reusability**
- Framework-agnostic design (not Balatro-specific)
- Composable components
- Shared memory and template systems

### 4. **Production Ready**
- Proper error handling and retries
- Token limit management
- Conversation persistence
- Resource cleanup (context managers)

## Example Usage Patterns

### Single Agent Execution
```python
provider = OpenRouterProvider(model_name="anthropic/claude-3.5-sonnet")
agent = GameAnalyzerAgent(provider)

with provider:
    context = AgentContext(game_state={"cards": [...], "score": 1000})
    result = agent.run(context)

    if result.success and result.action:
        # Execute the planned action
        execute_game_action(result.action)
```

### Multi-Agent Pipeline
```python
# Specialized agents
analyzer = GameAnalyzerAgent(provider)
planner = StrategyPlannerAgent(provider)
executor = ActionExecutorAgent(provider)

orchestrator = AgentOrchestrator()
orchestrator.register_agent(analyzer, 0)
orchestrator.register_agent(planner, 1)
orchestrator.register_agent(executor, 2)

with provider:
    results = orchestrator.run_agents(context)

    # Each agent builds on previous agent's output
    for result in results:
        logger.info(f"Agent result: {result.reasoning}")
```

### Custom Templates
```python
# Define custom prompt template
custom_template = PromptTemplate("card_evaluation", TemplateType.ANALYSIS)
custom_template.add_section("cards", "Cards to evaluate:\n{card_details}")
custom_template.add_section("criteria", "Evaluation criteria:\n{criteria}")
custom_template.add_section("question", "Which card should I prioritize and why?")

template_manager.register_template(custom_template)

# Use in agent
prompt = render_prompt("card_evaluation", {
    "card_details": ["Ace: High value", "Joker: Special effect"],
    "criteria": ["Score potential", "Synergy", "Risk"]
})
```

## Future Extensions

### Planned Components
- `LocalEngine` - CUDA/MPS local inference
- `TransformersEngine` - HuggingFace transformers integration
- `VLMProvider` - Vision-language model support
- `ReinforcementLearningAgent` - RL-based game learning
- `MultiModalAgent` - Combined vision + text processing

### Integration Points
- **Game State Providers**: External classes providing structured game state
- **Action Executors**: UI automation integration
- **Learning Systems**: Outcome tracking and strategy adaptation
- **Evaluation Metrics**: Performance measurement and optimization

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Individual component testing
- Mock LLM providers for isolated testing
- Template rendering validation
- Memory management verification

### Integration Tests (`tests/integration/`)
- End-to-end agent execution
- Multi-agent orchestration
- Real LLM provider integration
- Conversation flow validation

This framework provides the foundation for building sophisticated, maintainable AI agents for complex game-playing tasks while remaining general enough for other domains.