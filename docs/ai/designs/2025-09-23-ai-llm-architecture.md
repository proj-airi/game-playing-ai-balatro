---
title: "AI/LLM Architecture for Balatro Game Playing"
date: "2025-09-23"
coding_agents:
  authors: ["neko", "Claude Code"]
  project: "proj-airi/game-playing-ai-balatro"
  context: "Designing composable AI/LLM system for game decision making"
  technologies: ["LLM", "VLM", "YOLO", "OCR", "Function Calling"]
tags: ["architecture", "ai", "llm", "game-ai"]
---

# AI/LLM Architecture for Balatro Game Playing

## Overview

Design a composable, modular AI system that processes multi-modal game information (visual + text) and generates intelligent gameplay decisions.

## Architecture Components

### 1. **Core Processors** (`ai/processors/`)

#### **A. Information Processors**
- `text_processor.py`: Extract and normalize text from OCR results
- `visual_processor.py`: Process YOLO detections and spatial relationships
- `card_processor.py`: Specialized logic for understanding card information
- `game_state_processor.py`: Aggregate multi-modal information into structured game state

#### **B. Understanding Processors**
- `context_processor.py`: Build contextual understanding from game history
- `scoring_processor.py`: Understand scoring mechanics and potential
- `strategy_processor.py`: High-level strategic understanding

### 2. **AI Engines** (`ai/engines/`)

#### **A. Language Models**
- `llm_engine.py`: Text-based reasoning and decision making
- `vlm_engine.py`: Vision-Language model for visual understanding
- `function_calling_engine.py`: Tool use and action generation

#### **B. Specialized Models**
- `card_understanding_engine.py`: Fine-tuned for card game mechanics
- `strategy_engine.py`: Long-term planning and meta-strategy

### 3. **Decision Framework** (`ai/decision/`)

#### **A. Decision Types**
- `card_selection.py`: Which cards to play/keep/discard
- `blind_strategy.py`: How to approach different blind types
- `scoring_optimization.py`: Maximize score potential
- `risk_assessment.py`: Balance risk vs reward

#### **B. Action Planning**
- `action_planner.py`: Generate sequences of UI actions
- `action_executor.py`: Execute planned actions via UI automation

### 4. **Memory & Context** (`ai/memory/`)

- `game_history.py`: Track game progression and outcomes
- `learning_memory.py`: Accumulate insights from gameplay
- `context_manager.py`: Manage conversation and decision context

### 5. **Composition Framework** (`ai/composition/`)

- `pipeline_composer.py`: Chain processors dynamically
- `decision_composer.py`: Combine multiple decision strategies
- `agent_composer.py`: Orchestrate full AI agent behavior

## Data Flow

```
Game Screenshot
       ↓
[YOLO Detection] → [OCR Extraction]
       ↓                    ↓
[Visual Processor] → [Text Processor]
       ↓                    ↓
      [Game State Processor]
              ↓
      [Context Processor]
              ↓
    [Decision Composers]
              ↓
      [Action Planner]
              ↓
     [Action Executor]
```

## Key Design Principles

### 1. **Composability**
- Each processor is independent and focused
- Processors can be chained in different orders
- Easy to swap implementations (e.g., different LLM providers)

### 2. **Multi-Modal Integration**
- Visual and textual information processed separately then fused
- Flexible fusion strategies based on information quality
- Fallback mechanisms when one modality fails

### 3. **Context Awareness**
- Maintain game state across multiple turns
- Learn from past decisions and outcomes
- Adapt strategy based on game progression

### 4. **Extensibility**
- Plugin architecture for new processors
- Configuration-driven behavior
- Easy A/B testing of different strategies

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Base processor interfaces and abstract classes
2. Game state data structures
3. Basic LLM integration with function calling

### Phase 2: Information Processing
1. Text and visual processors
2. Card and game state understanding
3. Context management

### Phase 3: Decision Making
1. Decision frameworks for different game aspects
2. Action planning and execution
3. Composition and orchestration

### Phase 4: Learning & Optimization
1. Memory systems for improvement
2. Strategy adaptation
3. Performance metrics and evaluation

## Technology Choices

### **LLM Providers**
- Primary: Anthropic Claude (good at reasoning)
- Fallback: OpenAI GPT-4 (function calling)
- Local: Llama for offline scenarios

### **VLM Options**
- Claude 3.5 Sonnet (multimodal)
- GPT-4 Vision
- Local vision models for privacy

### **Function Calling**
- OpenAI format for action planning
- Custom tool definitions for UI automation
- Structured output for reliable parsing

## Example Workflows

### **Card Selection Decision**
1. Visual Processor → identifies cards in hand
2. Text Processor → extracts card descriptions
3. Card Processor → understands card effects
4. Game State Processor → builds current state
5. Strategy Processor → evaluates options
6. Decision Composer → weighs factors
7. Action Planner → generates UI actions

### **Blind Strategy Planning**
1. Text Processor → reads blind requirements
2. Context Processor → considers deck composition
3. Scoring Processor → calculates potential
4. Risk Assessment → evaluates difficulty
5. Strategy Engine → plans approach
6. Action Planner → sequences decisions

## Success Metrics

- **Decision Quality**: Win rate, score improvement
- **Efficiency**: Time per decision, action success rate
- **Adaptability**: Performance across different game scenarios
- **Reliability**: System uptime, error handling