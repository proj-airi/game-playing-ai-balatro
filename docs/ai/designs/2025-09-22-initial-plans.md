---
title: "Initial Plans"
date: "2025-09-22"
coding_agents:
  authors: ["Neko Ayaka", "RainbowBird", "Claude Code"]
  project: "proj-airi/game-playing-ai-balatro"
  context: "AI system architecture for Balatro game playing agent"
  technologies: ["YOLO", "VLM", "LLM", "PyTorch", "ONNX", "FastVLM", "OpenRouter"]
tags: ["ai-design", "computer-vision", "game-ai", "multi-modal", "yolo", "vlm"]
---

# Initial Plans

## Executive Summary

This document outlines the multi-modal AI system for playing Balatro (2024 card game), combining computer vision (YOLO), vision-language models (VLM), large language models (LLM), and UI automation to create an intelligent game-playing agent that leverages the game's built-in accessibility features.

## Context & Motivation

### Project Background
- **Game**: Balatro (2024) - A poker-themed roguelike deckbuilder
- **Challenge**: Complex game state with hundreds of card variants and strategic depth
- **Approach**: Multi-modal AI combining visual detection, language understanding, and strategic reasoning

### Key Innovation: The Hover-Description Bridge
Rather than manually annotating every card variant (impractical with Balatro's extensive card catalog), we leverage the game's built-in hover descriptions as an information bridge between visual detection and language understanding.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Screen Capture │───▶│  YOLO Detection  │───▶│ Coordinate Map  │
│   (MSS/PyAutoGUI)│    │ (Ultralytics 11) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Game State     │◀───│   UI Automation  │◀───│  Action System  │
│  Tracking       │    │   (Mouse/Hover)  │    │   (Decision)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        ▲
        ▼                        ▼                        │
┌─────────────────┐    ┌──────────────────┐               │
│   VLM System    │───▶│  LLM Decision    │──────────────┘
│   (FastVLM      │    │     Engine       │
│    ~500ms)      │    │ (OpenRouter/     │
└─────────────────┘    │  Ollama)         │
                       └──────────────────┘
```

## Available Assets & Resources

### Pre-Trained YOLO Model
- **Location**: `models/games-balatro-2024-yolo-entities-detection/`
- **Formats**: PyTorch (model.pt), ONNX (onnx/model.onnx)
- **Training Data**: `data/datasets/games-balatro-2024-entities-detection/`
- **HuggingFace**: Published under `proj-airi/games-balatro-2024-yolo-entities-detection`

### YOLO Detection Classes
```
card_description       # Hover tooltip descriptions ⭐ KEY
card_pack             # Card packs for purchase
joker_card            # Special joker cards
planet_card           # Planet enhancement cards
poker_card_back       # Face-down cards
poker_card_description # Poker card tooltips ⭐ KEY
poker_card_front      # Face-up poker cards
poker_card_stack      # Multiple stacked cards
spectral_card         # Spectral effect cards
tarot_card           # Tarot enhancement cards
```

## Core Implementation Strategy

### 1. Visual Detection Pipeline

**Technology Stack**:
- **YOLO v11** (Ultralytics) for entity detection
- **Device Support**: CUDA (Linux/Windows) → MPS (macOS) → CPU fallback
- **Performance Target**: 2-5 FPS for game AI responsiveness

**Critical Design Decision**:
Focus on detecting card *locations* and *description tooltips* rather than specific card content. This allows us to handle Balatro's extensive card variety without exhaustive annotation.

### 2. Information Extraction via Hover Descriptions

**Workflow**:
1. YOLO detects card positions (`poker_card_front`, `joker_card`, etc.)
2. UI automation hovers over cards of interest
3. YOLO detects description tooltip (`card_description`, `poker_card_description`)
4. VLM extracts structured information from tooltip text
5. LLM incorporates card info into strategic decision making

**Advantages**:
- ✅ No need for manual annotation of 100+ card variants
- ✅ Uses official game descriptions (always accurate)
- ✅ Automatically handles new cards and game updates
- ✅ Leverages existing accessibility features

### 3. Technology Integration

**Screen Capture**:
- **MSS** for fast cross-platform screen capture
- **PyAutoGUI** for mouse control and UI automation
- **OpenCV** for image processing and coordinate transformation

**AI Models**:
- **VLM**: FastVLM (~500ms on Apple Silicon + CoreML)
- **LLM**: OpenRouter (development) → Ollama (local deployment)
- **Inference**: PyTorch for CUDA/MPS, ONNX for broader compatibility

## Game State Understanding

### UI Layout Analysis (Based on Sample Screenshot)
- **Left Panel**: Score (16,196), round info, blind requirements
- **Hand Area**: Current poker hand (5 cards selected, 8 cards available)
- **Controls**: Blue "Play" button, Red "Discard" button
- **Card Descriptions**: Hover tooltips (visible in screenshot) ⭐

### Information Gaps & Solutions
**Missing from YOLO Training**:
- Score/numerical displays → **OCR integration**
- Button states → **Color/shape analysis**
- Round indicators → **Template matching**
- Resource counters → **VLM for complex UI**

## Implementation Phases

### Phase 1: Foundation (Current Priority)
- [x] YOLO class analysis and capabilities assessment
- [ ] Screen capture system with device detection
- [ ] YOLO inference pipeline (PyTorch + ONNX)
- [ ] Basic UI automation and coordinate mapping
- [ ] Visual debugging system (save intermediate images)

### Phase 2: Intelligence Integration
- [ ] VLM system for description extraction
- [ ] LLM integration for strategic decision making
- [ ] Game state tracking and persistence
- [ ] Action validation and error recovery

### Phase 3: Optimization & Robustness
- [ ] Performance profiling and optimization
- [ ] Multi-resolution and DPI handling
- [ ] Advanced error recovery mechanisms
- [ ] Learning from gameplay outcomes

### Phase 4: Advanced Features
- [ ] Real-time strategy adaptation
- [ ] Multi-session learning
- [ ] Performance analytics
- [ ] Competitive play optimization

## Technical Requirements

### Performance Targets
- **Detection Latency**: <200ms per frame
- **Action Response**: <1 second total pipeline
- **Memory Usage**: <2GB RAM
- **Accuracy**: >95% for card detection, >90% for description extraction

### Cross-Platform Support
- **macOS**: MPS acceleration, CoreML for VLM
- **Linux**: CUDA acceleration, standard PyTorch
- **Windows**: CUDA acceleration, ONNX runtime support

### Development Environment
- **Package Manager**: Pixi (conda/mamba alternative with uv)
- **Dependencies**: Pre-configured in pixi.toml
- **Testing**: Pytest with visual debugging capabilities
- **Version Control**: Git LFS for models, submodules for HuggingFace repos

## Risk Assessment & Mitigation

### Technical Risks
1. **YOLO Detection Failures**
   - *Mitigation*: Multiple confidence thresholds, fallback detection methods
2. **VLM Response Latency**
   - *Mitigation*: Caching, async processing, model optimization
3. **UI Automation Timing**
   - *Mitigation*: Adaptive delays, state validation, retry mechanisms

### Operational Risks
1. **Game Updates Breaking Detection**
   - *Mitigation*: Automated testing pipeline, model retraining capability
2. **Hardware Compatibility**
   - *Mitigation*: Comprehensive device support, graceful fallbacks
3. **User Environment Variations**
   - *Mitigation*: Robust configuration system, debugging tools

## Success Metrics

### Technical KPIs
- Detection accuracy >95%
- End-to-end latency <1s
- System uptime >99%
- Memory efficiency <2GB

### Gameplay KPIs
- Game completion rate
- Score progression over time
- Strategy consistency
- Win rate by difficulty level

## Future Considerations

### Short-Term Enhancements
- Dynamic FPS adjustment based on game state
- Advanced coordinate calibration for different screen configurations
- Comprehensive logging and analytics system

### Long-Term Vision
- Reinforcement learning integration for strategy optimization
- Multi-game support (other card games)
- Real-time adaptation to opponent strategies
- Community strategy sharing and learning

---

**Next Steps**: Begin implementation with Phase 1 foundation components, starting with screen capture system and YOLO inference pipeline.

**Document Status**: Living document - will be updated as implementation progresses and new insights emerge.

**Related Files**:
- `CLAUDE.md` - Development guidelines and conventions
- `.cursorrules` - IDE-specific development rules
- `apps/yolo_detector.py` - Reference implementation for YOLO inference
