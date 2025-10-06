---
title: "New Card Action Functions - Index-Based Interface"
date: "2025-10-06"
version: "1.0"
status: "completed"
coding_agents:
  authors: ["RainbowBird", "Claude Code"]
  project: "proj-airi/game-playing-ai-balatro"
  context: "Refactoring card action interface for better LLM integration"
  technologies: ["Python", "OpenRouter LLM", "Function Calling"]
  phase: "optimization"
tags: ["llm", "api-design", "function-calling", "refactoring"]
---

# New Card Action Functions - Index-Based Interface

## Overview

Refactored the card action interface to use two separate functions (`play_cards` and `discard_cards`) instead of the position array approach, making it more intuitive for LLM agents.

## Motivation

### Previous Interface (Position Array)

```python
# Old way - confusing for LLM
select_cards_by_position(positions=[1, 1, 1, 0, -1, -1])
# What does this mean? Mix of 1, -1, and 0 is unclear
```

Problems:
- 🔴 **Confusing**: LLM needs to understand 1=play, -1=discard, 0=skip
- 🔴 **Verbose**: Need to specify every position, even unused ones
- 🔴 **Error-prone**: Easy to mix positive and negative values incorrectly
- 🔴 **Not semantic**: Doesn't clearly express intent

### New Interface (Index-Based)

```python
# New way - clear and intuitive
play_cards(indices=[0, 1, 2])      # Play first three cards
discard_cards(indices=[5, 6])      # Discard cards 5 and 6
```

Benefits:
- ✅ **Clear intent**: Separate functions for play and discard
- ✅ **Concise**: Only specify cards you want to act on
- ✅ **LLM-friendly**: Natural language mapping ("play cards 0, 1, 2")
- ✅ **Type-safe**: Indices must be non-negative integers

## Changes Made

### 1. schemas.py

Added two new function definitions:

```python
{
    'name': 'play_cards',
    'description': 'Play selected cards from your hand by their index numbers (0-based)',
    'parameters': {
        'type': 'object',
        'properties': {
            'indices': {
                'type': 'array',
                'items': {'type': 'integer', 'minimum': 0},
                'description': 'Array of card indices to play'
            },
            'description': {'type': 'string'}
        },
        'required': ['indices']
    }
}

{
    'name': 'discard_cards',
    'description': 'Discard selected cards from your hand by their index numbers (0-based)',
    'parameters': {
        'type': 'object',
        'properties': {
            'indices': {
                'type': 'array',
                'items': {'type': 'integer', 'minimum': 0},
                'description': 'Array of card indices to discard'
            },
            'description': {'type': 'string'}
        },
        'required': ['indices']
    }
}
```

The old `select_cards_by_position` function has been removed. The `execute_from_array` convenience method still exists and automatically converts to the new interface.

### 2. executor.py

Added two new execution methods:

- `_execute_play_cards(args)`: Converts indices to position array, executes play action
- `_execute_discard_cards(args)`: Converts indices to position array, executes discard action

Implementation highlights:
- Validates indices are non-negative integers
- Automatically extends position array to accommodate highest index
- Converts indices to position array internally (preserves existing logic)
- Returns enhanced result with both indices and positions

### 3. balatro_agent.py

Updated prompts for LLM:

**System Message:**
```
Available actions:
- play_cards(indices=[0,1,2]): Play selected cards by their index numbers (0-based)
  Example: play_cards(indices=[0,1,2]) plays the first three cards
- discard_cards(indices=[3,4]): Discard selected cards by their index numbers (0-based)
  Example: discard_cards(indices=[3,4]) discards cards at positions 3 and 4

Card indexing:
- Cards are numbered starting from 0 (leftmost card is index 0)
- Each card in your hand has a unique index
- Use the provided card information (Card 0, Card 1, etc.) to select cards
```

**Analysis Prompt:**
```
ACTION INSTRUCTIONS:
1. Analyze the cards listed above (Card 0, Card 1, Card 2, etc.) with their descriptions
2. Identify the best poker hand you can form from these cards
3. Choose ONE action:
   
   a) If you have a strong playable hand:
      - Use play_cards(indices=[...]) with the indices of cards to play
      - Example: play_cards(indices=[0, 1, 2, 3, 4]) to play first 5 cards
   
   b) If you need better cards:
      - Use discard_cards(indices=[...]) with the indices of cards to discard
      - Example: discard_cards(indices=[5, 6, 7]) to discard last 3 cards

Remember: 
- Cards are indexed from 0 (Card 0 is the leftmost)
- You can only play OR discard in one action, not both
```

### 4. README.md

Updated documentation with:
- New function interface examples
- Comparison with legacy interface
- Advantages of new approach
- Python code examples
- Migration guide

### 5. Test Script

Created `examples/test_new_functions.py` to demonstrate:
- `play_cards` usage
- `discard_cards` usage

## Usage Examples

### For LLM Function Calling

```json
{
  "name": "play_cards",
  "arguments": {
    "indices": [0, 1, 2, 3, 4],
    "description": "Playing royal flush"
  }
}
```

```json
{
  "name": "discard_cards",
  "arguments": {
    "indices": [5, 6, 7],
    "description": "Discarding low value cards"
  }
}
```

### For Python Code

```python
from ai_balatro.ai.actions import ActionExecutor

executor = ActionExecutor(...)

# Play cards
result = executor.process({
    'function_call': {
        'name': 'play_cards',
        'arguments': {
            'indices': [0, 1, 2],
            'description': 'Playing three of a kind'
        }
    }
})

# Discard cards
result = executor.process({
    'function_call': {
        'name': 'discard_cards',
        'arguments': {
            'indices': [3, 4],
            'description': 'Discarding weak cards'
        }
    }
})
```

## Testing

Run the test script:

```bash
pixi run python examples/test_new_functions.py
```

This will guide you through testing:
1. `play_cards` function
2. `discard_cards` function

## Benefits for LLM Integration

1. **Natural Language Mapping**: 
   - "Play cards 0, 1, and 2" → `play_cards(indices=[0,1,2])`
   - Much clearer than "Use position array [1,1,1,0]"

2. **Reduced Cognitive Load**:
   - LLM doesn't need to understand position array semantics
   - Separate functions for separate actions (play vs discard)

3. **Better Error Messages**:
   - Type validation: indices must be non-negative integers
   - Clear separation: can't mix play and discard in one call

4. **Improved Reasoning**:
   - LLM can explain: "I'm playing cards 0, 1, 2 because..."
   - Instead of: "I'm using position array [1,1,1,0] because..."

## Future Improvements

Potential enhancements:
- Add `play_all_cards()` convenience function
- Add `discard_all_cards()` convenience function
- Support card selection by rank/suit: `play_cards_by_rank(['A', 'K', 'Q'])`
- Add batch operations: `batch_actions([play_cards(...), click_button(...)])`

## Internal Convenience Method

The `execute_from_array(positions)` convenience method still exists for demo scripts and testing:

```python
# This convenience method automatically converts to new interface
executor.execute_from_array([1, 1, 1, 0], "Play first 3 cards")
# Internally converts to: play_cards(indices=[0, 1, 2])

executor.execute_from_array([-1, -1, 0, 0], "Discard first 2 cards")  
# Internally converts to: discard_cards(indices=[0, 1])
```

This allows existing demo code to work without changes.

## Conclusion

The new index-based interface significantly improves the developer and LLM experience when working with card actions. It's more intuitive, concise, and maintainable. The old `select_cards_by_position` function has been removed to keep the API clean and focused.

## Authors

- RainbowBird (Human developer)
- Claude Code (AI Assistant)

---

Last updated: 2025-10-06
