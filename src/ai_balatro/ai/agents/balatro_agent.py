"""Balatro game playing agent with LLM reasoning and strategic planning."""

import time
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from ..providers.base import LLMProvider
from ..actions.executor import ActionExecutor
from ..actions.schemas import GAME_ACTIONS
from ...core.multi_yolo_detector import MultiYOLODetector
from ...core.screen_capture import ScreenCapture
from ...core.detection import Detection
from ...services.ui_text_service import UITextExtractionService
from ...services.game_state_extraction import GameStateExtractionService
from ...utils.logger import get_logger

logger = get_logger(__name__)


class BalatroReasoningAgent(BaseAgent):
    """
    Balatro-specific reasoning agent that combines vision processing,
    LLM strategic planning, and action execution.
    """

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        screen_capture: ScreenCapture,
        multi_detector: Optional[MultiYOLODetector] = None,
        **kwargs,
    ):
        """
        Initialize Balatro reasoning agent.

        Args:
            name: Agent identifier
            llm_provider: LLM provider for reasoning
            screen_capture: Screen capture for game state observation
            multi_detector: Multi-YOLO detector for entity detection
        """
        super().__init__(name, llm_provider, **kwargs)

        self.screen_capture = screen_capture
        self.multi_detector = multi_detector or MultiYOLODetector()

        # Initialize action executor
        self.action_executor = ActionExecutor(
            screen_capture=screen_capture, multi_detector=self.multi_detector
        )
        self.action_executor.initialize()

        self.ui_text_service = UITextExtractionService()
        self.game_state_extractor = GameStateExtractionService(
            screen_capture=self.screen_capture,
            multi_detector=self.multi_detector,
            mouse_controller=self.action_executor.card_engine.mouse_controller,
            card_tooltip_service=self.action_executor.card_engine.card_tooltip_service,
            ui_text_service=self.ui_text_service,
        )

        # Game state tracking
        self.current_game_state: Dict[str, Any] = {}
        self.game_history: List[Dict[str, Any]] = []

        logger.info(f"BalatroReasoningAgent '{name}' initialized")

    def run(self, context: AgentContext) -> AgentResult:
        """Run a complete reasoning cycle with integrated decision making."""
        logger.info(f'Running {self.name} - integrated decision cycle')

        try:
            result = self.analyze_situation(context)

            if result.success and result.action:
                execution_result = self.execute_action(result.action, result.context)
                return execution_result
            elif result.success:
                return result
            else:
                return result

        except Exception as e:
            logger.error(f'Integrated run cycle failed: {e}')
            return AgentResult(
                success=False, errors=[f'Run cycle failed: {e}'], state=AgentState.ERROR
            )

    def _get_system_message(self) -> str:
        """Get Balatro-specific system message."""
        return """You are a Balatro expert AI agent. You analyze game states and make immediate strategic decisions to maximize scores.

Your capabilities:
- Analyze detected cards with full OCR text descriptions (card names, ranks, suits)
- Understand poker hands and scoring mechanics
- Make strategic card plays and discards based on card indices
- Execute actions through simple function calls

Available actions:
- play_cards(indices=[0,1,2]): Play selected cards by their index numbers (0-based)
  Example: play_cards(indices=[0,1,2]) plays the first three cards
- discard_cards(indices=[3,4]): Discard selected cards by their index numbers (0-based)
  Example: discard_cards(indices=[3,4]) discards cards at positions 3 and 4
- click_button(button_type='...'): Press UI buttons (play, discard, skip, shop, next)

Card indexing:
- Cards are numbered starting from 0 (leftmost card is index 0)
- Each card in your hand has a unique index
- Use the provided card information (Card 0, Card 1, etc.) to select cards

Make immediate, optimal decisions based on the complete card information provided."""

    def analyze_situation(self, context: AgentContext) -> AgentResult:
        """Analyze current Balatro game state and make immediate strategic decision."""
        try:
            logger.info('Analyzing Balatro game situation and planning action...')

            game_state = self.game_state_extractor.capture_state(
                capture_card_descriptions=True
            )

            if game_state is None:
                return AgentResult(
                    success=False,
                    errors=['Failed to capture screen'],
                    state=AgentState.ERROR,
                )

            entities_detection = game_state.get('entities_raw', [])
            ui_detection = game_state.get('ui_elements_raw', [])

            # Update context with current state
            context.game_state.update(game_state)
            self.current_game_state = game_state

            # Create strategic decision prompt
            decision_prompt = self._create_analysis_prompt(game_state)

            # Query LLM with function calling for immediate action decision
            result = self._llm_query(
                decision_prompt, use_functions=True, functions=GAME_ACTIONS
            )

            if result.success:
                logger.info(f'Strategic decision: {result.reasoning}')
                return AgentResult(
                    success=True,
                    action=result.action,  # Action is planned immediately
                    reasoning=result.reasoning,
                    context=context,
                    metadata={
                        'game_state': game_state,
                        'entities_count': len(entities_detection),
                        'ui_elements_count': len(ui_detection),
                    },
                )
            else:
                return AgentResult(
                    success=False, errors=result.errors, state=AgentState.ERROR
                )

        except Exception as e:
            logger.error(f'Failed to analyze situation and plan action: {e}')
            return AgentResult(
                success=False,
                errors=[f'Analysis/planning failed: {e}'],
                state=AgentState.ERROR,
            )

    def plan_action(self, context: AgentContext) -> AgentResult:
        """Plan strategic action - now integrated into analyze_situation for efficiency."""
        logger.info(
            'Action planning is now integrated into analyze_situation for efficiency'
        )

        # Check if action was already planned during analysis
        if hasattr(context, 'planned_action') and context.planned_action:
            return AgentResult(
                success=True,
                action=context.planned_action,
                reasoning='Action already planned during analysis phase',
                context=context,
                metadata={'reused_planned_action': True},
            )

        # Fallback: re-run analysis if no action planned
        return self.analyze_situation(context)

    def execute_action(
        self, action: Dict[str, Any], context: AgentContext
    ) -> AgentResult:
        """Execute the planned action using the action executor."""
        try:
            logger.info(f'Executing action: {action}')

            # Execute action through action executor
            execution_result = self.action_executor.process({'function_call': action})

            # Record action in game history
            action_record = {
                'timestamp': time.time(),
                'action': action,
                'game_state_before': self.current_game_state.copy(),
                'success': execution_result.success,
                'errors': execution_result.errors
                if not execution_result.success
                else [],
            }
            self.game_history.append(action_record)

            if execution_result.success:
                logger.info('Action executed successfully')
                return AgentResult(
                    success=True,
                    action=action,
                    reasoning=f'Successfully executed {action.get("name", "unknown action")}',
                    context=context,
                    metadata={
                        'execution_data': execution_result.data,
                        'execution_time': time.time(),
                    },
                )
            else:
                logger.error(f'Action execution failed: {execution_result.errors}')
                return AgentResult(
                    success=False,
                    action=action,
                    errors=execution_result.errors,
                    state=AgentState.ERROR,
                )

        except Exception as e:
            logger.error(f'Failed to execute action: {e}')
            return AgentResult(
                success=False,
                action=action,
                errors=[f'Execution failed: {e}'],
                state=AgentState.ERROR,
            )

    def _extract_game_state(
        self,
        entities_detection: Optional[List[Detection]] = None,
        ui_detection: Optional[List[Detection]] = None,
        frame=None,
        capture_card_descriptions: bool = True,
    ) -> Dict[str, Any]:
        """Compatibility wrapper delegating to game state extractor."""
        snapshot = self.game_state_extractor.capture_state(
            frame=frame,
            entities_detection=entities_detection,
            ui_detection=ui_detection,
            capture_card_descriptions=capture_card_descriptions,
        )

        return snapshot or {
            'timestamp': time.time(),
            'entities_raw': entities_detection or [],
            'ui_elements_raw': ui_detection or [],
            'cards': [],
            'jokers': [],
            'ui_buttons': [],
            'game_phase': 'unknown',
            'card_descriptions': [],
            'ui_text_elements': [],
            'ui_text_values': {},
        }

    def _create_analysis_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create prompt for game state analysis."""
        cards_info = []

        for i, card in enumerate(game_state.get('cards', [])):
            base_line = (
                f'Card {i}: {card["class_name"]} (confidence: {card["confidence"]:.2f})'
            )

            parsed = card.get('parsed_description') or {}
            if parsed.get('english_name'):
                descriptor = parsed['english_name']
                short_code = parsed.get('short_code')
                if short_code:
                    descriptor += f' [{short_code}]'
                cards_info.append(f'{base_line} -> {descriptor}')
                continue

            if card.get('description_text'):
                desc_text = card['description_text']
                if len(desc_text) > 100:
                    desc_text = desc_text[:100] + '...'
                cards_info.append(f'{base_line} -> {desc_text}')
            else:
                cards_info.append(base_line)

        jokers_info = []
        for joker in game_state.get('jokers', []):
            jokers_info.append(
                f'Joker: {joker["class_name"]} (confidence: {joker["confidence"]:.2f})'
            )

        buttons_info = []
        for button in game_state.get('ui_buttons', []):
            buttons_info.append(f'Button: {button["class_name"]}')

        ui_text_info = []
        for ui_item in game_state.get('ui_text_elements', []):
            value_text = ui_item.get('text', '')
            value_compact = ' '.join(value_text.split()) if value_text else '(no text)'
            confidence = ui_item.get('ocr_confidence')
            if confidence is not None and confidence > 0:
                ui_text_info.append(
                    f'{ui_item["class_name"]}: {value_compact} (OCR {confidence:.2f})'
                )
            else:
                ui_text_info.append(f'{ui_item["class_name"]}: {value_compact}')

        # Add OCR capture status
        ocr_status = ''
        if game_state.get('cards'):
            captured_count = sum(
                1 for card in game_state['cards'] if card.get('description_detected')
            )
            total_cards = len(game_state['cards'])
            ocr_status = (
                f'\nCARD DESCRIPTIONS CAPTURED: {captured_count}/{total_cards} cards'
            )

        poker_objectives = (
            '\nPOKER OBJECTIVES:\n'
            '- Form the strongest five-card poker hand from the detected cards.\n'
            '- Favor high-ranking combinations (pairs, straights, flushes, full houses, etc.).\n'
            '- Discard low-value cards that do not contribute to potential strong hands.'
        )

        return f"""You are playing a game called Balatro, a game borrowed the concept of Texas Hold'em Poker and enhanced the gameplay with rogue-like level setup, and many different joker cards to manipulate the game rules.
Most strategy comes from understanding the Texas Hold'em poker rules and making optimal plays based on the current hand and game phase.

In Texas Hold'em, the poker hand rankings from highest to lowest are:

1. Royal Flush
2. Straight Flush
3. Four of a Kind
4. Full House
5. Flush
6. Straight
7. Three of a Kind
8. Two Pair
9. One Pair
10. High Card

It's the same in Balatro, but instead of gaming with other opponents, you are playing against the game system to achieve the highest score possible.
The total play card stack is default to 52 without joker cards, you may purchase more playing cards later on the progressing the game.
During the play, you can play cards or discard cards to optimize your hand, once played, new numbers of cards you played will be drawn from the card stack to refill your hand to the maximum hand size (default to 5 cards).
When discarding cards, you will not be able to draw new cards to refill your hand too.
Your goal is to maximize your score by forming the best possible poker hands using the cards in your hand, while strategically managing joker cards that can modify game rules.

Here is your current known game state:

Current cards in ({len(game_state.get('cards', []))}) cards) hand:
{chr(10).join(cards_info) if cards_info else 'No cards detected'}{ocr_status}

Current Joker cards ({len(game_state.get('jokers', []))} active) enabled:
{chr(10).join(jokers_info) if jokers_info else 'No jokers detected'}

UI elements you can interact with ({len(game_state.get('ui_buttons', []))} buttons):
{chr(10).join(buttons_info) if buttons_info else 'No buttons detected'}

Dynamic UI values ({len(game_state.get('ui_text_elements', []))} tracked):
{chr(10).join(ui_text_info) if ui_text_info else 'No dynamic UI text detected'}

GAME PHASE: {game_state.get('game_phase', 'unknown')}

Based on the card descriptions and game state, make the optimal strategic decision:
{poker_objectives}

ACTION INSTRUCTIONS:
1. Analyze the cards listed above (Card 0, Card 1, Card 2, etc.) with their descriptions
2. Identify the best poker hand you can form from these cards
3. Choose ONE action:

   a) If you have a strong playable hand:
      - Use play_cards(indices=[...]) with the indices of cards to play
      - Example: play_cards(indices=[0, 1, 2, 3, 4]) to play first 5 cards
      - Example: play_cards(indices=[0, 2, 4, 6, 7]) to play specific cards

   b) If you need better cards:
      - Use discard_cards(indices=[...]) with the indices of cards to discard
      - Example: discard_cards(indices=[5, 6, 7]) to discard last 3 cards
      - Example: discard_cards(indices=[1, 3]) to discard specific unwanted cards

   c) If you need to interact with UI:
      - Use click_button(button_type='...') for UI actions

Remember:
- Cards are indexed from 0 (Card 0 is the leftmost)
- You can only play OR discard in one action, not both
- Provide clear reasoning for your choice

Execute the best action immediately and explain your strategic reasoning."""

    def _create_decision_prompt_legacy(self, game_state: Dict[str, Any]) -> str:
        """Legacy planning prompt - replaced by integrated decision making."""
        cards_count = len(game_state.get('cards', []))

        # This method is kept for backward compatibility but not used
        position_example = 'Example position arrays:\n'
        position_example += (
            '- Play first 3 cards: [1, 1, 1' + ', 0' * max(0, cards_count - 3) + ']\n'
        )
        position_example += (
            '- Discard last 2 cards: [0' * max(0, cards_count - 2) + ', -1, -1]\n'
        )
        position_example += (
            '- Play cards 0 and 2: [1, 0, 1' + ', 0' * max(0, cards_count - 3) + ']'
        )

        return f"""Based on the game state analysis, plan your next strategic action.

CURRENT SITUATION:
- Hand size: {cards_count} cards
- Game phase: {game_state.get('game_phase', 'unknown')}
- Available buttons: {[btn['class_name'] for btn in game_state.get('ui_buttons', [])]}

STRATEGIC OPTIONS:
1. **Play cards**: Use position array with 1s for cards to play
2. **Discard cards**: Use position array with -1s for cards to discard
3. **Hover card**: Examine specific card details first
4. **Click button**: Use available UI buttons

{position_example}

Choose the action that maximizes your score potential. Consider:
- Poker hand strength and scoring
- Joker synergies and multipliers
- Long-term strategy vs immediate gains
- Risk vs reward of different plays

Make your strategic decision and call the appropriate function."""

    def _should_stop(self, result: AgentResult) -> bool:
        """Determine if agent should continue or stop."""
        # Continue playing unless there's a critical error
        if not result.success:
            return True

        # Stop if we've reached a game over state
        if result.metadata and result.metadata.get('game_over'):
            return True

        # Stop if no more valid actions are available
        if result.metadata and result.metadata.get('no_actions_available'):
            return True

        return False

    def get_game_history(self) -> List[Dict[str, Any]]:
        """Get the complete game action history."""
        return self.game_history.copy()

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current game state."""
        return self.current_game_state.copy()

    def reset_game_state(self):
        """Reset game state tracking for a new game."""
        self.current_game_state = {}
        self.game_history = []
        logger.info('Game state reset')
