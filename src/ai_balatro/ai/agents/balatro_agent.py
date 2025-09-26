"""Balatro game playing agent with LLM reasoning and strategic planning."""

import time
from typing import Dict, Any, List, Optional, Tuple

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from ..providers.base import LLMProvider
from ..actions.executor import ActionExecutor
from ..actions.schemas import GAME_ACTIONS
from ...core.multi_yolo_detector import MultiYOLODetector
from ...core.screen_capture import ScreenCapture
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
        **kwargs
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
            screen_capture=screen_capture,
            multi_detector=self.multi_detector
        )
        self.action_executor.initialize()

        # Game state tracking
        self.current_game_state: Dict[str, Any] = {}
        self.game_history: List[Dict[str, Any]] = []

        logger.info(f"BalatroReasoningAgent '{name}' initialized")

    def run(self, context: AgentContext) -> AgentResult:
        """Run a complete reasoning cycle with integrated decision making."""
        logger.info(f"Running {self.name} - integrated decision cycle")

        try:
            # Single integrated analysis and decision phase
            result = self.analyze_situation(context)

            if result.success and result.action:
                # Execute the decided action
                execution_result = self.execute_action(result.action, result.context)
                return execution_result
            elif result.success:
                # Analysis successful but no action decided
                return result
            else:
                # Analysis failed
                return result

        except Exception as e:
            logger.error(f"Integrated run cycle failed: {e}")
            return AgentResult(
                success=False,
                errors=[f"Run cycle failed: {e}"],
                state=AgentState.ERROR
            )

    def _get_system_message(self) -> str:
        """Get Balatro-specific system message."""
        return """You are a Balatro expert AI agent. You analyze game states and make immediate strategic decisions to maximize scores.

Your capabilities:
- Analyze detected cards with full OCR text descriptions
- Understand poker hands and scoring mechanics
- Make strategic card plays and discards
- Execute actions using position-based arrays

Available actions:
- select_cards_by_position: Use position array [1,1,0,0] for play, [-1,-1,0,0] for discard
- click_button: Press UI buttons (play, discard, skip, shop, next)

Make immediate, optimal decisions based on the complete card information provided. No additional information gathering needed."""

    def analyze_situation(self, context: AgentContext) -> AgentResult:
        """Analyze current Balatro game state and make immediate strategic decision."""
        try:
            logger.info("Analyzing Balatro game situation and planning action...")

            # Capture current screen
            frame = self.screen_capture.capture_once()
            if frame is None:
                return AgentResult(
                    success=False,
                    errors=["Failed to capture screen"],
                    state=AgentState.ERROR
                )

            # Detect entities using multi-YOLO system
            entities_detection = self.multi_detector.detect_entities(frame)
            ui_detection = self.multi_detector.detect_ui(frame)

            # Extract game state information WITH card descriptions
            game_state = self._extract_game_state(entities_detection, ui_detection, frame, capture_card_descriptions=True)

            # Update context with current state
            context.game_state.update(game_state)
            self.current_game_state = game_state

            # Create strategic decision prompt
            decision_prompt = self._create_analysis_prompt(game_state)

            # Query LLM with function calling for immediate action decision
            result = self._llm_query(
                decision_prompt,
                use_functions=True,
                functions=GAME_ACTIONS
            )

            if result.success:
                logger.info(f"Strategic decision: {result.reasoning}")
                return AgentResult(
                    success=True,
                    action=result.action,  # Action is planned immediately
                    reasoning=result.reasoning,
                    context=context,
                    metadata={
                        'game_state': game_state,
                        'entities_count': len(entities_detection),
                        'ui_elements_count': len(ui_detection)
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    errors=result.errors,
                    state=AgentState.ERROR
                )

        except Exception as e:
            logger.error(f"Failed to analyze situation and plan action: {e}")
            return AgentResult(
                success=False,
                errors=[f"Analysis/planning failed: {e}"],
                state=AgentState.ERROR
            )

    def plan_action(self, context: AgentContext) -> AgentResult:
        """Plan strategic action - now integrated into analyze_situation for efficiency."""
        logger.info("Action planning is now integrated into analyze_situation for efficiency")

        # Check if action was already planned during analysis
        if hasattr(context, 'planned_action') and context.planned_action:
            return AgentResult(
                success=True,
                action=context.planned_action,
                reasoning="Action already planned during analysis phase",
                context=context,
                metadata={'reused_planned_action': True}
            )

        # Fallback: re-run analysis if no action planned
        return self.analyze_situation(context)

    def execute_action(self, action: Dict[str, Any], context: AgentContext) -> AgentResult:
        """Execute the planned action using the action executor."""
        try:
            logger.info(f"Executing action: {action}")

            # Execute action through action executor
            execution_result = self.action_executor.process({
                'function_call': action
            })

            # Record action in game history
            action_record = {
                'timestamp': time.time(),
                'action': action,
                'game_state_before': self.current_game_state.copy(),
                'success': execution_result.success,
                'errors': execution_result.errors if not execution_result.success else []
            }
            self.game_history.append(action_record)

            if execution_result.success:
                logger.info("Action executed successfully")
                return AgentResult(
                    success=True,
                    action=action,
                    reasoning=f"Successfully executed {action.get('name', 'unknown action')}",
                    context=context,
                    metadata={
                        'execution_data': execution_result.data,
                        'execution_time': time.time()
                    }
                )
            else:
                logger.error(f"Action execution failed: {execution_result.errors}")
                return AgentResult(
                    success=False,
                    action=action,
                    errors=execution_result.errors,
                    state=AgentState.ERROR
                )

        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return AgentResult(
                success=False,
                action=action,
                errors=[f"Execution failed: {e}"],
                state=AgentState.ERROR
            )

    def _extract_game_state(
        self,
        entities_detection: List,
        ui_detection: List,
        frame=None,
        capture_card_descriptions: bool = True
    ) -> Dict[str, Any]:
        """Extract structured game state from detection results."""
        game_state = {
            'timestamp': time.time(),
            'entities_raw': entities_detection,
            'ui_elements_raw': ui_detection,
            'cards': [],
            'jokers': [],
            'ui_buttons': [],
            'game_phase': 'unknown',
            'card_descriptions': []
        }

        # Process entity detections
        for detection in entities_detection:
            if 'card' in detection.class_name.lower():
                x1, y1, x2, y2 = detection.bbox
                game_state['cards'].append({
                    'index': len(game_state['cards']),
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'position': [x1, y1, x2, y2],  # bbox format
                    'center': detection.center,
                    'width': detection.width,
                    'height': detection.height
                })
            elif 'joker' in detection.class_name.lower():
                x1, y1, x2, y2 = detection.bbox
                game_state['jokers'].append({
                    'index': len(game_state['jokers']),
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'position': [x1, y1, x2, y2],  # bbox format
                    'center': detection.center,
                    'width': detection.width,
                    'height': detection.height
                })

        # Process UI detections
        for detection in ui_detection:
            if 'button' in detection.class_name.lower():
                x1, y1, x2, y2 = detection.bbox
                game_state['ui_buttons'].append({
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'position': [x1, y1, x2, y2],  # bbox format
                    'center': detection.center,
                    'width': detection.width,
                    'height': detection.height
                })

        # Sort cards by position (left to right)
        game_state['cards'].sort(key=lambda c: c['position'][0])

        # Re-index cards after sorting
        for i, card in enumerate(game_state['cards']):
            card['index'] = i

        # Determine game phase based on available buttons
        button_types = [btn['class_name'].lower() for btn in game_state['ui_buttons']]
        if any('play' in btn for btn in button_types):
            game_state['game_phase'] = 'playing'
        elif any('shop' in btn for btn in button_types):
            game_state['game_phase'] = 'shop'
        elif any('next' in btn for btn in button_types):
            game_state['game_phase'] = 'transition'

        # Capture card descriptions if requested and cards are available
        if capture_card_descriptions and game_state['cards'] and game_state['game_phase'] == 'playing':
            logger.info("Capturing card descriptions for enhanced game state analysis...")
            try:
                combined_detections = list(entities_detection) + list(ui_detection)
                card_descriptions = self.action_executor.card_engine.get_card_descriptions_from_frame(
                    frame,
                    combined_detections,
                    auto_hover_missing=True,
                    save_debug_images=False
                )
                game_state['card_descriptions'] = card_descriptions
                logger.info(f"Successfully captured descriptions for {len(card_descriptions)} cards")
            except Exception as e:
                logger.warning(f"Failed to capture card descriptions: {e}")
                game_state['card_descriptions'] = []

        return game_state

    def _create_analysis_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create prompt for game state analysis."""
        cards_info = []
        card_descriptions = game_state.get('card_descriptions', [])

        # Enhanced card info with OCR descriptions
        for i, card in enumerate(game_state.get('cards', [])):
            card_line = f"Card {i}: {card['class_name']} (confidence: {card['confidence']:.2f})"

            # Add OCR description if available
            if i < len(card_descriptions) and card_descriptions[i].get('description_detected'):
                desc_text = card_descriptions[i]['description_text']
                if desc_text:
                    # Truncate long descriptions
                    if len(desc_text) > 100:
                        desc_text = desc_text[:100] + "..."
                    card_line += f" - {desc_text}"

            cards_info.append(card_line)

        jokers_info = []
        for joker in game_state.get('jokers', []):
            jokers_info.append(f"Joker: {joker['class_name']} (confidence: {joker['confidence']:.2f})")

        buttons_info = []
        for button in game_state.get('ui_buttons', []):
            buttons_info.append(f"Button: {button['class_name']}")

        # Add OCR capture status
        ocr_status = ""
        if card_descriptions:
            captured_count = sum(1 for desc in card_descriptions if desc.get('description_detected'))
            ocr_status = f"\nCARD DESCRIPTIONS CAPTURED: {captured_count}/{len(card_descriptions)} cards"

        return f"""Make an immediate strategic decision for the current Balatro game state:

CURRENT HAND ({len(game_state.get('cards', []))}) cards):
{chr(10).join(cards_info) if cards_info else "No cards detected"}{ocr_status}

JOKERS ({len(game_state.get('jokers', []))} active):
{chr(10).join(jokers_info) if jokers_info else "No jokers detected"}

UI ELEMENTS:
{chr(10).join(buttons_info) if buttons_info else "No buttons detected"}

GAME PHASE: {game_state.get('game_phase', 'unknown')}

Based on the card descriptions and game state, make the optimal strategic decision:
1. Identify the best poker hand you can form
2. If in playing phase, select cards to play using select_cards_by_position with optimal positions array
3. If no good hand available, select cards to discard
4. Consider joker effects when making decisions

Execute the best action immediately using the available functions."""

    def _create_decision_prompt_legacy(self, game_state: Dict[str, Any]) -> str:
        """Legacy planning prompt - replaced by integrated decision making."""
        cards_count = len(game_state.get('cards', []))

        # This method is kept for backward compatibility but not used
        position_example = "Example position arrays:\n"
        position_example += f"- Play first 3 cards: [1, 1, 1" + ", 0" * max(0, cards_count - 3) + "]\n"
        position_example += f"- Discard last 2 cards: [0" * max(0, cards_count - 2) + ", -1, -1]\n"
        position_example += f"- Play cards 0 and 2: [1, 0, 1" + ", 0" * max(0, cards_count - 3) + "]"

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
        logger.info("Game state reset")
