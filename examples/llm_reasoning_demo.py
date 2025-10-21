#!/usr/bin/env python3
"""
LLM Reasoning Demo - Balatro AI with Strategic Planning

This demo showcases the integrated LLM reasoning system that combines:
- Computer vision (YOLO detection)
- Strategic AI reasoning (OpenRouter LLM)
- Action execution (UI automation)

Usage:
    python examples/llm_reasoning_demo.py

Requirements:
    - OPENROUTER_API_KEY environment variable
    - Balatro game running and visible
"""

import sys
import os
import time
from typing import Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_balatro.ai.providers.openrouter import OpenRouterProvider
from ai_balatro.ai.agents.balatro_agent import BalatroReasoningAgent, AgentContext
from ai_balatro.core.multi_yolo_detector import MultiYOLODetector
from ai_balatro.core.screen_capture import ScreenCapture
from ai_balatro.utils.logger import get_logger

logger = get_logger(__name__)


class LLMReasoningDemo:
    """Demo class for LLM-powered Balatro AI reasoning."""

    def __init__(self):
        """Initialize demo components."""
        self.llm_provider: Optional[OpenRouterProvider] = None
        self.screen_capture: Optional[ScreenCapture] = None
        self.multi_detector: Optional[MultiYOLODetector] = None
        self.reasoning_agent: Optional[BalatroReasoningAgent] = None

    def initialize_components(self) -> bool:
        """Initialize all AI components."""

        try:
            # 1. Initialize LLM Provider
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                print('   ❌ OPENROUTER_API_KEY environment variable not set')
                print('   💡 Please set your OpenRouter API key:')
                print("      export OPENROUTER_API_KEY='your-api-key-here'")
                return False

            # Get model from environment variable or use default
            model_name = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')

            self.llm_provider = OpenRouterProvider(
                model_name=model_name, timeout=60, max_retries=3
            )

            if not self.llm_provider.initialize():
                print('   ❌ Failed to initialize LLM provider')
                return False

            # 2. Initialize Computer Vision
            self.multi_detector = MultiYOLODetector()

            model_info = self.multi_detector.get_model_info()
            for model_name, info in model_info.items():
                if info['available']:
                    print(
                        f'     - {model_name.upper()}: {info["classes_count"]} classes'
                    )

            # 3. Initialize Screen Capture
            self.screen_capture = ScreenCapture()

            # 4. Initialize Reasoning Agent
            self.reasoning_agent = BalatroReasoningAgent(
                name='BalatroMaster',
                llm_provider=self.llm_provider,
                screen_capture=self.screen_capture,
                multi_detector=self.multi_detector,
                max_iterations=5,  # Limit iterations for demo
            )

            return True

        except Exception as e:
            print(f'   ❌ Initialization failed: {e}')
            logger.error(f'Component initialization failed: {e}')
            return False

    def setup_game_window(self) -> bool:
        """Setup game window capture region."""

        print("\nPlease position your Balatro game window so it's clearly visible.")
        print('The AI will analyze the game state using computer vision.')

        input('\nPress Enter when ready to select the game region...')

        if not self.screen_capture.select_region_interactive():
            print('   ❌ Failed to select game region')
            print('   💡 Make sure Balatro is running and visible')
            return False

        frame = self.screen_capture.capture_once()
        if frame is None:
            print('   ❌ Failed to capture screen')
            return False

        return True

    def run_reasoning_demo(self):
        """Run the main LLM reasoning demonstration."""
        try:
            self._demo_multi_turn()
        except Exception as e:
            logger.error(f'Demo execution failed: {e}')

    def _demo_multi_turn(self):
        """Demonstrate multi-turn strategic gameplay."""

        num_turns = 10

        try:
            # Reset agent state for clean demo
            self.reasoning_agent.reset_game_state()

            # Create context for multi-turn session
            context = AgentContext(session_id='multi_turn_demo')

            for turn in range(1, num_turns + 1):
                print('\n' + '=' * 40)
                print(f'🎯 TURN {turn}/{num_turns}')
                print('=' * 40)

                # Run complete agent cycle
                print('🤖 Running AI reasoning cycle...')

                # Update context with turn information
                context.metadata['current_turn'] = turn
                context.metadata['total_turns'] = num_turns

                result = self.reasoning_agent.run(context)

                if result.success:
                    print(f'✅ Turn {turn} completed successfully')
                    if result.action:
                        print(
                            f'   Action taken: {result.action.get("name", "unknown")}'
                        )
                    print(f'   AI Reasoning: {result.reasoning}')

                    # Update context for next turn
                    context = result.context or context

                    # Brief pause between turns
                    if turn < num_turns:
                        time.sleep(2)

                else:
                    print(f'❌ Turn {turn} failed: {result.errors}')
                    break

            # Show game history
            history = self.reasoning_agent.get_game_history()
            print('\n📊 Game Session Summary:')
            print(f'   Total actions taken: {len(history)}')
            successful_actions = sum(1 for action in history if action['success'])
            print(f'   Successful actions: {successful_actions}/{len(history)}')

            if history:
                print('\n📈 Action History:')
                for i, action in enumerate(
                    history[-3:], max(1, len(history) - 2)
                ):  # Show last 3
                    status = '✅' if action['success'] else '❌'
                    print(
                        f'   {i}. {status} {action["action"].get("name", "unknown")} '
                        f'({action["action"].get("arguments", {})})'
                    )

        except Exception as e:
            logger.error(f'Multi-turn demo error: {e}')

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.llm_provider:
                self.llm_provider.shutdown()
        except KeyboardInterrupt:
            logger.log('forcing exit...')
        except Exception as e:
            logger.error(f'cleanup error: {e}')


def main():
    """Main demo function."""
    demo = LLMReasoningDemo()

    try:
        if not demo.initialize_components():
            return
        if not demo.setup_game_window():
            return
        demo.run_reasoning_demo()

    except KeyboardInterrupt:
        logger.log('exit requested...')
    except Exception as e:
        logger.error(f'error: {e}')
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
