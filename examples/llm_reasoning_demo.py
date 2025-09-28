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
        print('🤖 Initializing LLM Reasoning Demo...')
        print('=' * 60)

        try:
            # 1. Initialize LLM Provider
            print('\n1. Setting up OpenRouter LLM Provider...')
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

            if self.llm_provider.initialize():
                print(
                    f'   ✅ LLM Provider initialized: {self.llm_provider.config.model_name}'
                )
            else:
                print('   ❌ Failed to initialize LLM provider')
                return False

            # 2. Initialize Computer Vision
            print('\n2. Setting up Computer Vision System...')
            self.multi_detector = MultiYOLODetector()
            available_models = self.multi_detector.get_available_models()
            print(f'   ✅ Multi-YOLO detector ready: {available_models}')

            model_info = self.multi_detector.get_model_info()
            for model_name, info in model_info.items():
                if info['available']:
                    print(
                        f'     - {model_name.upper()}: {info["classes_count"]} classes'
                    )

            # 3. Initialize Screen Capture
            print('\n3. Setting up Screen Capture...')
            self.screen_capture = ScreenCapture()
            print('   ✅ Screen capture ready')

            # 4. Initialize Reasoning Agent
            print('\n4. Creating Balatro Reasoning Agent...')
            self.reasoning_agent = BalatroReasoningAgent(
                name='BalatroMaster',
                llm_provider=self.llm_provider,
                screen_capture=self.screen_capture,
                multi_detector=self.multi_detector,
                max_iterations=5,  # Limit iterations for demo
            )
            print('   ✅ Reasoning agent created')

            print('\n🚀 All components initialized successfully!')
            return True

        except Exception as e:
            print(f'   ❌ Initialization failed: {e}')
            logger.error(f'Component initialization failed: {e}')
            return False

    def setup_game_window(self) -> bool:
        """Setup game window capture region."""
        print('\n' + '=' * 60)
        print('🎮 Game Window Setup')
        print('=' * 60)

        print("\nPlease position your Balatro game window so it's clearly visible.")
        print('The AI will analyze the game state using computer vision.')

        input('\nPress Enter when ready to select the game region...')

        if not self.screen_capture.select_region_interactive():
            print('   ❌ Failed to select game region')
            print('   💡 Make sure Balatro is running and visible')
            return False

        print('   ✅ Game region selected successfully')

        # Test capture
        print('\n🔍 Testing screen capture...')
        frame = self.screen_capture.capture_once()
        if frame is None:
            print('   ❌ Failed to capture screen')
            return False

        print(f'   ✅ Screen capture working: {frame.shape}')
        return True

    def run_reasoning_demo(self):
        """Run the main LLM reasoning demonstration."""
        print('\n' + '=' * 60)
        print('🧠 LLM Strategic Reasoning Demo')
        print('=' * 60)

        try:
            demo_modes = [
                ('Single Turn Analysis', self._demo_single_turn),
                ('Multi-Turn Strategic Play', self._demo_multi_turn),
                ('Interactive Reasoning Session', self._demo_interactive),
                ('Reasoning History Analysis', self._demo_history),
            ]

            while True:
                print('\n🎯 Demo Modes:')
                for i, (name, _) in enumerate(demo_modes, 1):
                    print(f'   {i}. {name}')
                print('   5. Exit Demo')

                try:
                    choice = input('\nSelect demo mode (1-5): ').strip()

                    if choice == '5':
                        print('\n👋 Exiting LLM Reasoning Demo')
                        break

                    mode_idx = int(choice) - 1
                    if 0 <= mode_idx < len(demo_modes):
                        mode_name, mode_func = demo_modes[mode_idx]
                        print(f'\n🚀 Starting: {mode_name}')
                        print('-' * 40)
                        mode_func()
                    else:
                        print('   ❌ Invalid selection')

                except ValueError:
                    print('   ❌ Please enter a number')
                except KeyboardInterrupt:
                    print('\n\n👋 Demo interrupted')
                    break

        except Exception as e:
            print(f'   ❌ Demo failed: {e}')
            logger.error(f'Demo execution failed: {e}')

    def _demo_single_turn(self):
        """Demonstrate single turn analysis and action planning."""
        print('🔍 Single Turn Analysis - The AI will:')
        print('   1. Capture current game state')
        print('   2. Analyze cards and game situation')
        print('   3. Plan strategic action')
        print('   4. Execute the planned action')

        input('\nPress Enter to start analysis...')

        try:
            # Create initial context
            context = AgentContext(session_id='single_turn_demo')

            # Run analysis phase
            print('\n📊 Phase 1: Analyzing Game Situation...')
            analysis_result = self.reasoning_agent.analyze_situation(context)

            if analysis_result.success:
                print('✅ Analysis Complete:')
                print(
                    f'   Game State: {len(analysis_result.metadata.get("game_state", {}).get("cards", []))} cards detected'
                )
                print(
                    f'   Entities: {analysis_result.metadata.get("entities_count", 0)} entities'
                )
                print(
                    f'   UI Elements: {analysis_result.metadata.get("ui_elements_count", 0)} elements'
                )
                print('\n🤔 AI Analysis:')
                print(f'   {analysis_result.reasoning}')
            else:
                print(f'❌ Analysis failed: {analysis_result.errors}')
                return

            # Run planning phase
            print('\n🎯 Phase 2: Strategic Action Planning...')
            planning_result = self.reasoning_agent.plan_action(analysis_result.context)

            if planning_result.success:
                print('✅ Planning Complete:')
                if planning_result.action:
                    print(
                        f'   Planned Action: {planning_result.action.get("name", "unknown")}'
                    )
                    print(
                        f'   Arguments: {planning_result.action.get("arguments", {})}'
                    )
                else:
                    print('   No specific action planned')
                print('\n🎲 AI Strategic Reasoning:')
                print(f'   {planning_result.reasoning}')
            else:
                print(f'❌ Planning failed: {planning_result.errors}')
                return

            # Ask for execution confirmation
            if planning_result.action:
                execute = (
                    input('\n🎮 Execute the planned action? (y/N): ').strip().lower()
                )
                if execute in ['y', 'yes']:
                    print('\n⚡ Phase 3: Executing Action...')
                    execution_result = self.reasoning_agent.execute_action(
                        planning_result.action, planning_result.context
                    )

                    if execution_result.success:
                        print('✅ Action executed successfully!')
                        print(f'   Result: {execution_result.reasoning}')
                    else:
                        print(f'❌ Execution failed: {execution_result.errors}')
                else:
                    print('⏸️  Action execution skipped')

        except Exception as e:
            print(f'❌ Single turn demo failed: {e}')
            logger.error(f'Single turn demo error: {e}')

    def _demo_multi_turn(self):
        """Demonstrate multi-turn strategic gameplay."""
        print('🎮 Multi-Turn Strategic Play - The AI will:')
        print('   1. Play multiple turns automatically')
        print('   2. Build strategic decisions over time')
        print('   3. Learn from previous actions')
        print('   4. Adapt strategy based on outcomes')

        num_turns = input('\nNumber of turns to play (1-10, default 3): ').strip()
        try:
            num_turns = int(num_turns) if num_turns else 3
            num_turns = min(max(num_turns, 1), 10)  # Clamp to 1-10
        except ValueError:
            num_turns = 3

        print(f'\n🎯 Playing {num_turns} strategic turns...')

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
            print(f'❌ Multi-turn demo failed: {e}')
            logger.error(f'Multi-turn demo error: {e}')

    def _demo_interactive(self):
        """Demonstrate interactive reasoning session."""
        print('💬 Interactive Reasoning Session')
        print('   - Ask questions about the current game state')
        print('   - Get AI analysis and suggestions')
        print('   - Execute actions manually or automatically')

        try:
            # Capture and analyze current state
            context = AgentContext(session_id='interactive_demo')
            analysis_result = self.reasoning_agent.analyze_situation(context)

            if not analysis_result.success:
                print(f'❌ Failed to analyze current state: {analysis_result.errors}')
                return

            print('\n🎮 Current Game State Analyzed:')
            game_state = analysis_result.metadata.get('game_state', {})
            print(f'   Cards in hand: {len(game_state.get("cards", []))}')
            print(f'   Active jokers: {len(game_state.get("jokers", []))}')
            print(f'   Game phase: {game_state.get("game_phase", "unknown")}')

            print('\n🤔 AI Initial Analysis:')
            print(f'   {analysis_result.reasoning}')

            # Interactive Q&A session
            print("\n💭 Interactive Session (type 'exit' to finish):")
            while True:
                try:
                    question = input(
                        '\n❓ Ask about strategy, cards, or actions: '
                    ).strip()

                    if question.lower() in ['exit', 'quit', 'done']:
                        print('   👋 Interactive session ended')
                        break

                    if not question:
                        continue

                    # Create contextual prompt
                    interactive_prompt = f"""Based on the current Balatro game state I analyzed, please answer this question:

Question: {question}

Current game context:
- Cards in hand: {len(game_state.get('cards', []))}
- Jokers active: {len(game_state.get('jokers', []))}
- Game phase: {game_state.get('game_phase', 'unknown')}

Provide a helpful, strategic answer based on Balatro game mechanics."""

                    # Query LLM for interactive response
                    response = self.reasoning_agent._llm_query(interactive_prompt)

                    if response.success:
                        print('\n🤖 AI Response:')
                        print(f'   {response.reasoning}')
                    else:
                        print(f'   ❌ Failed to get response: {response.errors}')

                except KeyboardInterrupt:
                    print('\n   👋 Interactive session interrupted')
                    break

        except Exception as e:
            print(f'❌ Interactive demo failed: {e}')
            logger.error(f'Interactive demo error: {e}')

    def _demo_history(self):
        """Analyze and display reasoning history."""
        print('📊 Reasoning History Analysis')
        print('   - Review past AI decisions')
        print('   - Analyze strategic patterns')
        print('   - Show learning progression')

        try:
            history = self.reasoning_agent.get_game_history()

            if not history:
                print('\n📝 No game history available yet.')
                print('   Play a few turns first using the Multi-Turn demo.')
                return

            print(f'\n📈 Analysis of {len(history)} recorded actions:')

            # Success rate analysis
            successful = sum(1 for action in history if action['success'])
            success_rate = (successful / len(history)) * 100 if history else 0
            print(f'   Success Rate: {success_rate:.1f}% ({successful}/{len(history)})')

            # Action type analysis
            action_types = {}
            for action in history:
                action_name = action['action'].get('name', 'unknown')
                action_types[action_name] = action_types.get(action_name, 0) + 1

            print('\n🎯 Action Distribution:')
            for action_type, count in sorted(
                action_types.items(), key=lambda x: x[1], reverse=True
            ):
                print(f'   {action_type}: {count} times')

            # Recent action details
            print('\n🕒 Recent Actions (last 5):')
            for i, action in enumerate(history[-5:], max(1, len(history) - 4)):
                status = '✅' if action['success'] else '❌'
                timestamp = time.strftime(
                    '%H:%M:%S', time.localtime(action['timestamp'])
                )
                action_name = action['action'].get('name', 'unknown')
                args = action['action'].get('arguments', {})

                print(f'   {i}. {status} [{timestamp}] {action_name}')
                if args and isinstance(args, dict):
                    for key, value in args.items():
                        print(f'      {key}: {value}')

            # Strategic insights
            print('\n🧠 Strategic Insights:')

            # Position analysis for card actions
            position_actions = [
                a for a in history if 'positions' in a['action'].get('arguments', {})
            ]
            if position_actions:
                avg_play_count = sum(
                    sum(1 for p in a['action']['arguments']['positions'] if p > 0)
                    for a in position_actions
                ) / len(position_actions)
                avg_discard_count = sum(
                    sum(1 for p in a['action']['arguments']['positions'] if p < 0)
                    for a in position_actions
                ) / len(position_actions)

                print(f'   Average cards played per turn: {avg_play_count:.1f}')
                print(f'   Average cards discarded per turn: {avg_discard_count:.1f}')

            # Error analysis
            failed_actions = [a for a in history if not a['success']]
            if failed_actions:
                print('\n⚠️  Failed Action Analysis:')
                error_types = {}
                for action in failed_actions:
                    for error in action.get('errors', []):
                        error_types[error] = error_types.get(error, 0) + 1

                for error, count in sorted(
                    error_types.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f'   {error}: {count} times')

        except Exception as e:
            print(f'❌ History analysis failed: {e}')
            logger.error(f'History demo error: {e}')

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.llm_provider:
                self.llm_provider.shutdown()
                print('✅ LLM provider closed')
        except KeyboardInterrupt:
            print('\n⚡ Cleanup interrupted, forcing exit...')
        except Exception as e:
            print(f'⚠️  Cleanup error (non-critical): {e}')
            logger.error(f'Cleanup error: {e}')


def main():
    """Main demo function."""
    demo = LLMReasoningDemo()

    try:
        # Initialize all components
        if not demo.initialize_components():
            return

        # Setup game window
        if not demo.setup_game_window():
            return

        # Run the main demo
        demo.run_reasoning_demo()

    except KeyboardInterrupt:
        print('\n\n👋 Demo interrupted by user')
    except Exception as e:
        print(f'\n❌ Demo failed: {e}')
        logger.error(f'Main demo error: {e}')
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
