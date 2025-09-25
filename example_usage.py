"""Example usage of the agent framework."""

from src.ai_balatro.ai.providers import OpenRouterProvider
from src.ai_balatro.ai.templates import render_prompt, get_system_prompt
from src.ai_balatro.ai.agents import (
    BaseAgent,
    AgentContext,
    AgentResult,
    AgentOrchestrator,
)
from src.ai_balatro.ai.actions import GAME_ACTIONS


# Example implementation of a simple analyzer agent
class GameAnalyzerAgent(BaseAgent):
    """Agent that analyzes game state."""

    def __init__(self, llm_provider):
        super().__init__(name='GameAnalyzer', llm_provider=llm_provider)

    def _get_system_message(self) -> str:
        return get_system_prompt('balatro')

    def analyze_situation(self, context: AgentContext) -> AgentResult:
        """Analyze current game state."""
        prompt_context = {
            'game_state_summary': 'Cards detected, score visible, buttons available',
            'card_list': ['Card 1: Ace of Spades', 'Card 2: King of Hearts'],
        }

        prompt = render_prompt('game_state_analysis', prompt_context)

        return self._llm_query(prompt)

    def plan_action(self, context: AgentContext) -> AgentResult:
        """Plan next action."""
        prompt_context = {
            'available_cards': ['Ace of Spades', 'King of Hearts'],
            'objective': 'Score points by playing poker hands',
        }

        prompt = render_prompt('strategic_planning', prompt_context)

        return self._llm_query(prompt, use_functions=True, functions=GAME_ACTIONS)

    def execute_action(self, action: dict, context: AgentContext) -> AgentResult:
        """Execute action (placeholder)."""
        # In real implementation, this would call the actual game interface
        return AgentResult(
            success=True,
            action=action,
            reasoning=f'Executed action: {action.get("name", "unknown")}',
            metadata={'should_stop': True},  # Stop after one action for demo
        )


def main():
    """Example usage."""
    # Initialize LLM provider
    provider = OpenRouterProvider(model_name='anthropic/claude-3.5-sonnet')

    # Create agent
    analyzer = GameAnalyzerAgent(provider)

    # Initialize provider
    with provider:
        # Create initial context
        initial_context = AgentContext(
            session_id='demo_session', game_state={'cards_visible': 5, 'score': 1000}
        )

        # Run agent
        result = analyzer.run(initial_context)

        print(f'Agent execution result: {result.success}')
        if result.reasoning:
            print(f'Reasoning: {result.reasoning}')
        if result.action:
            print(f'Action: {result.action}')


def orchestrator_example():
    """Example of using multiple agents with orchestrator."""
    provider = OpenRouterProvider()

    # Create multiple agents
    analyzer = GameAnalyzerAgent(provider)
    # planner = GamePlannerAgent(provider)  # Would implement similarly
    # executor = ActionExecutorAgent(provider)  # Would implement similarly

    # Create orchestrator
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(analyzer, 0)
    # orchestrator.register_agent(planner, 1)
    # orchestrator.register_agent(executor, 2)

    with provider:
        # Run all agents in sequence
        results = orchestrator.run_agents()

        for i, result in enumerate(results):
            print(f'Agent {i + 1}: {result.success} - {result.reasoning}')


if __name__ == '__main__':
    # Set environment variable: export OPENROUTER_API_KEY="your-key"
    main()
    # orchestrator_example()
